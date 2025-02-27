/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/relax/transform/lambda_lift.cc
 * \brief Lift local functions into global functions.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/runtime/logging.h>

#include <iostream>
#include <vector>

namespace tvm {
namespace relax {

/* The goal of this class is to lift out any nested functions into top-level
 * functions.
 *
 * We will lift a function out into a global which takes the set of the free
 * vars and then return the new created function.
 */
class LambdaLifter : public ExprMutator {
 public:
  explicit LambdaLifter(const IRModule& module) : ExprMutator(module) { mod_ = module; }

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallNode* call_node) final {
    auto call = Downcast<Call>(ExprMutator::VisitExpr_(call_node));
    if (auto const* var = call_node->op.as<VarNode>()) {
      bool has_closure = HasClosure(GetRef<Var>(var));
      auto val = builder_->LookupBinding(GetRef<Var>(var));
      // Call "relax.invoke_closure" to invoke closure
      if (has_closure && val.as<CallNode>()) {
        Var clo_arg = GetRef<Var>(var);
        if (this->var_remap_.find(var->vid) != this->var_remap_.end()) {
          clo_arg = this->var_remap_.at(var->vid);
        }
        return Call(invoke_closure_op_, {clo_arg, Tuple(call_node->args)}, {},
                    {call_node->checked_type_});
      }
    }
    if (auto global_var_node = call_node->op.as<GlobalVarNode>()) {
      String rec_name = global_var_node->name_hint;
      auto global_var = GetRef<GlobalVar>(global_var_node);
      auto it = lambda_map_.find(global_var);
      if (it != lambda_map_.end()) {
        // flatten nested call, e.g. call(y)(x) -> call(x, y))
        Array<relay::Expr> new_args;
        for (const auto arg : call->args) {
          new_args.push_back(arg);
        }
        if (const auto* nest_call = it->second.as<CallNode>()) {
          for (const auto arg : nest_call->args) {
            new_args.push_back(arg);
          }
          return Call(nest_call->op, new_args, call_node->attrs, call_node->type_args);
        }
        return Call(it->second, call->args, call_node->attrs, call_node->type_args);
      }
    }
    return std::move(call);
  }

  Expr VisitExpr_(const FunctionNode* func_node) final {
    auto func = GetRef<Function>(func_node);

    // TODO(@yongwww): consider appending inner func name into the lifted func name
    String lift_func_name = "lifted_func_" + std::to_string(lift_func_num_++);
    auto global = GlobalVar(lift_func_name);
    Array<Var> captured_vars = FreeVars(func);
    recur_vars_ = CalledGlobalVars(func);
    auto all_global_vars = AllGlobalVars(func);

    Array<Var> typed_captured_vars;
    Map<Var, Expr> rebinding_map;
    for (auto free_var : captured_vars) {
      Var var = Var(free_var->name_hint(), NullOpt, free_var->checked_type_, free_var->span);
      var->shape_ = free_var->shape_;
      typed_captured_vars.push_back(var);
      rebinding_map.Set(free_var, var);
    }

    // recursive call
    if (!recur_vars_.empty()) {
      if (!captured_vars.empty()) {
        Array<Expr> fvs;
        for (auto fv : captured_vars) {
          fvs.push_back(fv);
        }
        lambda_map_.emplace(recur_vars_.back(), Call(global, fvs));
      } else {
        if (recur_vars_.size() > 0) {
          lambda_map_.emplace(recur_vars_.back(), global);
        }
      }
    }

    tvm::Array<Var> params;
    bool all_params_unchanged = true;
    for (Var param : func_node->params) {
      Var new_param = this->VisitVarDef(param);
      params.push_back(new_param);
      all_params_unchanged &= param.same_as(new_param);
    }

    Expr body = this->VisitWithNewScope(func_node->body);
    Expr visited_func;

    if (all_params_unchanged && body.same_as(func_node->body)) {
      visited_func = GetRef<Expr>(func_node);
    } else if (body->checked_type_.as<ObjectTypeNode>()) {
      // make_closure was introduced
      // TODO(@sslyu): Determine if we can fill in the return shape
      visited_func =
          Function(params, body, body->checked_type_, RuntimeDepShape(), func_node->attrs);
    } else {
      visited_func =
          Function(params, body, func_node->ret_type, func_node->ret_shape, func_node->attrs);
    }
    auto new_func = Downcast<Function>(visited_func);

    Function lifted_func;
    bool is_closure = IsClosure(captured_vars);
    if (!is_closure) {
      lifted_func = Function(
          /*params=*/new_func->params,
          /*body=*/new_func->body,
          /*ret_type=*/new_func->ret_type,
          /*ret_shape=*/new_func->ret_shape,
          /*attrs=*/new_func->attrs,
          /*span=*/new_func->span);
    } else {
      // Flatten the Closure
      std::vector<Var> closure_params;
      closure_params.reserve(func->params.size() + typed_captured_vars.size());
      for (size_t i = 0; i < func->params.size(); ++i) {
        closure_params.emplace_back(func->params[i]);
      }
      for (size_t i = 0; i < typed_captured_vars.size(); ++i) {
        closure_params.emplace_back(typed_captured_vars[i]);
      }

      lifted_func = Function(/*params=*/closure_params,
                             /*body=*/Bind(new_func->body, rebinding_map),
                             /*ret_type=*/new_func->ret_type,
                             /*ret_shape=*/new_func->ret_shape,
                             /*attrs=*/new_func->attrs,
                             /*span=*/func->span);

      Array<Type> param_types;
      for (Var param : closure_params) {
        CHECK(param->checked_type_.defined())
            << "relax.Function requires params to contain checked_type_";
        param_types.push_back(param->checked_type_);
      }
    }

    ICHECK(lifted_func.defined());

    // Add the lifted function to the module.
    builder_->UpdateFunction(global, lifted_func);
    UpdateType(global, lifted_func->checked_type());

    if (!is_closure) {
      return std::move(global);
    } else {
      // If we need to allocate a closure,
      // we pass the variables in its environment here.
      Array<Expr> fvs;
      for (auto fv : captured_vars) {
        fvs.push_back(fv);
      }
      // Call make_closure intrinsic
      return Call(make_closure_op_, {global, Tuple(fvs)}, {}, {});
    }
  }

  bool HasClosure(const Var& var) {
    auto val = builder_->LookupBinding(var);
    if (const auto* value = val.as<GlobalVarNode>()) {
      IRModule ctx_mod = builder_->GetContextIRModule();
      ICHECK(ctx_mod->functions.size() > 0);
      BaseFunc func = ctx_mod->Lookup(GetRef<GlobalVar>(value));
      if (const auto* func_node = func.as<FunctionNode>()) {
        if (const auto* call_node = func_node->body.as<CallNode>()) {
          if (call_node->op == make_closure_op_) {
            return true;
          }
        } else if (const auto* seq_expr_node = func_node->body.as<SeqExprNode>()) {
          // the return var points to a make_closure intrinsic
          if (const auto* var = seq_expr_node->body.as<VarNode>()) {
            return HasClosure(GetRef<Var>(var));
          }
        }
      }
    } else if (const auto* func_node = val.as<FunctionNode>()) {
      if (const auto* call_node = func_node->body.as<CallNode>()) {
        if (call_node->op == make_closure_op_) {
          return true;
        }
      }
    } else if (const auto* call_node = val.as<relax::CallNode>()) {
      // recursive call
      auto op = call_node->op;
      if (make_closure_op_ == op) {
        return true;
      }
      if (const auto* lv = op.as<VarNode>()) {
        return HasClosure(GetRef<Var>(lv));
      }
    }
    return false;
  }

  bool IsClosure(const Array<Var>& captured_vars) { return captured_vars.size() > 0; }

  IRModule Lift() {
    auto glob_funcs = mod_->functions;
    for (auto pair : glob_funcs) {
      if (auto* n = pair.second.as<FunctionNode>()) {
        auto func = GetRef<Function>(n);
        func = Function(func->params, VisitExpr(func->body), func->ret_type, func->ret_shape,
                        func->attrs);
        builder_->UpdateFunction(pair.first, func);
      }
    }
    return builder_->GetContextIRModule();
  }

 private:
  std::unordered_map<GlobalVar, Expr, ObjectPtrHash, ObjectPtrEqual> lambda_map_;
  Array<GlobalVar> recur_vars_;
  IRModule mod_;
  size_t lift_func_num_ = 0;
  /*! \brief Cache ops that would be used later to reduce lookup overhead. */
  const Op& make_closure_op_ = Op::Get("relax.make_closure");
  const Op& invoke_closure_op_ = Op::Get("relax.invoke_closure");
};

namespace transform {

Pass LambdaLift() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relax::LambdaLifter(m).Lift(); };
  return CreateModulePass(pass_func, 1, "LambdaLift", {});
}

TVM_REGISTER_GLOBAL("relax.transform.LambdaLift").set_body_typed(LambdaLift);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
