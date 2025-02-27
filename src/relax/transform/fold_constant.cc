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

#include <tvm/driver/driver_api.h>
#include <tvm/ir/function.h>
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>

#include "../../relay/transforms/pattern_utils.h"

namespace {
inline bool isPowerOfTwo(uint32_t x) { return x && (!(x & (x - 1))); }
}

namespace tvm {
namespace relax {

class ConstantFolder : public ExprMutator {
 public:
  explicit ConstantFolder(IRModule ctx_module) : ctx_module_(ctx_module) {}

 private:
  /*!
   * \brief Pattern match expr to a constant shape and get runtime shape tuple from it.
   * \return The runtime shape tuple, or nullopt if it is not a constant shape.
   */
  static Optional<runtime::ShapeTuple> MatchConstShape(const Expr& expr) {
    auto* shape = expr.as<ShapeExprNode>();
    if (!shape) return NullOpt;

    std::vector<int64_t> shape_values;
    for (const auto v : shape->values) {
      auto* ptr = v.as<IntImmNode>();
      if (!ptr) return NullOpt;
      shape_values.push_back(ptr->value);
    }
    return runtime::ShapeTuple(shape_values.begin(), shape_values.end());
  }

  /*!
   * \brief Pattern match op to constant array arguments.
   * \return The constant array arguments, or nullopt if match fails.
   */
  static Optional<Array<runtime::NDArray>> MatchConstArrayArgs(const Array<Expr>& args) {
    Array<runtime::NDArray> res;
    for (auto arg : args) {
      auto* ptr = arg.as<relax::ConstantNode>();
      if (!ptr) return NullOpt;
      res.push_back(ptr->data);
    }
    return res;
  }

  /*!
   * \brief Pattern match op to a TIR function and look it up.
   * \return The TIR function, or nullopt if pattern match fails.
   */
  Optional<tir::PrimFunc> MatchPrimFunc(const Expr& op) {
    if (auto* ptr = op.as<GlobalVarNode>()) {
      // NOTE: as check works for nullptr(returns null)
      Optional<BaseFunc> base_func = ctx_module_->functions.Get(GetRef<GlobalVar>(ptr));
      if (auto* pfunc = base_func.as<tir::PrimFuncNode>()) {
        return GetRef<tir::PrimFunc>(pfunc);
      }
    }
    return NullOpt;
  }

  /*!
   * \brief Get a cached build version of func
   * \return The cached func, nullopt if func cannot be built.
   */
  Optional<PackedFunc> GetCachedBuild(tir::PrimFunc func) {
    // TODO(tvm-team): consider another way of bulk extract and build PrimFunc once
    // would be helpful for future cases where PrimFunc recursively call into each other
    Target eval_cpu_target{"llvm"};

    auto it = func_build_cache_.find(func);
    if (it != func_build_cache_.end()) {
      return it->second;
    }
    Optional<PackedFunc> build_func = NullOpt;

    try {
      // Not all the primfunc can be directly built via llvm, for example, if a function is
      // already scheduled to only work on GPU, we will need to skip this in the const folder for
      // now
      // TODO(Hongyi): further check and narrow the scope of foldable function
      runtime::Module rt_module =
          build(LowerPrimFunc(func, "tir_function"), eval_cpu_target, eval_cpu_target);
      build_func = rt_module.GetFunction("tir_function");
    } catch (const tvm::Error& err) {
      // build failure may happen in which case we skip
      DLOG(WARNING) << "Build failure for function " << func << ", Error message: " << err.what();
    }
    func_build_cache_[func] = build_func;
    return build_func;
  }

  // Try constant evaluate the function call
  // if failed return NullOpt
  Optional<Expr> ConstEvaluateCallTIR(tir::PrimFunc tir_func, Array<runtime::NDArray> arr_args,
                                      runtime::ShapeTuple shape, DataType ret_type) {
    // obtain function from the cache.
    Optional<PackedFunc> func = GetCachedBuild(tir_func);
    if (!func) return NullOpt;

    // here the vector size has an additional + 1 because we need to put ret_tensor at the end
    std::vector<TVMValue> values(arr_args.size() + 1);
    std::vector<int> type_codes(arr_args.size() + 1);

    DLDevice cpu_dev = {DLDeviceType::kDLCPU, 0};
    runtime::NDArray ret_tensor = runtime::NDArray::Empty(shape, ret_type, cpu_dev);

    // avoid set rvalue ref which get de-allocated later, store args in a vector
    // where temp_args[i] are lvalue ref that is stable
    std::vector<runtime::NDArray> temp_args(arr_args.begin(), arr_args.end());

    size_t arg_offset = 0;
    for (; arg_offset < arr_args.size(); ++arg_offset) {
      runtime::TVMArgsSetter(values.data(), type_codes.data())(arg_offset, temp_args[arg_offset]);
    }
    // set return value
    runtime::TVMArgsSetter(values.data(), type_codes.data())(arg_offset++, ret_tensor);

    TVMRetValue ret;
    // invoke
    func.value().CallPacked(TVMArgs(values.data(), type_codes.data(), values.size()), &ret);
    return Constant(ret_tensor);
  }

  Expr VisitCallTIR(Call call) {
    // call_tir needs to have at least three arguments
    ICHECK_GE(call->args.size(), 3);
    Optional<tir::PrimFunc> func = MatchPrimFunc(call->args[0]);
    ICHECK(call->args[1].as<TupleNode>()) << "call_tir.args[1] must be Tuple";
    Optional<Array<runtime::NDArray>> arr_args =
        MatchConstArrayArgs(call->args[1].as<TupleNode>()->fields);
    Optional<runtime::ShapeTuple> shape = MatchConstShape(call->args[2]);
    bool output_not_tuple = call->type_args.size() == 1;
    // Pattern 0: call constant function, const argument with const shape.
    if (func && arr_args && shape && output_not_tuple) {
      DynTensorType ret_type = Downcast<DynTensorType>(call->checked_type());
      // value_or will return value if it is not null, otherwise return or
      return ConstEvaluateCallTIR(func.value(), arr_args.value(), shape.value(), ret_type->dtype)
          .value_or(call);
    }
    // TODO(hongyi): support const-fold tuple outputs
    return std::move(call);
  }

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallNode* call) final {
    Call post_call = Downcast<Call>(VisitExprPostOrder_(call));
    static const Op& call_tir_op = Op::Get("relax.call_tir");

    if (call->op.same_as(call_tir_op)) {
      return VisitCallTIR(post_call);
    }

    // Folds have all constants as arguments
    // Algebraic Identities have one tensor and one constant
    bool foldable = true, algebraic = false;
    for (auto &arg : call->args) {
      if (!tvm::relay::IsConstScalar(arg)) foldable = false;
      if (tvm::relay::IsConstScalar(arg)) algebraic = true;
    }

    /** 
     * Perform Constant Folding
     * NOTE: Assumes call->args is size 2
    */
    DataType dtype = DataType::Int(32);

    if (foldable) {
      uint32_t arg[] = {
        call->args[0].as<ConstantNode>()->toInt(),
        call->args[1].as<ConstantNode>()->toInt()
      };

      if (call->op == Op::Get("relax.add")) {
        return tvm::relay::MakeConstantScalar(dtype, arg[0] + arg[1]);
      }
      
      if (call->op == Op::Get("relax.multiply")) {
        return tvm::relay::MakeConstantScalar(dtype, arg[0] * arg[1]);
      }

      if (call->op == Op::Get("relax.left_shift")) {
        return tvm::relay::MakeConstantScalar(dtype, arg[0] << arg[1]);
      }

      if (call->op == Op::Get("relax.xor")) {
        return tvm::relay::MakeConstantScalar(dtype, arg[0] ^ arg[1]);
      }
      
      if (call->op == Op::Get("relax.or")) {
        return tvm::relay::MakeConstantScalar(dtype, arg[0] | arg[1]);
      }
      
      if (call->op == Op::Get("relax.divide")) {
        return tvm::relay::MakeConstantScalar(dtype, arg[0] / arg[1]);
      }

      if (call->op == Op::Get("relax.right_shift")) {
        return tvm::relay::MakeConstantScalar(dtype, arg[0] >> arg[1]);
      }
      
      if (call->op == Op::Get("relax.and")) {
        return tvm::relay::MakeConstantScalar(dtype, arg[0] & arg[1]);
      }
    }

    /**
     * Algebraic Identities 
    */
    if (algebraic) {
      uint32_t constant, tensor_idx;
      if (const ConstantNode *arg0 = call->args[0].as<ConstantNode>()) {
        constant = arg0->toInt();
        tensor_idx = 1;
      } else {
        constant = call->args[1].as<ConstantNode>()->toInt();
        tensor_idx = 0;
      }

      // c *///& 0 = 0
      if (constant == 0
        && (call->op == Op::Get("relax.multiply")
            || (call->op == Op::Get("relax.divide") && tensor_idx == 1)
            || call->op == Op::Get("relax.and"))) {

        return tvm::relay::MakeConstantScalar(dtype, 0);
      }

      // c *// 1 = c
      if (constant == 1
        && (call->op == Op::Get("relax.multiply")
            || (call->op == Op::Get("relax.divide") && tensor_idx == 0))) {

        return call->args[tensor_idx];
      }

      // c +/-/|/^/<</>> 0 = c
      if (constant == 0
        && (call->op == Op::Get("relax.add")
            || (call->op == Op::Get("relax.subtract") && tensor_idx == 0)
            || call->op == Op::Get("relax.or")
            || call->op == Op::Get("relax.xor")
            || (call->op == Op::Get("relax.left_shift") && tensor_idx == 0)
            || (call->op == Op::Get("relax.right_shift") && tensor_idx == 0))) {

        return call->args[tensor_idx];
      }

      // x * c -> leftshift x by log(c) if c is a multiple of 2
      if (isPowerOfTwo(constant)
        && call->op == Op::Get("relax.multiply")) {
        
        static const Op& op = Op::Get("relax.left_shift");
        return Call(op, {
            call->args[tensor_idx],
            tvm::relay::MakeConstantScalar(dtype, std::log2(constant))
          }, Attrs(), {}
        );
      }
    }
    return std::move(post_call);
  }

  Expr VisitExpr_(const DataflowVarNode* op) final {
    Optional<Expr> opt = LookupBinding(GetRef<Var>(op));
    // `as` check checks if opt is not null and is instance of constant
    if (opt.as<relax::ConstantNode>()) {
      return opt.value();
    }
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const VarNode* op) final {
    Optional<Expr> opt = LookupBinding(GetRef<Var>(op));
    // `as` check checks if opt is not null and is instance of constant
    if (opt.as<relax::ConstantNode>()) {
      return opt.value();
    }
    return ExprMutator::VisitExpr_(op);
  }


  // the context module to lookup functions
  IRModule ctx_module_;
  // cache for function build, via structural equality
  std::unordered_map<tir::PrimFunc, Optional<runtime::PackedFunc>, StructuralHash, StructuralEqual>
      func_build_cache_;
};

namespace transform {

Pass FoldConstant() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        ConstantFolder folder(m);
        return Downcast<Function>(folder(f));
      };
  return CreateFunctionPass(pass_func, 0, "FoldConstant", {});
}

TVM_REGISTER_GLOBAL("relax.transform.FoldConstant").set_body_typed(FoldConstant);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
