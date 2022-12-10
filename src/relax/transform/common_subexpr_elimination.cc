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
 *
 * \file eliminate_common_subexpr.cc
 * \brief Combine common subexpressions.
 *
 * This is an optimization pass that eliminates common subexpressions. During the pass, it tries
 * to replace an expression with a previously appeared expression with the same input and
 * attributes.
 */
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relax/transform.h>

#include <unordered_map>

namespace tvm {
namespace relax {

class CommonSubexprEliminator : public tvm::relay::MixedModeMutator {
 public:
  explicit CommonSubexprEliminator() {}

  Expr Rewrite_(const CallNode* call, const Expr& post) {
    static auto op_stateful = Op::GetAttrMap<bool>("TOpIsStateful");
    Expr new_expr = post;
    const CallNode* new_call = new_expr.as<CallNode>();
    ICHECK(new_call);
    const OpNode* op = new_call->op.as<OpNode>();
    StructuralEqual attrs_equal;

    if (new_call->args.size() == 0 || op == nullptr || op_stateful.get(GetRef<Op>(op), false)) {
      return new_expr;
    }

    auto it = expr_map_.find(new_call->op);
    if (it != expr_map_.end()) {
      for (const Expr& candidate_expr : it->second) {
        if (const CallNode* candidate = candidate_expr.as<CallNode>()) {
          bool is_equivalent = true;
          if (!attrs_equal(new_call->attrs, candidate->attrs)) {
            continue;
          }
          for (size_t i = 0; i < new_call->args.size(); i++) {
            if (!new_call->args[i].same_as(candidate->args[i]) &&
                !IsEqualScalar(new_call->args[i], candidate->args[i])) {
              is_equivalent = false;
              break;
            }
          }
          if (!is_equivalent) continue;
          return GetRef<Call>(candidate);
        }
      }
    }
    expr_map_[new_call->op].push_back(new_expr);
    return new_expr;
  }

  Expr Rewrite_(const TupleGetItemNode* op, const Expr& post) {
    Expr new_expr = post;
    const TupleGetItemNode* new_tuple_item = new_expr.as<TupleGetItemNode>();
    ICHECK(new_tuple_item);

    auto it = expr_map_.find(new_tuple_item->tuple);
    if (it != expr_map_.end()) {
      for (const Expr& candidate_expr : it->second) {
        if (const TupleGetItemNode* candidate = candidate_expr.as<TupleGetItemNode>()) {
          if (new_tuple_item->index == candidate->index) {
            return GetRef<Expr>(candidate);
          }
        }
      }
    }
    expr_map_[new_tuple_item->tuple].push_back(new_expr);
    return new_expr;
  }

  std::unordered_map<Expr, std::vector<Expr>, ObjectPtrHash, ObjectPtrEqual> expr_map_;

  inline bool IsEqualScalar(const Expr& a, const Expr& b) {
    const auto* constant_a = a.as<ConstantNode>();
    const auto* constant_b = b.as<ConstantNode>();
    if (!constant_a || !constant_b || !constant_a->is_scalar() || !constant_b->is_scalar()) {
      return false;
    }
    return tvm::StructuralEqual()(a, b);
}
};

Expr CommonSubexpressionElimination(const Expr& expr) {
  return CommonSubexprEliminator()(expr);
}

namespace transform {

Pass CommonSubexpressionElimination() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(CommonSubexpressionElimination(f));
      };
  return CreateFunctionPass(pass_func, 3, "CommonSubexpressionElimination", {"InferType"});
}

TVM_REGISTER_GLOBAL("relax.transform.CommonSubexpressionElimination")
    .set_body_typed(CommonSubexpressionElimination);

}  // namespace transform

}  // namespace relay
}  // namespace tvm


