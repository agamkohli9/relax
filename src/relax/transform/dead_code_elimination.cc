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

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

class Tracer : ExprVisitor {
  public:
    Tracer(IRModule mod_) : mod(mod_) {}

    void VisitExpr(const Expr &expr) final {
      // Stop visiting var that's already been visited at least twice
      std::cout << "Visiting: " << PrettyPrint(expr);
      if (++visit_counter[expr.get()] <= 2) {
        VisitExpr(expr);
      }
    }

    void VisitExpr_(const VarNode* var) final {
      std::cout << "Check var: " << var->name_hint();
    }

  private:
    IRModule mod;
    std::unordered_map<const Object*, size_t> visit_counter;
};

class DeadCodeEliminator : public ExprMutator {
 public:
  DeadCodeEliminator() {}

 private:
   
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const Expr &expr) {
    printf("debug\n");
    return expr;
  }
};

Expr DeadCodeElimination(const Expr& e) { return DeadCodeEliminator().VisitExpr(e); }

namespace transform {

Pass DeadCodeElimination() {
  const tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)> pass_func = 
  [=](IRModule mod, PassContext pc) {
    IRModule result({}, mod->type_definitions, mod->Imports(), mod->source_map, mod->attrs);
    DeadCodeEliminator dce;

    for (const auto& kv : mod->functions) {
      // Count uses of variable
      Tracer tracer(mod);
      tracer.VisitExpr(kv.second);

      // Eliminate dead code
      result->Add(kv.first, Downcast<Function>(dce.VisitExpr(kv.second)));
    }

    return result;
  };

  return tvm::transform::CreateModulePass(pass_func, 1, "DeadCodeElimination", {});
}

TVM_REGISTER_GLOBAL("relax.transform.DeadCodeElimination").set_body_typed(DeadCodeElimination);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
