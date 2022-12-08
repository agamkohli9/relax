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

/*!
  * \brief TODO
  */
class DeadCodeEliminator : public ExprMutator {
 public:
  DeadCodeEliminator() {}

 private:
   
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const VarNode* op) final {
    /** 
     * TODO: Eliminate Dead Vars
    */
    return ExprMutator::VisitExpr_(op);
  }
};

Expr DeadCodeElimination(const Expr& e) { return DeadCodeEliminator().VisitExpr(e); }

namespace transform {

Pass DeadCodeElimination() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(DeadCodeElimination(f));
      };
  return CreateFunctionPass(pass_func, 1, "DeadCodeElimination", {});
}

TVM_REGISTER_GLOBAL("relax.transform.DeadCodeElimination").set_body_typed(DeadCodeElimination);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
