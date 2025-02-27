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
 * \file src/relax/block_builder.cc
 */

#include <tvm/arith/analyzer.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/type.h>
#include <tvm/relax/type_analysis.h>
#include <tvm/relax/utils.h>
#include <tvm/relay/op.h>
#include <tvm/tir/function.h>

namespace tvm {
namespace relax {

// ================================
// BlockBuilderNode::ExprNormalizer
// Invariance:
// After Normalize: an Expr always have checked_type (with the exception of Op).
class BlockBuilderNode::ExprNormalizer : public ExprFunctor<Expr(const Expr&)> {
 public:
  ExprNormalizer(BlockBuilderNode* builder) : builder_(builder) {}

#define RELAX_EXPR_NORMALIZER_LEAF(OP) \
  Expr VisitExpr_(const OP* op) final { return GetRef<Expr>(op); }

  RELAX_EXPR_NORMALIZER_LEAF(RuntimeDepShapeNode);
  RELAX_EXPR_NORMALIZER_LEAF(ExternFuncNode);
  RELAX_EXPR_NORMALIZER_LEAF(GlobalVarNode);
  RELAX_EXPR_NORMALIZER_LEAF(OpNode);
  RELAX_EXPR_NORMALIZER_LEAF(ShapeExprNode);

  // TODO(@altanh): CopyOnWrite

  Expr VisitExpr(const Expr& expr) {
    // TODO(relax-team): generalize prim_func support
    if (expr->IsInstance<tir::PrimFuncNode>()) {
      return expr;
    }
    Optional<Expr> post = expr_memo_.Get(expr);
    if (post) {
      ICHECK(post.as<VarNode>()) << "memoized expressions should map to variables";
      return post.value();
    }
    return ExprFunctor::VisitExpr(expr);
  }

  Expr VisitExpr_(const DataflowVarNode* var) final {
    bool shape_unchanged = true;
    Expr new_shape;
    if (var->shape_) {
      new_shape = this->VisitExpr(Downcast<Expr>(var->shape_.value()));
      shape_unchanged &= new_shape.same_as(var->shape_);
    }

    if (shape_unchanged) {
      return GetRef<Var>(var);
    } else {
      Var new_var = DataflowVar(var->vid, NullOpt, var->checked_type_, var->span);
      UpdateShape(new_var, new_shape);
      return new_var;
    }
  }

  Expr VisitExpr_(const VarNode* var) final {
    bool shape_unchanged = true;
    Expr new_shape;
    if (var->shape_) {
      new_shape = this->VisitExpr(Downcast<Expr>(var->shape_.value()));
      shape_unchanged &= new_shape.same_as(var->shape_);
    }

    if (shape_unchanged) {
      return GetRef<Var>(var);
    } else {
      Var new_var = Var(var->vid, NullOpt, var->checked_type_, var->span);
      UpdateShape(new_var, new_shape);
      return new_var;
    }
  }

  Expr VisitExpr_(const TupleNode* op) final {
    bool unchanged = true;
    Array<Expr> new_fields;
    for (const Expr& field : op->fields) {
      Expr new_field = this->Bind(field);
      new_fields.push_back(new_field);
      unchanged &= new_field.same_as(field);
    }
    Tuple tuple = unchanged ? GetRef<Tuple>(op) : Tuple(new_fields);

    // only do shape/type inference if the Tuple does not have shape/type
    if (tuple->shape_ && tuple->checked_type_.defined()) {
      return tuple;
    }

    // Tuple's checked_type must not be null
    if (!tuple->checked_type_.defined()) {
      if (tuple->fields.size() == 0) {
        UpdateType(tuple, VoidType());
      }
      Array<Type> tuple_type;
      for (Expr field : tuple->fields) {
        ICHECK(field->checked_type_.defined())
            << "The checked_type_ of the field " << field << " of Tuple has not propagated.";
        tuple_type.push_back(field->checked_type_);
      }
      UpdateType(tuple, TupleType(tuple_type));
    }

    // NOTE: Tuple's shape can be null
    // When a tuple consists of all DynTensorType elements or nested tuple of DynTensorTypes,
    // it has a shape.
    if (!tuple->shape_) {
      UpdateShape(tuple, GetTupleShape(tuple));
    }

    // recurse into its shape in case its shape also need to be normalized
    if (tuple->shape_ && tuple->shape_.value()->IsInstance<TupleNode>()) {
      this->VisitExpr(Downcast<Expr>(tuple->shape_.value()));
    }

    return tuple;
  }

  Expr VisitExpr_(const FunctionNode* op) final {
    Expr new_body = this->VisitWithNewScope(op->body);
    Function func;
    if (new_body.same_as(op->body)) {
      func = GetRef<Function>(op);
    } else {
      func = Function(op->params, new_body, op->ret_type, op->ret_shape, op->attrs);
    }

    // NOTE: the shape_ of Function is left as null for now
    return func;
  }

  Expr VisitExpr_(const CallNode* op) final {
    Expr new_op = this->Bind(op->op);
    bool unchanged = new_op.same_as(op->op);

    Array<Expr> new_args;
    if (!op->args.empty()) {
      for (const Expr& arg : op->args) {
        Expr new_arg = this->Bind(arg);
        new_args.push_back(new_arg);
        unchanged &= new_arg.same_as(arg);
      }
    }

    Call call;
    if (unchanged) {
      call = GetRef<Call>(op);
    } else {
      call = Call(new_op, new_args, op->attrs, op->type_args);
    }

    // only do shape/type inference if the Call does not have shape/type
    if (call->shape_.defined() && call->checked_type_.defined()) {
      return call;
    }

    // Update the type prior to updating the shape, since the shape inference may need the updated
    // type in cases of Call for ExternFunc.
    if (!call->checked_type_.defined()) {
      // type inference
      auto inferred_type = InferType(call, this->builder_->diag_ctx_, this->builder_->context_mod_);
      UpdateType(call, inferred_type);
    }

    if (!call->shape_) {
      // shape inference
      auto inferred_shape =
          InferShape(call, this->builder_->diag_ctx_, this->builder_->context_mod_);
      if (inferred_shape) {
        UpdateShape(call, inferred_shape.value());
      }
    }

    CheckShapeTypeConsistency(call->shape_, call->checked_type_);
    return call;
  }

  Expr VisitExpr_(const SeqExprNode* op) final {
    bool unchanged = true;
    Array<BindingBlock> new_blocks;
    for (const BindingBlock& block : op->blocks) {
      BindingBlock new_block = this->VisitBindingBlock(block);
      new_blocks.push_back(new_block);
      unchanged &= new_block.same_as(block);
    }

    builder_->BeginBindingBlock();
    // the body may not be a leaf expression, so check for that
    Expr new_body = this->Bind(op->body);
    unchanged &= new_body.same_as(op->body);
    BindingBlock prologue = builder_->EndBlock();

    if (!prologue->bindings.empty()) {
      new_blocks.push_back(prologue);
      unchanged = false;
    }

    // Combine nearby blocks if possible
    Array<BindingBlock> normalized_blocks = NormalizeBlocks(new_blocks);
    unchanged &= normalized_blocks.same_as(new_blocks);

    SeqExpr seq_expr;
    if (unchanged) {
      seq_expr = GetRef<SeqExpr>(op);
    } else {
      seq_expr = SeqExpr(normalized_blocks, new_body);
    }

    // only do shape/type inference if the SeqExpr does not have shape/type
    if (seq_expr->shape_ && seq_expr->checked_type_.defined()) {
      return seq_expr;
    }

    if (!seq_expr->shape_ && seq_expr->body->shape_) {
      UpdateShape(seq_expr, seq_expr->body->shape_);
    }

    if (!seq_expr->checked_type_.defined() && seq_expr->body->checked_type_.defined()) {
      UpdateType(seq_expr, seq_expr->body->checked_type_);
    }
    return seq_expr;
  }

  Expr VisitExpr_(const ConstantNode* op) final {
    Constant constant = GetRef<Constant>(op);

    // only do shape/type inference if the Constant does not have shape/type
    if (constant->shape_ && constant->checked_type_.defined()) {
      return constant;
    }

    auto shape_tuple = constant->data.Shape();
    if (!constant->shape_) {
      Array<PrimExpr> values;
      for (size_t dim = 0; dim < shape_tuple.size(); dim++) {
        values.push_back(IntImm(DataType::Int(64), shape_tuple[dim]));
      }
      UpdateShape(constant, relax::ShapeExpr(values));
    }

    if (!constant->checked_type_.defined()) {
      DataType dtype = constant->data.DataType();
      Type type = relax::DynTensorType(shape_tuple.size(), dtype);
      UpdateType(constant, type);
    }
    return constant;
  }

  Expr VisitExpr_(const IfNode* op) final {
    Expr new_cond = this->Bind(op->cond);
    Expr new_true = this->VisitWithNewScope(op->true_branch);
    Expr new_false = this->VisitWithNewScope(op->false_branch);

    If if_node;
    if (new_cond.same_as(op->cond) && new_true.same_as(op->true_branch) &&
        new_false.same_as(op->false_branch)) {
      if_node = GetRef<If>(op);
    } else {
      if_node = If(new_cond, new_true, new_false);
    }

    if (!op->checked_type_.defined()) {
      ICHECK(new_true->checked_type_.defined() && new_false->checked_type_.defined())
          << "The checked_type_ of true and false branches must not be nullptr.";
      UpdateType(if_node, FindLCA(new_true->checked_type_, new_false->checked_type_));
    }

    if (!op->shape_.defined()) {
      if (new_true->shape_ && new_false->shape_ &&
          builder_->CanProveShapeEqual(Downcast<Expr>(new_true->shape_.value()),
                                       Downcast<Expr>(new_false->shape_.value()))) {
        UpdateShape(if_node, new_true->shape_);
      } else {
        UpdateShape(if_node, RuntimeDepShape());
      }
    }

    return if_node;
  }

  Expr VisitExpr_(const TupleGetItemNode* op) final {
    Expr new_tuple = this->Bind(op->tuple);
    TupleGetItem node;
    if (new_tuple.same_as(op->tuple)) {
      node = GetRef<TupleGetItem>(op);
    }
    node = TupleGetItem(new_tuple, op->index);

    // only do shape/type inference if the TupleGetItem does not have shape/type
    if (node->shape_ && node->checked_type_.defined()) {
      return node;
    }

    if (!node->checked_type_.defined()) {
      const TupleTypeNode* tuple_type = node->tuple->checked_type_.as<TupleTypeNode>();
      ICHECK(tuple_type) << "The checked_type_ of Tuple must be TupleTypeNode.";
      UpdateType(node, tuple_type->fields[node->index]);
    }

    if (!node->shape_ && node->tuple->shape_) {
      // TODO(@prakalp, @yuchen): assign the shape_ to RuntimeDepShape when we cannot obtain the
      // field
      if (const TupleNode* shape = node->tuple->shape_.as<TupleNode>()) {
        UpdateShape(node, shape->fields[node->index]);
      }
    }

    return node;
  }

  Binding VisitBinding(const Binding& binding) {
    if (binding.as<VarBindingNode>()) {
      return this->VisitVarBinding(Downcast<VarBinding>(binding));
    } else {
      ICHECK(binding.as<MatchShapeNode>()) << "expected VarBinding or MatchShape, got " << binding;
      return this->VisitMatchShape(Downcast<MatchShape>(binding));
    }
  }

  VarBinding VisitVarBinding(const VarBinding& binding) {
    Expr new_value = this->VisitExpr(binding->value);
    if (new_value.same_as(binding->value) || new_value.same_as(binding->var)) {
      // if new_value = binding->var, then we have found an ANF binding site, so just return it
      return binding;
    }
    return VarBinding(binding->var, new_value);
  }

  MatchShape VisitMatchShape(const MatchShape& binding) {
    Expr new_value = this->VisitExpr(binding->value);
    if (new_value.same_as(binding->value)) {
      return binding;
    }
    return MatchShape(new_value, binding->pattern, binding->var);
  }

  BindingBlock VisitBindingBlock(const BindingBlock& block) {
    if (block.as<DataflowBlockNode>()) {
      builder_->BeginDataflowBlock();
    } else {
      builder_->BeginBindingBlock();
    }

    bool unchanged = true;
    for (const Binding& binding : block->bindings) {
      Binding new_binding = this->VisitBinding(binding);
      unchanged &= new_binding.same_as(binding);
      if (new_binding.as<VarBindingNode>()) {
        VarBinding var_binding = Downcast<VarBinding>(new_binding);
        if (builder_->CurrentBlockIsDataFlow() && !var_binding->var.as<DataflowVarNode>()) {
          builder_->EmitOutput(var_binding);
        } else {
          builder_->Emit(var_binding);
        }
      } else {
        ICHECK(new_binding.as<MatchShapeNode>());
        builder_->EmitMatchShape(Downcast<MatchShape>(new_binding));
      }
    }
    BindingBlock new_block = builder_->EndBlock();
    unchanged &= new_block->bindings.size() == block->bindings.size();
    if (unchanged) {
      return block;
    }
    return new_block;
  }

  void ResetMemo() { expr_memo_.Reset(); }

 private:
  /*!
   * \brief Memoization map for expressions using Id for equality of variables.
   */
  class ExprMemo {
   public:
    Optional<Expr> Get(const Expr& expr) {
      if (const VarNode* var = expr.as<VarNode>()) {
        auto it = var_memo_.find(var->vid);
        if (it != var_memo_.end()) {
          return it->second;
        }
      } else {
        auto it = expr_memo_.find(expr);
        if (it != expr_memo_.end()) {
          return it->second;
        }
      }
      return NullOpt;
    }

    void Set(const Expr& pre, const Expr& post) {
      if (const VarNode* var = pre.as<VarNode>()) {
        var_memo_[var->vid] = post;
      } else {
        expr_memo_[pre] = post;
      }
    }

    void Reset() {
      var_memo_ = std::unordered_map<Id, Expr, ObjectPtrHash, ObjectPtrEqual>();
      expr_memo_ = std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual>();
    }

   private:
    std::unordered_map<Id, Expr, ObjectPtrHash, ObjectPtrEqual> var_memo_;
    std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> expr_memo_;
  };

  // Helper function to check if a ShapeExpr is constant shape or tuple of constant shape
  bool IsConstantShapes(const Expr& shape) const {
    if (const auto* shape_expr = shape.as<ShapeExprNode>()) {
      return std::all_of(shape_expr->values.begin(), shape_expr->values.end(),
                         [](const PrimExpr& e) { return e->IsInstance<IntImmNode>(); });
    } else if (const auto* shape_tuple = shape.as<TupleNode>()) {
      return std::all_of(shape_tuple->fields.begin(), shape_tuple->fields.end(),
                         [&](const Expr& e) { return IsConstantShapes(e); });
    } else {
      return false;
    }
  }

  // Helper function to infer the shape of a Call.
  Optional<Expr> InferShape(const Call& call, DiagnosticContext diag_ctx, IRModule ctx_mod) {
    if (call->op.as<ExternFuncNode>()) {
      std::function<Expr(const Type&)> f_create_type = [&f_create_type](const Type& type) -> Expr {
        if (!type.defined() || type->IsInstance<ShapeTypeNode>() ||
            type->IsInstance<FuncTypeNode>() || type->IsInstance<ObjectTypeNode>()) {
          return Expr();
        }
        if (const auto* tuple_type = type.as<TupleTypeNode>()) {
          if (tuple_type->fields.size() == 0) {
            // VoidType (i.e. empty TupleType) does not have shape
            return Expr();
          }
          Array<Expr> fields;
          fields.reserve(tuple_type->fields.size());
          for (const Type& field_type : tuple_type->fields) {
            fields.push_back(f_create_type(field_type));
          }
          return Tuple(fields);
        } else if (type->IsInstance<DynTensorTypeNode>()) {
          return RuntimeDepShape();
        } else {
          LOG(FATAL) << "Unsupported relax type: " << type->GetTypeKey();
          throw;
        }
      };
      return f_create_type(call->checked_type_);
    } else if (call->op.as<OpNode>()) {
      // primitive op: look up FInferShape attribute
      Op op = Downcast<Op>(call->op);
      if (op_map_infer_shape_.count(op)) {
        return op_map_infer_shape_[op](call, diag_ctx);
      }
    } else if (const auto* gv = call->op.as<GlobalVarNode>()) {
      // global function: find the function's shape_
      auto it_func = ctx_mod->functions.find(GetRef<GlobalVar>(gv));

      if (it_func != ctx_mod->functions.end()) {
        if (const auto* func = (*it_func).second.as<FunctionNode>()) {
          if (!func->body.defined()) {
            return func->ret_shape;
          }
          // TODO(relax-team): migrate shape deduction to `ret_shape`
          Expr func_shape = Downcast<Expr>(func->body->shape_);
          if (IsConstantShapes(func_shape)) {
            // Case 1. Nested tuples of constant shapes
            return func_shape;
          } else {
            // TODO(@yuchen): add deducer for other cases
            return RuntimeDepShape();
          }
        }
      }
      // TODO(@yuchen): add this check after normalization in parser
      // else {
      //   LOG(FATAL) << "ValueError: Cannot find function " << gv->name_hint
      //              << " in the context IRModule.";
      // }
    } else if (const auto* var = call->op.as<VarNode>()) {
      if (var->shape_) {
        return Downcast<Expr>(var->shape_.value());
      }
      Optional<Expr> val = builder_->LookupBinding(GetRef<Var>(var));
      if (const auto* func_node = val.value().as<FunctionNode>()) {
        Function func = GetRef<Function>(func_node);
        if (func->ret_type.as<DynTensorTypeNode>()) {
          Expr func_shape = Downcast<Expr>(func_node->body->shape_);
          if (IsConstantShapes(func_shape)) {
            return func_shape;
          } else {
            // TODO(@yuchen, @yongwww): add deducer for other cases
            return RuntimeDepShape();
          }
        }
      }
    } else {
      LOG(FATAL) << "ValueError: Failed to do shape inference for " << call->op->GetTypeKey();
    }

    return NullOpt;
  }

  // Helper function to infer the type of a Call.
  Type InferType(const Call& call, DiagnosticContext diag_ctx, IRModule ctx_mod) {
    if (call->op.as<OpNode>()) {
      // Case 1: the op field is a primitive op, look up FInferType attribute
      Op op = Downcast<Op>(call->op);
      if (op_map_infer_type_.count(op)) {
        return op_map_infer_type_[op](call, diag_ctx);
      } else {
        LOG(FATAL) << "ValueError: Cannot find the FInferType attribute registered to op: "
                   << op->name;
      }
    } else {
      // Case 2: the op field is of callable type
      ICHECK(call->op->checked_type_.defined())
          << "When the op field is not an OpNode, the CallNode's op must have checked_type_.";
      if (call->op->checked_type_.as<PackedFuncTypeNode>()) {
        if (call->type_args.defined()) {
          if (call->type_args.size() == 0) {
            return ObjectType();
          } else if (call->type_args.size() == 1) {
            return call->type_args.front();
          } else {
            return TupleType(call->type_args);
          }
        } else {
          LOG(FATAL) << "ExternFunc call must have type args.";
        }
      } else if (auto* func_node = call->op->checked_type_.as<FuncTypeNode>()) {
        return func_node->ret_type;
      }
    }
    LOG(FATAL) << "ValueError: the CallNode's op has to be either an OpNode, or has "
               << " Callable (i.e., PackedFuncType or FuncType) as its checked_type_";
    throw;
  }

  // Helper function to check if the provided shape and type is consistent.
  // Throw internal exceptions if they are not consistent.
  void CheckShapeTypeConsistency(const Optional<ObjectRef>& opt_shape, const Type& type) {
    if (!type.defined() || type->IsInstance<ShapeTypeNode>() || type->IsInstance<FuncTypeNode>() ||
        type->IsInstance<ObjectTypeNode>()) {
      ICHECK(!opt_shape.defined())
          << "When the type of an Expr is undefined/ShapeType/FuncType/ObjectType, the shape of "
             "this Expr is expected to be undefined. However, the actual shape is defined and is "
          << opt_shape.value();
    } else if (const auto* dyn_tensor_type = type.as<DynTensorTypeNode>()) {
      // `opt_shape` should either be a relax::Expr or undefined.
      if (opt_shape.defined()) {
        const auto* shape = opt_shape.as<ExprNode>();
        ICHECK(shape != nullptr) << "The shape of an Expr, if defined, is expected to be a Relax "
                                    "Expr. However, the actual shape is not a Relax Expr and is "
                                 << opt_shape.value()->GetTypeKey();
        ICHECK(shape->checked_type()->IsInstance<ShapeTypeNode>())
            << "The shape of an Expr, if defined, is expected to be a Relax Expr which has type "
               "ShapeType. However, the actual shape has type "
            << shape->checked_type()->GetTypeKey();
      }

      const auto* shape_expr = opt_shape.as<ShapeExprNode>();
      if (dyn_tensor_type->IsUnknownNdim()) {
        ICHECK(shape_expr == nullptr)
            << "When the type of an Expr is DynTensorType with unknown ndim, the shape of the Expr "
               "is expected not to be a ShapeExpr. However, the actual shape is ShapeExpr "
            << GetRef<ShapeExpr>(shape_expr);
      } else if (shape_expr != nullptr) {
        ICHECK(dyn_tensor_type->ndim == static_cast<int>(shape_expr->values.size()))
            << "When the type of an Expr is DynTensorType with known ndim and the shape of that "
               "Expr is a ShapeExpr, the ShapeExpr should have as many values as the ndim "
               "indicates. However, the actual Expr type has ndim "
            << dyn_tensor_type->ndim << " while the actual Expr shape is "
            << GetRef<ShapeExpr>(shape_expr) << ", which has length " << shape_expr->values.size();
      }
    } else if (const auto* tuple_type = type.as<TupleTypeNode>()) {
      const auto* tuple_shape = opt_shape.as<TupleNode>();
      if (tuple_shape == nullptr) {
        ICHECK(tuple_type->fields.size() == 0)
            << "When the type of an Expr is TupleType and the shape of that Expr is not a Tuple, "
               "it means that the type should be a VoidType, which is represented as an empty "
               "TupleType. However, here the shape is not a tuple while the type has "
            << tuple_type->fields.size() << " field(s).";
      } else {
        ICHECK_EQ(tuple_shape->fields.size(), tuple_type->fields.size())
            << "When the type of an Expr is TupleType and the shape of that Expr is a Tuple, the "
               "two should have the same number of fields. However, the type has "
            << tuple_type->fields.size() << " field(s) while the shape has "
            << tuple_shape->fields.size() << " field(s)";
        int n_field = tuple_shape->fields.size();
        // Recursively check the consistency.
        for (int i = 0; i < n_field; ++i) {
          CheckShapeTypeConsistency(tuple_shape->fields[i], tuple_type->fields[i]);
        }
      }
    } else {
      LOG(FATAL) << "Unsupported relax type: " << type->GetTypeKey();
    }
  }

  // Helper function to get the shape of a Tuple based on its fields
  Optional<Expr> GetTupleShape(const Tuple& tuple) {
    Array<Expr> tuple_shape;
    for (Expr field : tuple->fields) {
      if (field->shape_) {
        tuple_shape.push_back(Downcast<Expr>(field->shape_.value()));
      } else {
        break;
      }
    }
    if (tuple_shape.size() == tuple->fields.size()) {
      return Tuple(tuple_shape);
    }
    return NullOpt;
  }

  Expr VisitWithNewScope(const Expr& expr) {
    builder_->BeginBindingBlock();
    Expr post = this->VisitExpr(expr);
    BindingBlock prologue = builder_->EndBlock();
    // "New scopes" (function bodies, if/else clauses) must be wrapped in seq exprs.
    // Don't wrap if it's already a seq and there are no bindings to add
    if (post.as<SeqExprNode>() && prologue->bindings.empty()) {
      return post;
    }
    Array<BindingBlock> bindings;
    if (!prologue->bindings.empty()) {
      bindings.push_back(prologue);
    }
    auto seq = SeqExpr(bindings, post);
    // visit in case post is not a leaf and we need to bind it too
    return this->VisitExpr(seq);
  }

  Array<BindingBlock> NormalizeBlocks(const Array<BindingBlock>& blocks) {
    bool changed = false;
    Array<BindingBlock> ret;
    for (const BindingBlock& block : blocks) {
      if (block->bindings.empty()) {
        // Case 1. Skip empty blocks
        changed = true;
      } else if (!ret.empty() && ret.back()->type_index() == block->type_index()) {
        // Case 2. Merge with previous block if possible
        BindingBlock merged;
        // NOTE: should check DataflowBlockNode first.
        if (const auto* dataflow_block = ret.back().as<DataflowBlockNode>()) {
          auto n = make_object<DataflowBlockNode>(*dataflow_block);
          n->bindings.insert(n->bindings.end(), block->bindings.begin(), block->bindings.end());
          merged = DataflowBlock(n);
        } else if (const auto* binding_block = ret.back().as<BindingBlockNode>()) {
          auto n = make_object<BindingBlockNode>(*binding_block);
          n->bindings.insert(n->bindings.end(), block->bindings.begin(), block->bindings.end());
          merged = BindingBlock(n);
        } else {
          LOG(FATAL) << "Unknown block type: " << ret.back()->GetTypeKey();
        }
        ret.pop_back();
        ret.push_back(merged);
        changed = true;
      } else if (false) {
        // Case 3. TODO(@Hzfengsy): normalize nested SeqExprs and BindingBlocks
      } else {
        // Case 4. Add to the result
        ret.push_back(block);
      }
    }
    return changed ? ret : blocks;
  }

  Expr Bind(const Expr& expr) {
    Expr post = this->VisitExpr(expr);
    if (!IsLeafExpr(post)) {
      post = builder_->Emit(post);
      expr_memo_.Set(expr, post);
    }
    return post;
  }

  /*! \brief BlockBuilder used for emitting intermediate variables. */
  BlockBuilderNode* builder_;

  /*! \brief Memoization table for mapping expressions to their ANF variables. */
  ExprMemo expr_memo_;

  /*! \brief Operator to shape inference map. */
  tvm::OpAttrMap<FInferShape> op_map_infer_shape_ = Op::GetAttrMap<FInferShape>("FInferShape");

  /*! \brief Operator to type inference map. */
  tvm::OpAttrMap<FInferType> op_map_infer_type_ = Op::GetAttrMap<FInferType>("FInferType");
};

// ================
// BlockBuilderNode

TVM_REGISTER_NODE_TYPE(BlockBuilderNode);

BlockBuilderNode::BlockBuilderNode() {
  name_table_ = std::make_unique<NameTable>();
  normalizer_ = std::make_unique<ExprNormalizer>(this);
}

BlockBuilderNode::~BlockBuilderNode() {
  if (!block_stack_.empty()) {
    LOG(WARNING) << "BlockBuilder destroyed with remaining blocks!";
  }
}

void BlockBuilderNode::BeginDataflowBlock() { this->block_stack_.push({{}, true}); }

void BlockBuilderNode::BeginBindingBlock() { this->block_stack_.push({{}, false}); }

BindingBlock BlockBuilderNode::EndBlock() {
  BlockFrame* cur_frame = CurrentFrame();
  BindingBlock ret = cur_frame->is_dataflow ? DataflowBlock(cur_frame->bindings)
                                            : BindingBlock(cur_frame->bindings);
  block_stack_.pop();
  return ret;
}

Var BlockBuilderNode::Emit(const Expr& expr, std::string name_hint) {
  return Emit(expr, CurrentFrame()->is_dataflow, name_hint);
}

Var BlockBuilderNode::Emit(const Expr& expr, bool is_dataflow, std::string name_hint) {
  BlockFrame* cur_frame = CurrentFrame();
  Expr normalized = Normalize(expr);

  if (name_hint.empty()) {
    name_hint = is_dataflow ? "lv" : "gv";
  }
  Id vid = Id(name_table_->GetUniqueName(name_hint));
  Var var = is_dataflow ? DataflowVar(vid, NullOpt, NullOpt) : Var(vid, NullOpt, NullOpt);
  UpdateType(var, normalized->checked_type_);
  UpdateShape(var, normalized->shape_);

  cur_frame->bindings.push_back(VarBinding(var, expr));
  binding_table_[var->vid] = expr;

  return var;
}

Var BlockBuilderNode::Emit(const VarBinding& binding) {
  BlockFrame* cur_frame = CurrentFrame();
  if (cur_frame->is_dataflow) {
    ICHECK(binding->var.as<DataflowVarNode>())
        << "Emit can only be used for local bindings in a dataflow block, use EmitOutput for "
           "output bindings instead";
  }
  cur_frame->bindings.push_back(binding);
  binding_table_[binding->var->vid] = binding->value;
  return binding->var;
}

Var BlockBuilderNode::EmitMatchShape(const Expr& value, const Array<PrimExpr>& pattern,
                                     std::string name_hint) {
  BlockFrame* cur_frame = CurrentFrame();

  if (name_hint.empty()) {
    name_hint = cur_frame->is_dataflow ? "lv" : "gv";
  }
  Id vid = Id(name_table_->GetUniqueName(name_hint));
  Var var =
      cur_frame->is_dataflow ? DataflowVar(vid, NullOpt, NullOpt) : Var(vid, NullOpt, NullOpt);

  if (value->checked_type().as<ShapeTypeNode>()) {
    UpdateType(var, ShapeType());
  } else if (const DynTensorTypeNode* tty = value->checked_type().as<DynTensorTypeNode>()) {
    ShapeExpr shape = ShapeExpr(pattern);
    UpdateShape(var, shape);
    DataType dtype = tty->dtype;
    UpdateType(var, DynTensorType(pattern.size(), dtype));
  } else {
    this->diag_ctx_.EmitFatal(
        Diagnostic::Error(value->span)
        << "The value passed to EmitMatchShape must be of DynTensorType or ShapeType.");
  }

  MatchShape match_shape = MatchShape(value, pattern, var);
  cur_frame->bindings.push_back(match_shape);
  binding_table_[var->vid] = value;
  return var;
}

Var BlockBuilderNode::EmitMatchShape(const MatchShape& binding) {
  BlockFrame* cur_frame = CurrentFrame();
  if (binding->var.defined()) {
    ICHECK(cur_frame->is_dataflow || !binding->var.as<DataflowVarNode>())
        << "cannot emit dataflow vars outside a dataflow block: " << binding->var->name_hint();
    binding_table_[binding->var->vid] = binding->value;
  }
  cur_frame->bindings.push_back(binding);
  return binding->var;
}

Var BlockBuilderNode::EmitOutput(const Expr& output, std::string name_hint) {
  BlockFrame* cur_frame = CurrentFrame();

  ICHECK(cur_frame->is_dataflow) << "EmitOutput has to be called inside dataflow block.";

  return Emit(output, false, name_hint);
}

Var BlockBuilderNode::EmitOutput(const VarBinding& binding) {
  BlockFrame* cur_frame = CurrentFrame();

  ICHECK(cur_frame->is_dataflow) << "EmitOutput has to be called inside dataflow block.";
  ICHECK(!binding->var.as<DataflowVarNode>()) << "EmitOutput can only emit Var bindings.";

  cur_frame->bindings.push_back(binding);
  binding_table_[binding->var->vid] = binding->value;
  return binding->var;
}

Optional<Expr> BlockBuilderNode::LookupBinding(const Var& var) {
  auto it = binding_table_.find(var->vid);
  if (it == binding_table_.end()) return NullOpt;
  return it->second;
}

bool BlockBuilderNode::CanProveShapeEqual(const Expr& lhs, const Expr& rhs) {
  if (lhs == rhs) {
    return true;
  }
  if (lhs->IsInstance<RuntimeDepShapeNode>() && rhs->IsInstance<RuntimeDepShapeNode>()) {
    return true;
  } else if (lhs->IsInstance<ShapeExprNode>() && rhs->IsInstance<ShapeExprNode>()) {
    const auto* lhs_shape = lhs.as<ShapeExprNode>();
    const auto* rhs_shape = rhs.as<ShapeExprNode>();
    size_t lhs_ndim = lhs_shape->values.size();
    size_t rhs_ndim = rhs_shape->values.size();
    if (lhs_ndim != rhs_ndim) {
      return false;
    }
    arith::Analyzer analyzer;
    for (size_t i = 0; i < lhs_ndim; ++i) {
      PrimExpr lhs_dim = lhs_shape->values[i];
      PrimExpr rhs_dim = rhs_shape->values[i];
      if (lhs_dim.dtype() != rhs_dim.dtype() || !analyzer.CanProveEqual(lhs_dim, rhs_dim)) {
        return false;
      }
    }
    return true;
  } else if (lhs->IsInstance<TupleNode>() && rhs->IsInstance<TupleNode>()) {
    const auto* lhs_tuple = lhs.as<TupleNode>();
    const auto* rhs_tuple = rhs.as<TupleNode>();
    if (lhs_tuple->fields.size() != rhs_tuple->fields.size()) {
      return false;
    }
    for (size_t i = 0; i < lhs_tuple->fields.size(); ++i) {
      if (!CanProveShapeEqual(lhs_tuple->fields[i], rhs_tuple->fields[i])) {
        return false;
      }
    }
    return true;
  }
  return false;
}

void BlockBuilderNode::ResetMemo() { normalizer_->ResetMemo(); }

// TODO(@altanh, @yuchen): need an internal Emit_ that doesn't call normalize
Expr BlockBuilderNode::Normalize(const Expr& expr) {
  Expr normalized = normalizer_->VisitExpr(expr);

  if (!normalized->IsInstance<OpNode>()) {
    ICHECK(normalized->checked_type_.defined())
        << "The checked_type_ of an Expr except OpNode after "
           "normalization must not be nullptr. However, this Expr does not have checked_type_: "
        << normalized;
  }

  return normalized;
}

BlockBuilderNode::BlockFrame* BlockBuilderNode::CurrentFrame() {
  ICHECK(!block_stack_.empty()) << "no block is being built";
  return &block_stack_.top();
}

NameTable* BlockBuilderNode::name_table() { return name_table_.get(); }

GlobalVar BlockBuilderNode::AddFunction(const BaseFunc& func, const String& func_name_hint) {
  auto it = func_map_.find(func);
  if (it == func_map_.end()) {
    context_mod_.CopyOnWrite();
    String func_name = name_table_->GetUniqueName(func_name_hint);
    while (context_mod_->ContainGlobalVar(func_name)) {
      func_name = name_table_->GetUniqueName(func_name_hint);
    }
    GlobalVar gvar = GlobalVar(func_name);
    if (const tir::PrimFuncNode* prim_func = func.as<tir::PrimFuncNode>()) {
      tir::PrimFunc fn = GetRef<tir::PrimFunc>(prim_func);
      ICHECK(fn->checked_type_.defined())
          << "The function to be added does not have checked_type_.";
      gvar->checked_type_ = fn->checked_type_;
      context_mod_->Add(gvar, fn);
    } else {
      ICHECK(func->checked_type_.defined())
          << "The function to be added does not have checked_type_.";
      gvar->checked_type_ = func->checked_type_;
      context_mod_->Add(gvar, func);
    }
    func_map_.emplace(func, gvar);
    return gvar;
  } else {
    return it->second;
  }
}

void BlockBuilderNode::UpdateFunction(const GlobalVar& gv, BaseFunc updated_function) {
  context_mod_.CopyOnWrite();
  context_mod_->Update(gv, updated_function);
  func_map_[updated_function] = gv;
}

IRModule BlockBuilderNode::GetContextIRModule() const { return context_mod_; }

BlockBuilder BlockBuilder::Create(Optional<IRModule> mod) {
  ObjectPtr<BlockBuilderNode> n = make_object<BlockBuilderNode>();
  if (mod) {
    n->context_mod_ = mod.value();
  }
  BlockBuilder block_builder(n);
  for (const auto& kv : n->context_mod_->functions) {
    const GlobalVar& gv = kv.first;
    const BaseFunc& func = kv.second;
    block_builder->func_map_.emplace(func, gv);
  }
  return block_builder;
}

TVM_REGISTER_GLOBAL("relax.BlockBuilderCreate").set_body_typed([](Optional<IRModule> mod) {
  return BlockBuilder::Create(mod);
});

TVM_REGISTER_GLOBAL("relax.BlockBuilderBeginDataflowBlock")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::BeginDataflowBlock);

TVM_REGISTER_GLOBAL("relax.BlockBuilderBeginBindingBlock")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::BeginBindingBlock);

TVM_REGISTER_GLOBAL("relax.BlockBuilderEndBlock")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::EndBlock);

TVM_REGISTER_GLOBAL("relax.BlockBuilderNormalize")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::Normalize);

TVM_REGISTER_GLOBAL("relax.BlockBuilderEmit").set_body_typed([](BlockBuilder builder, Expr expr) {
  return builder->Emit(expr);
});

TVM_REGISTER_GLOBAL("relax.BlockBuilderEmitVarBinding")
    .set_body_typed([](BlockBuilder builder, VarBinding binding) {
      return builder->Emit(binding);
    });

TVM_REGISTER_GLOBAL("relax.BlockBuilderEmitMatchShape")
    .set_body_typed([](BlockBuilder builder, Expr value, Array<PrimExpr> pattern) {
      return builder->EmitMatchShape(value, pattern);
    });

TVM_REGISTER_GLOBAL("relax.BlockBuilderEmitMatchShapeBinding")
    .set_body_typed([](BlockBuilder builder, MatchShape binding) {
      return builder->EmitMatchShape(binding);
    });

TVM_REGISTER_GLOBAL("relax.BlockBuilderEmitOutput")
    .set_body_typed([](BlockBuilder builder, const Expr& output) {
      return builder->EmitOutput(output);
    });

TVM_REGISTER_GLOBAL("relax.BlockBuilderEmitOutputVarBinding")
    .set_body_typed([](BlockBuilder builder, VarBinding binding) {
      return builder->EmitOutput(binding);
    });

TVM_REGISTER_GLOBAL("relax.BlockBuilderGetUniqueName")
    .set_body_typed([](BlockBuilder builder, String name_hint) {
      return builder->name_table()->GetUniqueName(name_hint);
    });

TVM_REGISTER_GLOBAL("relax.BlockBuilderAddFunction")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::AddFunction);

TVM_REGISTER_GLOBAL("relax.BlockBuilderUpdateFunction")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::UpdateFunction);

TVM_REGISTER_GLOBAL("relax.BlockBuilderGetContextIRModule")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::GetContextIRModule);

TVM_REGISTER_GLOBAL("relax.BlockBuilderCanProveShapeEqual")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::CanProveShapeEqual);

TVM_REGISTER_GLOBAL("relax.BlockBuilderCurrentBlockIsDataFlow")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::CurrentBlockIsDataFlow);

TVM_REGISTER_GLOBAL("relax.BlockBuilderLookupBinding")
    .set_body_method<BlockBuilder>(&BlockBuilderNode::LookupBinding);
}  // namespace relax
}  // namespace tvm
