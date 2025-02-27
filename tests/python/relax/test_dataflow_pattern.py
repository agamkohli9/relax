# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import pytest
import tvm.testing

from tvm import relay
from tvm.relax.dpl import *
from tvm.relax.analysis import get_var2val
from tvm import relax as rx, tir
from tvm.script import relax as R, tir as T


@tvm.script.ir_module
class Module:
    @T.prim_func
    def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
        T.func_attr({"global_symbol": "tir_matmul"})
        k = T.var("int32")
        A = T.match_buffer(x, (32, 32))
        B = T.match_buffer(y, (32, 32))
        C = T.match_buffer(z, (32, 32))

        for (i0, j0, k0) in T.grid(32, 32, 32):
            with T.block():
                i, j, k = T.axis.remap("SSR", [i0, j0, k0])
                with T.init():
                    C[i, j] = 0.0
                C[i, j] += A[i, k] * B[j, k]

    @T.prim_func
    def tir_relu(x: T.handle, y: T.handle):
        T.func_attr({"global_symbol": "tir_relu"})
        A = T.match_buffer(x, (32, 32))
        B = T.match_buffer(y, (32, 32))
        for (i, j) in T.grid(32, 32):
            with T.block():
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = T.max(A[vi, vj], 0.0)

    @R.function
    def main(x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")) -> R.Tensor:
        with R.dataflow():
            lv0 = R.call_tir(tir_matmul, (x, w), (32, 32), dtype="float32")
            lv1 = R.call_tir(tir_relu, (lv0), (32, 32), dtype="float32")
            R.output(lv1)
        return lv1


main_fn = Module["main"]
bindings = main_fn.body.blocks[0].bindings

## Node-wise Matching
def test_expr_pattern():
    ep = is_expr(rx.Var("x"))
    assert isinstance(ep, ExprPattern)
    assert isinstance(ep.expr, rx.Var)


def test_var_pattern():
    v = is_var("x")
    assert isinstance(v, VarPattern)
    assert v.name == "x"
    assert v.match(rx.Var("x"))
    assert is_var().match(rx.Var("x"))
    assert is_var().match(rx.DataflowVar("x"))  # DataflowVar is also a Var
    assert not v.match(rx.GlobalVar("x"))


def test_dataflow_var_pattern():
    v = is_dfv("x")
    assert isinstance(v, DataflowVarPattern)
    assert v.name == "x"
    assert v.match(rx.DataflowVar("x"))
    assert not v.match(rx.GlobalVar("x"))
    assert is_dfv().match(bindings[0].var)


def test_global_var_pattern():
    assert is_gv("x").match(rx.GlobalVar("x"))
    assert is_gv().match(rx.GlobalVar("x"))
    assert not is_gv("x").match(rx.GlobalVar("y"))
    assert not is_gv("x").match(rx.Var("x"))


def test_constant_pattern():
    c = is_const()
    assert isinstance(c, ConstantPattern)
    assert c.match(rx.const([[0.1, 1.1, 2.1], [3.1, 4.1, 5.1]]))


def test_wildcard_pattern():
    wc = wildcard()
    assert isinstance(wc, WildcardPattern)
    assert wc.match(rx.Var("x"))


def test_call_pattern():
    wc1 = wildcard()
    wc2 = wildcard()
    c = is_op("relax.add")(wc1, wc2)
    assert isinstance(c, CallPattern)
    assert isinstance(c.args[0], WildcardPattern)
    assert isinstance(c.args[1], WildcardPattern)
    assert c.match(rx.op.add(rx.Var("x"), rx.Var("y")))


def test_function_pattern():
    wc1 = wildcard()
    wc2 = wildcard()
    f = FunctionPattern([wc1, wc2], is_op("relax.add")(wc1, wc2))
    assert isinstance(f, FunctionPattern)
    assert isinstance(f.params[0], WildcardPattern)
    assert isinstance(f.params[1], WildcardPattern)
    assert isinstance(f.body, CallPattern)
    assert isinstance(f.body.args[0], WildcardPattern)
    assert isinstance(f.body.args[1], WildcardPattern)
    ttype = rx.DynTensorType(-1, "float32")
    x = rx.Var("x", type_annotation=ttype)
    y = rx.Var("y", type_annotation=ttype)
    assert f.match(
        rx.Function([x, y], rx.op.add(x, y), ret_type=ttype, ret_shape=rx.RuntimeDepShape())
    )
    assert not f.match(
        rx.Function([x, y], rx.op.multiply(x, y), ret_type=ttype, ret_shape=rx.RuntimeDepShape())
    )


def test_tuple_pattern():
    wc1 = wildcard()
    wc2 = is_dfv()
    t = is_tuple([wc1, wc2])
    assert isinstance(t, TuplePattern)
    assert isinstance(t.fields[0], WildcardPattern)
    assert isinstance(t.fields[1], DataflowVarPattern)
    assert t.match(rx.Tuple([rx.GlobalVar("x"), rx.DataflowVar("y")]))
    assert not t.match(rx.Tuple([rx.DataflowVar("x"), rx.GlobalVar("y")]))
    assert not t.match(rx.Tuple([]))
    assert t[0].match(rx.TupleGetItem(rx.Tuple([rx.GlobalVar("x"), rx.DataflowVar("y")]), 0))
    assert t[1].match(rx.TupleGetItem(rx.Tuple([rx.GlobalVar("x"), rx.DataflowVar("y")]), 1))
    # Negative index is also allowed
    assert t[-1].match(rx.TupleGetItem(rx.Tuple([rx.GlobalVar("x"), rx.DataflowVar("y")]), 1))
    # None means any index.
    assert t[None].match(rx.TupleGetItem(rx.Tuple([rx.GlobalVar("x"), rx.DataflowVar("y")]), 0))
    assert t[None].match(rx.TupleGetItem(rx.Tuple([rx.GlobalVar("x"), rx.DataflowVar("y")]), 1))
    with pytest.raises(IndexError):
        t[2]  # index cannot be greater than or equal to the tuple size.


def test_unordered_tuple_pattern():
    t = is_tuple([is_const(), is_dfv()], unordered=True)
    assert isinstance(t, UnorderedTuplePattern)
    assert isinstance(t.fields[0], ConstantPattern)
    assert isinstance(t.fields[1], DataflowVarPattern)
    assert t.match(rx.Tuple([rx.const([]), rx.DataflowVar("x")]))
    assert t.match(rx.Tuple([rx.DataflowVar("x"), rx.const([])]))
    assert not t.match(rx.Tuple([rx.DataflowVar("x"), rx.DataflowVar("y")]))
    assert not t.match(rx.Tuple([]))


def test_tuple_get_item_pattern():
    assert is_tuple_get_item(is_tuple([is_gv("x"), is_dfv("y")]), 0).match(
        rx.TupleGetItem(rx.Tuple([rx.GlobalVar("x"), rx.DataflowVar("y")]), 0)
    )
    assert is_tuple_get_item(is_tuple([is_gv("x"), is_dfv("y")]), 0).match(
        rx.TupleGetItem(rx.Tuple([rx.GlobalVar("x"), rx.DataflowVar("y")]), 0)
    )


def test_or_pattern():
    dfv_or_gv = is_dfv("x") | is_gv("x")
    assert isinstance(dfv_or_gv, OrPattern)
    assert dfv_or_gv.match(rx.DataflowVar("x"))
    assert dfv_or_gv.match(rx.GlobalVar("x"))
    assert not dfv_or_gv.match(rx.Var("x"))
    assert not dfv_or_gv.match(rx.DataflowVar("y"))
    assert not dfv_or_gv.match(rx.GlobalVar("y"))


def test_and_pattern():
    # float[2, 3, 3]
    f32_233 = wildcard().has_shape((2, 3, 3)) & has_dtype("float32")
    assert isinstance(f32_233, AndPattern)
    assert f32_233.match(rx.Var("x", (2, 3, 3), rx.DynTensorType(3, "float32")))
    assert not f32_233.match(rx.Var("x", (3, 3, 3), rx.DynTensorType(3, "float32")))
    assert not f32_233.match(rx.Var("x", rx.RuntimeDepShape(), rx.DynTensorType(3, "float32")))


def test_not_pattern():
    no_shape233 = ~wildcard().has_shape((2, 3, 3))
    assert isinstance(no_shape233, NotPattern)
    assert no_shape233.match(rx.Var("x", (3, 3, 3), rx.DynTensorType(3, "float32")))
    assert not no_shape233.match(rx.Var("x", (2, 3, 3), rx.DynTensorType(3, "float32")))


def test_type_pattern():
    assert wildcard().has_type(rx.DynTensorType(2, "float32")).match(bindings[0].var)


def test_dtype_pattern():
    dtype = "float16"
    pattern = has_dtype(dtype)
    assert isinstance(pattern, DataTypePattern)
    assert pattern.dtype == dtype
    assert has_dtype("float32").match(bindings[0].var)


def test_shape_pattern():
    shape = [32, 32]
    pattern = wildcard().has_shape(shape)
    assert isinstance(pattern, ShapePattern)
    tvm.ir.structural_equal(pattern.shape, shape)
    assert pattern.match(bindings[0].var)
    assert wildcard().has_shape([32, 32]).match(bindings[0].var)
    n, m = tir.Var("n", dtype="int64"), tir.Var("m", dtype="int64")
    symbolic_shape = rx.ShapeExpr([n, m, n + m])
    symsh_var = rx.Var("x", symbolic_shape, rx.DynTensorType(3, "float32"))
    assert wildcard().has_shape([n, m, n + m]).match(symsh_var)
    assert wildcard().has_shape([n, m, m + n]).match(symsh_var)  # + is commutative.
    assert not wildcard().has_shape([1, 2, 3]).match(symsh_var)
    assert not wildcard().has_shape([m, n, n + m]).match(symsh_var)


def test_prim_arr_pattern():
    """
    The difference between is_shape and has_shape is that:
    1) is_shape directly matches a shape (e.g., as an argument);
    2) has_shape matches a tensor and puts assumptions on the tensor's shape.
    """
    pattern = is_shape([32, 32])
    assert pattern[0] == 32
    assert pattern[1] == 32
    assert isinstance(pattern, PrimArrPattern)
    assert pattern.match(bindings[0].var.shape)
    n, m = tir.Var("n", dtype="int64"), tir.Var("m", dtype="int64")
    symbolic_shape = rx.ShapeExpr([n, m, n + m])
    assert is_shape([n, m, n + m]).match(symbolic_shape)
    assert not is_shape([n, m, n * m]).match(symbolic_shape)


def test_rt_dep_shape_pattern():
    # runtime-dep-shape var
    rts_var = rx.Var("rts_var", rx.RuntimeDepShape(), rx.DynTensorType(4, "float32"))
    # static-shape var
    ss_var = rx.Var("ss_var", rx.ShapeExpr([32, 32]), rx.DynTensorType(4, "float32"))
    assert wildcard().has_rt_dep_shape().match(rts_var)
    assert not wildcard().has_rt_dep_shape().match(ss_var)


def test_extern_fn_pattern():
    pattern = ExternFuncPattern("test.blockbuilder.nop")
    assert pattern.match(rx.ExternFunc("test.blockbuilder.nop"))


def test_op_attr():
    ttype = rx.DynTensorType(-1, "float32")
    x = rx.Var("x", type_annotation=ttype)
    y = rx.Var("y", type_annotation=ttype)
    conv2d = relay.nn.conv2d(x, y, kernel_size=(3, 3))
    xp = is_var("x")
    yp = is_var("y")
    assert is_op("nn.conv2d")(xp, yp).has_attr({"kernel_size": [3, 3]}).match(conv2d)
    assert not is_op("nn.conv2d")(xp, yp).has_attr({"kernel_size": [4, 3]}).match(conv2d)
    assert not is_op("nn.conv2d")(xp, yp).has_attr({"kernel_size_": [3, 3]}).match(conv2d)


def test_match_call_attr():
    ttype = rx.DynTensorType(-1, "float32")
    x = rx.Var("x", type_annotation=ttype)
    y = rx.Var("y", type_annotation=ttype)
    fn = rx.Function([x, y], rx.op.add(x, y), ret_type=ttype, ret_shape=rx.RuntimeDepShape())
    annotated_fn = fn.with_attr({"Codegen": "test-codegen", "global_symbol": "test-symbol"})
    xp = is_var("x")
    yp = is_var("y")
    root_pattern = FunctionPattern([xp, yp], is_op("relax.add")(xp, yp))
    assert root_pattern.has_attr({"Codegen": "test-codegen", "global_symbol": "test-symbol"}).match(
        annotated_fn
    )

    assert root_pattern.has_attr({"Codegen": "test-codegen"}).match(annotated_fn)
    assert not root_pattern.has_attr({"ping": "pong"}).match(annotated_fn)
    assert root_pattern.has_attr({}).match(annotated_fn)


def test_is_call_tir():
    lv1_val = bindings[1].value
    var2val = get_var2val(Module["main"])
    assert is_call_tir("tir_relu").match(lv1_val)
    assert is_call_tir("tir_relu", [is_call_tir("tir_matmul")]).match(lv1_val, var2val=var2val)
    assert not is_call_tir("tir_relu", [is_call_tir("tir_relu")]).match(lv1_val, var2val=var2val)


@R.function
def simple_call_packed(
    x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")
) -> R.Tensor:
    gv0 = R.call_packed("test.vm.mul", x, w, type_args=(R.Tensor(ndim=2, dtype="float32")))
    return gv0


def test_varg_default_wildcard():
    expr = simple_call_packed.body.blocks[0].bindings[0].value
    yes_pattern_explicit = ExternFuncPattern("test.vm.mul")(wildcard(), wildcard())
    yes_pattern_implicit = ExternFuncPattern("test.vm.mul")(varg_default_wildcard=True)
    no_pattern = ExternFuncPattern("test.vm.mul")(wildcard())

    assert yes_pattern_explicit.match(expr)
    assert yes_pattern_implicit.match(expr)
    assert not no_pattern.match(expr)


def test_simple_call_packed():
    expr = simple_call_packed.body.blocks[0].bindings[0].value
    assert is_call_packed("test.vm.mul").match(expr)
    assert is_call_packed("test.vm.mul", [is_var("x"), is_var("w")]).match(expr)


## Graph-wise Matching
def test_simple_used_by():
    with PatternContext() as ctx:
        n0 = is_var("x")  # x is a free var (fn arg)
        n1 = wildcard()
        n0 ^ n1
        dfb = main_fn.body.blocks[0]
        matched = ctx.match_dfb(dfb)
        assert matched
        assert matched[n0] == main_fn.params[0]
        assert matched[n1] == dfb.bindings[0].var


def test_simple_call_tir_edge():
    with PatternContext() as ctx:
        n0 = is_call_tir("tir_matmul")
        n1 = is_call_tir("tir_relu")
        n0.used_by(n1)
        dfb = main_fn.body.blocks[0]
        matched = ctx.match_dfb(dfb)
        assert matched
        assert matched[n0] == dfb.bindings[0].var
        assert matched[n1] == dfb.bindings[1].var


def test_simple_oub():
    with PatternContext() as ctx:
        n0 = is_call_tir("tir_matmul")
        n1 = is_call_tir("tir_relu")
        n0 >> n1
        dfb = main_fn.body.blocks[0]
        matched = ctx.match_dfb(dfb)
        assert matched
        assert matched[n0] == dfb.bindings[0].var
        assert matched[n1] == dfb.bindings[1].var


def test_counter_syntax_match():
    with PatternContext() as ctx:
        n0 = is_call_tir_extern("tir_matmul")
        n1 = is_call_tir_extern("tir_impossible")
        n0 >> n1
        dfb = main_fn.body.blocks[0]
        assert not ctx.match_dfb(dfb)

    with PatternContext() as ctx:
        n0 = is_call_tir_extern("tir_matmul")
        n1 = is_call_tir_extern("tir_impossible")
        n0 ^ n1
        dfb = main_fn.body.blocks[0]
        assert not ctx.match_dfb(dfb)


@tvm.script.ir_module
class Diamond:
    @R.function
    def main(x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")) -> R.Tensor:
        with R.dataflow():
            #   matmul
            #  /      \
            # relu  sigmoid
            #  \      /
            #    add
            lv0 = R.call_tir("tir_matmul", (x, w), (32, 32), dtype="float32")
            lv1 = R.call_tir("tir_relu", (lv0,), (32, 32), dtype="float32")
            lv2 = R.call_tir("tir_sigmoid", (lv0), (32, 32), dtype="float32")
            lv3 = R.call_tir("tir_add", (lv1, lv2), (32, 32), dtype="float32")
            R.output(lv3)
        return lv3


def test_diamond():
    with PatternContext() as ctx:
        n0 = is_call_tir_extern("tir_matmul")
        n1 = is_call_tir_extern("tir_relu")
        n2 = is_call_tir_extern("tir_sigmoid")
        n3 = is_call_tir_extern("tir_add")

        n0 ^ n1
        n0 ^ n2
        n1 >> n3
        n2 >> n3

        dfb = Diamond["main"].body.blocks[0]
        assert ctx.match_dfb(dfb)

    # simplify it with fork_to
    with PatternContext() as ctx:
        n1 = is_call_tir_extern("tir_relu")
        n2 = is_call_tir_extern("tir_sigmoid")
        n3 = is_call_tir_extern("tir_add")

        is_call_tir_extern("tir_matmul").fork_to(n1, n2)
        n1 >> n3
        n2 >> n3

        dfb = Diamond["main"].body.blocks[0]
        assert ctx.match_dfb(dfb)


def test_diamond_counter_oub():
    with PatternContext() as ctx:
        n0 = is_call_tir_extern("tir_matmul")
        n1 = is_call_tir_extern("tir_relu")
        n2 = is_call_tir_extern("tir_sigmoid")
        n3 = is_call_tir_extern("tir_add")

        n0 >> n1
        n0 >> n2
        n1 >> n3
        n2 >> n3

        dfb = Diamond["main"].body.blocks[0]
        assert not ctx.match_dfb(dfb)


@tvm.script.ir_module
class SmallDiamond:
    @R.function
    def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        with R.dataflow():
            #    relu
            #  /      \
            #  \      /
            #    add
            lv0 = R.call_tir("my_relu", (x,), (32, 32), dtype="float32")
            lv1 = R.call_tir("my_add", (lv0, lv0), (32, 32), dtype="float32")
            R.output(lv1)
        return lv1


@tvm.script.ir_module
class SmallParallel:
    @R.function
    def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        with R.dataflow():
            # relu   relu
            #   \    /
            #    add
            lv0 = R.call_tir("my_relu", (x,), (32, 32), dtype="float32")
            lv1 = R.call_tir("my_relu", (x,), (32, 32), dtype="float32")
            lv2 = R.call_tir("my_add", (lv0, lv1), (32, 32), dtype="float32")
            R.output(lv2)
        return lv2


def test_distiguish_diamond_and_parallel():
    # relay pattern lang cannot distinguish the two cases above.
    diamond = SmallDiamond["main"].body.blocks[0]
    parallel = SmallParallel["main"].body.blocks[0]

    with PatternContext() as ctx:
        # describe a diamond pattern
        fork = is_call_tir_extern("my_relu")
        join = is_call_tir_extern("my_add")
        fork.only_used_by(join, index=0)
        fork.only_used_by(join, index=1)

        assert ctx.match_dfb(diamond)
        assert not ctx.match_dfb(parallel)

    with PatternContext() as ctx:
        # describe a parallel pattern
        join = is_call_tir_extern("my_add")
        # Due to one-one mathcing:
        # is_call_tir_extern("my_relu") creates the 1st relu
        is_call_tir_extern("my_relu") >> join
        # is_call_tir_extern("my_relu")
        # creates the another different relu (obj address is different)
        is_call_tir_extern("my_relu") >> join

        assert ctx.match_dfb(parallel)
        assert not ctx.match_dfb(diamond)


@tvm.script.ir_module
class CBRx2:
    @R.function
    def main(
        x: R.Tensor((32, 32), "float32"),
        w0: R.Tensor((1, 1), "float32"),
        bias0: R.Tensor((32, 32), "float32"),
        w1: R.Tensor((1, 1), "float32"),
        bias1: R.Tensor((32, 32), "float32"),
    ) -> R.Tensor:
        # R.TensorRT's CBR Optimization Pattern
        #     input
        #     /   \
        #  cbr0   cbr1
        #     \   /
        #     concat
        with R.dataflow():
            lv0 = R.call_tir("conv1x1", (x, w0), (32, 32), dtype="float32")
            lv1 = R.call_tir("bias_add", (lv0, bias0), (32, 32), dtype="float32")
            lv2 = R.call_tir("my_relu", (lv1), (32, 32), dtype="float32")
            lv3 = R.call_tir("conv1x1", (x, w1), (32, 32), dtype="float32")
            lv4 = R.call_tir("bias_add", (lv3, bias1), (32, 32), dtype="float32")
            lv5 = R.call_tir("my_relu", (lv4), (32, 32), dtype="float32")
            lv6 = R.call_tir("concat", (lv2, lv5), (32, 64), dtype="float32")
            R.output(lv6)
        return lv6


def test_single_cbr():
    with PatternContext() as ctx:
        (
            is_call_tir_extern("conv1x1")
            >> is_call_tir_extern("bias_add")
            >> is_call_tir_extern("my_relu")
        )
        dfb = CBRx2["main"].body.blocks[0]
        matched = ctx.match_dfb(dfb)
        assert matched

    with PatternContext() as ctx:
        chain = (
            is_call_tir_extern("conv1x1")
            >> is_call_tir_extern("bias_add")
            >> is_call_tir_extern("my_relu")
        )
        dfb = CBRx2["main"].body.blocks[0]
        # we want to specifically match the first CBR (lv0)
        matched = ctx.match_dfb(dfb, start_hint=dfb.bindings[0].var)
        assert matched
        assert matched[chain[0]] == dfb.bindings[0].var
        # we want to specifically match the second CBR (lv3)
        matched = ctx.match_dfb(dfb, start_hint=dfb.bindings[3].var)
        assert matched
        assert matched[chain[0]] == dfb.bindings[3].var


def test_counter_single_crb():
    with PatternContext() as ctx:
        (
            is_call_tir_extern("conv1x1")
            >> is_call_tir_extern("my_relu")
            >> is_call_tir_extern("bias_add")
        )
        dfb = CBRx2["main"].body.blocks[0]
        assert not ctx.match_dfb(dfb)
        # Quickly fails unpromising matches by assumiung `start_hint` must be matched by a pattern.
        # This is usually faster than the full match:
        # Full match: let one pattern to match -> all Var: complexity ~ #Var
        # must_include_hint: let `start_hint` to match -> all patterns: complexity ~ #patterns
        # Usually #patterns is much smaller than #Var, so this is faster.
        assert not ctx.match_dfb(dfb, start_hint=dfb.bindings[0].var, must_include_hint=True)


def test_nested_context():
    dfb = CBRx2["main"].body.blocks[0]
    with PatternContext() as ctx0:
        (
            is_call_tir_extern("conv1x1")
            >> is_call_tir_extern("bias_add")
            >> is_call_tir_extern("my_relu")
        )
        with PatternContext() as ctx1:
            is_call_tir_extern("conv1x1") >> is_call_tir_extern("my_relu")  # pattern to miss
            with PatternContext() as ctx2:
                is_call_tir_extern("bias_add") >> is_call_tir_extern("my_relu")
                assert ctx2.match_dfb(dfb)
                assert PatternContext.current() == ctx2
            assert not ctx1.match_dfb(dfb)
            assert PatternContext.current() == ctx1
        assert ctx0.match_dfb(dfb)
        assert PatternContext.current() == ctx0


def test_two_cbr():
    with PatternContext() as ctx:
        cbr0 = (
            is_call_tir_extern("conv1x1")
            >> is_call_tir_extern("bias_add")
            >> is_call_tir_extern("my_relu")
        )
        cbr1 = cbr0.dup()

        assert cbr0.patterns[0] != cbr1.patterns[0]
        assert cbr0.patterns[1] != cbr1.patterns[1]
        assert cbr0.patterns[2] != cbr1.patterns[2]

        is_var("x").fork_to(cbr0, cbr1)
        dfb = CBRx2["main"].body.blocks[0]
        assert ctx.match_dfb(dfb)

    with PatternContext() as ctx:
        # Deny the pattern
        cbr0 = (
            is_call_tir_extern("conv1x1")
            >> is_call_tir_extern("bias_add")
            >> is_call_tir_extern("my_relu")
        )
        cbr1 = cbr0.dup()

        # input has no fork at y.
        is_var("y").fork_to(cbr0, cbr1)
        dfb = CBRx2["main"].body.blocks[0]
        assert not ctx.match_dfb(dfb)


def test_two_matmul():
    # Same as Figure 2(a) in TASO paper.
    @tvm.script.ir_module
    class MatMul2:
        @R.function
        def main(
            a: R.Tensor((32, 16), "float32"),
            b: R.Tensor((16, 48), "float32"),
            c: R.Tensor((48, 32), "float32"),
        ) -> R.Tensor:
            with R.dataflow():
                lv0 = R.call_tir("matmul", (a, b), (32, 48), dtype="float32")
                lv1 = R.call_tir("matmul", (lv0, c), (32, 32), dtype="float32")
                R.output(lv1)
            return lv1

    with PatternContext() as ctx:
        is_call_tir_extern("matmul") >> is_call_tir_extern("matmul")
        dfb = MatMul2["main"].body.blocks[0]
        assert ctx.match_dfb(dfb)

    with PatternContext() as ctx:
        is_call_tir_extern("matmul").has_shape([32, 48]) >> is_call_tir_extern("matmul").has_shape(
            [32, 32]
        )
        dfb = MatMul2["main"].body.blocks[0]
        assert ctx.match_dfb(dfb)

    with PatternContext() as ctx:
        is_call_tir_extern("matmul") >> is_call_tir_extern("matmul") >> is_call_tir_extern("matmul")
        dfb = MatMul2["main"].body.blocks[0]
        # Three MatMul cannot match
        assert not ctx.match_dfb(dfb)


def test_concat_mm_split():
    # Same as Figure 2(b) in TASO paper.
    @tvm.script.ir_module
    class CMS:
        @R.function
        def main(
            a: R.Tensor((32, 32), "float32"),
            b: R.Tensor((16, 32), "float32"),
            c: R.Tensor((16, 32), "float32"),
        ) -> R.Tensor:
            with R.dataflow():
                lv0 = R.call_tir("my_concat", (b, c), (32, 32), dtype="float32")
                lv1 = R.call_tir("my_matmul", (a, lv0), (32, 32), dtype="float32")
                lv2 = R.call_tir(
                    "my_split", (lv1,), ((16, 32), (16, 32)), dtype=("float32", "float32")
                )
                lv3 = R.TupleGetItem(lv2, 0)
                lv4 = R.TupleGetItem(lv2, 1)
                lv5 = R.add(lv3, lv4)
                R.output(lv5)
            return lv5

    with PatternContext() as ctx:
        (
            is_call_tir_extern("my_concat")
            >> is_call_tir_extern("my_matmul")
            >> is_call_tir_extern("my_split")
        )
        dfb = CMS["main"].body.blocks[0]
        assert ctx.match_dfb(dfb)

    with PatternContext() as ctx:
        split = is_call_tir_extern("my_split")
        lv3 = TupleGetItemPattern(split, 0).has_shape([16, 32])
        lv4 = TupleGetItemPattern(split, 1).has_shape([16, 32])
        split.fork_to(lv3, lv4)
        add = is_op("relax.add")(lv3, lv4)
        # TODO(@ganler): simplify this through implicit graph pattern.
        lv3 >> add
        lv4 >> add

        dfb = CMS["main"].body.blocks[0]
        assert ctx.match_dfb(dfb)


def test_self_attention():
    # The example comes from.
    # https://developer.nvidia.com/blog/nlu-with-tensorrt-bert/
    @tvm.script.ir_module
    class SelfAttention:
        @R.function
        def main(
            x: R.Tensor(("b", "s", "n", "h"), "float32"),
            wq: R.Tensor(("h", "h"), "float32"),
            wk: R.Tensor(("h", "h"), "float32"),
            wv: R.Tensor(("h", "h"), "float32"),
        ) -> R.Tensor:
            b, s, n, h = T.var("int64"), T.var("int64"), T.var("int64"), T.var("int64")
            with R.dataflow():
                fcq = R.call_tir("my_fc", (x, wq), (b, s, n, h), dtype="float32")
                tpq = R.call_tir("my_transpose", (fcq,), (b, s, h, n), dtype="float32")

                fck = R.call_tir("my_fc", (x, wk), (b, s, n, h), dtype="float32")
                tpk = R.call_tir("my_transpose", (fck,), (b, s, h, n), dtype="float32")

                mul = R.multiply(tpq, tpk)
                scale = R.multiply(mul, R.const(1.1, "float32"))
                softmax = R.call_tir("softmax", (scale,), (b, s, n, h), dtype="float32")

                fcv = R.call_tir("my_fc", (x, wv), (b, s, n, h), dtype="float32")
                tpv = R.call_tir("my_transpose", (fcv,), (b, s, h, n), dtype="float32")

                out = R.multiply(softmax, tpv)
                R.output(out)

            return out

    with PatternContext() as ctx:
        fc_trans_q = is_call_tir_extern("my_fc") >> is_call_tir_extern("my_transpose")
        fc_trans_k = fc_trans_q.dup()
        fc_trans_v = fc_trans_q.dup()

        is_var("x").fork_to(fc_trans_q, fc_trans_k, fc_trans_v)
        dfb = SelfAttention["main"].body.blocks[0]
        assert ctx.match_dfb(dfb)


def test_nested_diamond():
    @tvm.script.ir_module
    class DiamondInDiamond:
        @R.function
        def main(x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                #   matmul0      matmul1
                #     /    \    /    \
                # sigmoid2  add4  sigmoid3
                #     \    /    \    /
                #      add5      add6
                #          \    /
                #           add7
                lv0 = R.call_tir("tir_matmul", (x, w), (32, 32), dtype="float32")
                lv1 = R.call_tir("tir_matmul", (x, w), (32, 32), dtype="float32")
                lv2 = R.call_tir("tir_sigmoid", (lv0), (32, 32), dtype="float32")
                lv3 = R.call_tir("tir_sigmoid", (lv1), (32, 32), dtype="float32")
                lv4 = R.call_tir("tir_add", (lv0, lv1), (32, 32), dtype="float32")
                lv5 = R.call_tir("tir_add", (lv2, lv4), (32, 32), dtype="float32")
                lv6 = R.call_tir("tir_add", (lv3, lv4), (32, 32), dtype="float32")
                lv7 = R.call_tir("tir_add", (lv5, lv6), (32, 32), dtype="float32")
                R.output(lv7)
            return lv7

    # match matmul0 diamond
    with PatternContext() as ctx:
        sigmoid2 = is_call_tir_extern("tir_sigmoid")
        add4 = is_call_tir_extern("tir_add")
        is_call_tir_extern("tir_matmul").fork_to(sigmoid2, add4)
        add5 = is_call_tir_extern("tir_add")
        sigmoid2 >> add5
        add4 ^ add5
        assert ctx.match_dfb(DiamondInDiamond["main"].body.blocks[0])

    # counter case: mis-match matmul0 diamond
    with PatternContext() as ctx:
        sigmoid2 = is_call_tir_extern("tir_sigmoid")
        add4 = is_call_tir_extern("tir_add")
        is_call_tir_extern("tir_matmul").fork_to(sigmoid2, add4)
        add5 = is_call_tir_extern("tir_add")
        sigmoid2 >> add5
        add4 >> add5  # not only-used-by relation
        assert not ctx.match_dfb(DiamondInDiamond["main"].body.blocks[0])

    # match matmul1 diamond
    with PatternContext() as ctx:
        sigmoid3 = is_call_tir_extern("tir_sigmoid")
        add4 = is_call_tir_extern("tir_add")
        is_call_tir_extern("tir_matmul").fork_to(sigmoid3, add4)
        add6 = is_call_tir_extern("tir_add")
        sigmoid3 >> add6
        add4 ^ add6
        assert ctx.match_dfb(DiamondInDiamond["main"].body.blocks[0])

    # match add-4-5-6-7
    with PatternContext() as ctx:
        add5, add6, add7 = (
            is_call_tir_extern("tir_add"),
            is_call_tir_extern("tir_add"),
            is_call_tir_extern("tir_add"),
        )
        is_call_tir_extern("tir_add").fork_to(add5, add6)  # add4
        add5 >> add7
        add6 >> add7
        assert ctx.match_dfb(DiamondInDiamond["main"].body.blocks[0])


def test_incremental_solving():
    @R.function
    def simple_chain(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        with R.dataflow():
            # relu -> sigmoid -> neg
            lv0 = R.call_tir("tir_relu", (x), (32, 32), dtype="float32")
            lv1 = R.call_tir("tir_sigmoid", (lv0), (32, 32), dtype="float32")
            lv2 = R.call_tir("tir_neg", (lv1), (32, 32), dtype="float32")
            R.output(lv2)
        return lv2

    relu = is_call_tir_extern("tir_relu")
    sigmoid = is_call_tir_extern("tir_sigmoid")
    neg = is_call_tir_extern("tir_neg")

    with PatternContext() as ctx0:
        relu >> sigmoid
        with PatternContext(incremental=True) as ctx1:
            # because we are doing incremental solving
            # relu >> sigmoid is still a constraint in this context.
            # that said the total constraint is:
            # relu >> sigmoid >> neg
            sigmoid >> neg
            assert ctx1.match_dfb(simple_chain.body.blocks[0])

        # match relue -> sigmoid
        assert ctx0.match_dfb(simple_chain.body.blocks[0])


def test_incremental_solving_counter():
    @R.function
    def simple_chain(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        with R.dataflow():
            # sigmoid -> neg
            lv0 = R.call_tir("tir_sigmoid", (x), (32, 32), dtype="float32")
            lv1 = R.call_tir("tir_neg", (lv0), (32, 32), dtype="float32")
            R.output(lv1)
        return lv1

    relu = is_call_tir_extern("tir_relu")
    sigmoid = is_call_tir_extern("tir_sigmoid")
    neg = is_call_tir_extern("tir_neg")

    with PatternContext() as ctx0:
        relu >> sigmoid  # cannot match

        with PatternContext(incremental=False) as ctx1:
            # total constraint: sigmoid >> neg
            sigmoid >> neg
            assert ctx1.match_dfb(simple_chain.body.blocks[0])

        with PatternContext(incremental=True) as ctx1:
            # total constraint: relu >> sigmoid >> neg
            sigmoid >> neg
            assert not ctx1.match_dfb(simple_chain.body.blocks[0])


if __name__ == "__main__":
    tvm.testing.main()
