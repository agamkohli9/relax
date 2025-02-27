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
import tvm
import tvm.testing
from tvm import relax
import tvm.script
from tvm.script import relax as R, tir as T
from tvm.relax import transform
from tvm.ir.base import assert_structural_equal


def _check_equal(x, y):
    tvm.ir.assert_structural_equal(x, y)
    tvm.ir.assert_structural_equal(y, x)

    xhash = tvm.ir.structural_hash(x)
    yhash = tvm.ir.structural_hash(y)

    assert xhash == yhash


def _check_save_roundtrip(x):
    y = tvm.ir.load_json(tvm.ir.save_json(x))
    _check_equal(x, y)


def test_basic():
    # the target IRModule
    @tvm.script.ir_module
    class Expected:
        @R.function
        def lifted_func_0(
            x2: R.Tensor((10, 5), "float32"), y2: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            s: R.Tensor((10, 5), "float32") = R.add(x2, y2)
            return s

        @R.function
        def main(
            x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            inner = lifted_func_0
            gv1: R.Tensor((10, 5), "float32") = inner(x1, y1)
            return gv1

    @tvm.script.ir_module
    class Before:
        @R.function
        def main(
            x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            @R.function
            def inner(
                x2: R.Tensor((10, 5), "float32"), y2: R.Tensor((10, 5), "float32")
            ) -> R.Tensor((10, 5), "float32"):
                s: R.Tensor((10, 5), "float32") = R.add(x2, y2)
                return s

            gv1: R.Tensor((10, 5), "float32") = inner(x1, y1)
            return gv1

    before = Before
    expected = Expected
    # Perform Lambda Lifting
    after = transform.LambdaLift()(before)
    assert len(after.functions) == 2
    assert_structural_equal(after, expected, map_free_vars=True)
    _check_save_roundtrip(after)


def test_closure():
    # the expected IRModule
    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")):
            outer_func = lifted_func_0
            in_call = outer_func(x)
            res = R.invoke_closure(in_call, (y,), type_args=(R.Tensor(ndim=2, dtype="float32")))
            return res

        @R.function
        def lifted_func_1(x1: R.Tensor((2, 3), "float32"), c1: R.Tensor((2, 3), "float32")):
            r_1: R.Tensor((2, 3), "float32") = R.add(x1, c1)
            return r_1

        @R.function
        def lifted_func_0(y: R.Tensor((2, 3), "float32")) -> R.Object:
            inner_func = R.make_closure(lifted_func_1, (y,))
            return inner_func

    # IRModule to perform Lambda Lifting
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")
        ) -> R.Tensor((2, 3), "float32"):
            @R.function
            def outer_func(c1: R.Tensor((2, 3), "float32")):
                @R.function
                def inner_func(x1: R.Tensor((2, 3), "float32")):
                    s: R.Tensor((2, 3), "float32") = R.add(x1, c1)
                    return s

                return inner_func

            in_call = outer_func(x)
            res = in_call(y)
            return res

    before = Before
    after = transform.LambdaLift()(before)
    expected = Expected
    assert_structural_equal(after, expected, map_free_vars=True)
    _check_save_roundtrip(after)


@pytest.mark.skip(reason="Need fix after parser switch over")
def test_recursive():
    # the expected IRModule
    @tvm.script.ir_module
    class Expected:
        @R.function
        def lifted_func_0(
            i: R.Tensor((), "int32"), s: R.Tensor((2, 3), "float32"), x: R.Tensor((2, 3), "float32")
        ) -> R.Tensor((2, 3), "float32"):
            cond: R.Tensor((), "bool") = R.call_packed(
                "test.vm.less", i, R.const(10), type_args=(R.Tensor(ndim=0, dtype="bool"))
            )
            c: R.Tensor((), "int32") = R.const(1, dtype="int32")
            if cond:
                new_i: R.Tensor((), "int32") = R.add(i, c)
                new_s: R.Tensor((2, 3), "float32") = R.add(s, x)
                r = lifted_func_0(new_i, new_s, x)
            else:
                r = s
            return r

        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor:
            while_loop = R.make_closure(lifted_func_0, (x,))
            gv = R.invoke_closure(
                while_loop,
                (relax.const(0), x),
                type_args=(R.Tensor(ndim=2, dtype="float32")),
            )
            return gv

    # the IRModule to apply lambda lifting
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor:
            @R.function
            def while_loop(
                i: R.Tensor((), "int32"), s: R.Tensor((2, 3), "float32")
            ) -> R.Tensor((2, 3), "float32"):
                cond: R.Tensor((), "bool") = R.call_packed(
                    "test.vm.less", i, R.const(10), type_args=(R.Tensor(ndim=0, dtype="bool"))
                )
                c: R.Tensor((), "int32") = R.const(1, dtype="int32")
                if cond:
                    new_i: R.Tensor((), "int32") = R.add(i, c)
                    new_s: R.Tensor((2, 3), "float32") = R.add(s, x)
                    r: R.Tensor((2, 3), "float32") = while_loop(new_i, new_s)
                else:
                    r: R.Tensor((2, 3), "float32") = s
                return r

            gv: R.Tensor((2, 3), "float32") = while_loop(relax.const(0), x)
            return gv

    before = Before
    expected = Expected
    # Perform Lamda Lifting
    after = transform.LambdaLift()(before)
    assert len(after.functions) == 2
    assert_structural_equal(after, expected, map_free_vars=True)
    _check_save_roundtrip(after)


@pytest.mark.skip(reason="Need fix after parser switch over")
def test_multi_func():
    # expected IRModule
    @tvm.script.ir_module
    class Expected:
        @R.function
        def glob_func_1(
            x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            inner = lifted_func_1
            gv1 = inner(x1, y1)
            return gv1

        @R.function
        def glob_func_2(
            x11: R.Tensor((10, 5), "float32"), y11: R.Tensor((10, 5), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            inner1 = lifted_func_0
            gv11 = inner1(x11, y11)
            return gv11

        @R.function
        def lifted_func_0(
            x2: R.Tensor((10, 5), "float32"), y2: R.Tensor((10, 5), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            s: R.Tensor((10, 5), "float32") = R.add(x2, y2)
            return s

        @R.function
        def lifted_func_1(
            x21: R.Tensor((10, 5), "float32"), y21: R.Tensor((10, 5), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            s1: R.Tensor((10, 5), "float32") = R.add(x21, y21)
            return s1

    # the IRModule to apply lambda lifting
    @tvm.script.ir_module
    class Before:
        @R.function
        def glob_func_1(
            x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            @R.function
            def inner(
                x2: R.Tensor((10, 5), "float32"), y2: R.Tensor((10, 5), "float32")
            ) -> R.Tensor((10, 5), "float32"):
                s: R.Tensor((10, 5), "float32") = R.add(x2, y2)
                return s

            gv1: R.Tensor((10, 5), "float32") = inner(x1, y1)
            return gv1

        @R.function
        def glob_func_2(
            x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            @R.function
            def inner(
                x2: R.Tensor((10, 5), "float32"), y2: R.Tensor((10, 5), "float32")
            ) -> R.Tensor((10, 5), "float32"):
                s: R.Tensor((10, 5), "float32") = R.add(x2, y2)
                return s

            gv1: R.Tensor((10, 5), "float32") = inner(x1, y1)
            return gv1

    before = Before
    expected = Expected
    # Perform Lamda Lifting
    after = transform.LambdaLift()(before)
    assert len(after.functions) == 4
    assert_structural_equal(after, expected, map_free_vars=True)
    _check_save_roundtrip(after)


def test_no_local_func():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def sub(
            A: T.Buffer[(16, 16), "float32"],
            B: T.Buffer[(16, 16), "float32"],
            C: T.Buffer[(16, 16), "float32"],
        ) -> None:
            for i, j in T.grid(16, 16):
                with T.block("sub"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    C[vi, vj] = A[vi, vj] - B[vi, vj]

        @R.function
        def before(c0: R.Tensor((16, 16), "float32"), x: R.Tensor(dtype="float32", ndim=2)):
            s = R.call_tir(sub, (c0, x), (16, 16), dtype="float32")
            return s

    before = Before
    # Perform lambda lifting
    after = transform.LambdaLift()(before)
    # No local functions are lifted
    assert_structural_equal(after, before, map_free_vars=True)
    _check_save_roundtrip(after)


if __name__ == "__main__":
    tvm.testing.main()
