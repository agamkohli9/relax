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

import tempfile

import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm import relax, relay
from tvm.ir.base import assert_structural_equal
from tvm.relax.testing import relay_translator
from tvm.relay import testing
from tvm.runtime import vm
from tvm.script import tir as T
from tvm.target import Target


def get_resnet(batch_size, dtype, layout, image_shape):
    relay_mod, params = testing.resnet.get_workload(
        num_layers=18,
        batch_size=batch_size,
        dtype=dtype,
        layout=layout,
        image_shape=image_shape,
    )

    return relay_mod, params


def relay_build_and_run(mod, target, dev, params, data):
    with tempfile.TemporaryDirectory() as work_dir:
        db = ms.relay_integration.tune_relay(
            mod=mod,
            params=params,
            target=target,
            num_trials_per_iter=32,
            max_trials_per_task=32,
            max_trials_global=1024,
            task_scheduler="round-robin",
            work_dir=work_dir,
        )
        ex = ms.relay_integration.compile_relay(
            db,
            mod=mod,
            target=target,
            params=params,
        )
    rt_mod = tvm.contrib.graph_executor.GraphModule(ex["default"](dev))
    rt_mod.set_input("data", data)
    rt_mod.run()
    out = rt_mod.get_output(0).numpy()
    return ex, rt_mod, out


def relax_build_and_run(mod, target, dev, params, data):
    mod = relax.transform.BindParams("main", params)(mod)
    with tempfile.TemporaryDirectory() as work_dir:
        db = ms.relax_integration.tune_relax(
            mod=mod,
            target=target,
            task_scheduler="round-robin",
            num_trials_per_iter=32,
            max_trials_per_task=32,
            max_trials_global=1024,
            work_dir=work_dir,
        )
        ex = ms.relax_integration.compile_relax(
            db,
            mod=mod,
            target=target,
            params=params,
        )
    vm = relax.VirtualMachine(ex, dev)
    res = vm["main"](data)
    out = res.numpy()
    return ex, vm, out


def verify_e2e_translation(target_str, layout, batch_size, image_shape):
    target = Target(target_str)
    dev = tvm.device(str(target), dev_id=0)
    relay_mod, params = get_resnet(batch_size, "float32", layout, image_shape)
    input_shape = (1, *image_shape)
    data = tvm.nd.array(np.random.rand(*input_shape).astype(np.float32), dev)
    relax_mod = relay_translator.from_relay(relay_mod["main"], target, params)
    assert relax_mod["main"].attrs["global_symbol"] == "main"

    _, _, relay_out = relay_build_and_run(relay_mod, target, dev, params, data)
    _, _, relax_out = relax_build_and_run(relax_mod, target, dev, params, data)
    tvm.testing.assert_allclose(relay_out, relax_out, atol=1e-5, rtol=1e-5)


@pytest.mark.skip(reason="take too much time")
@pytest.mark.parametrize(
    "layout, batch_size, image_shape", [("NCHW", 1, (3, 224, 224)), ("NHWC", 1, (224, 224, 3))]
)
def test_verify_e2e_translation_cpu(layout, batch_size, image_shape):
    verify_e2e_translation("llvm --num-cores=16", layout, batch_size, image_shape)


@pytest.mark.skip(reason="take too much time")
@tvm.testing.requires_gpu
@pytest.mark.parametrize(
    "layout, batch_size, image_shape", [("NCHW", 1, (3, 224, 224)), ("NHWC", 1, (224, 224, 3))]
)
def test_verify_e2e_translation_gpu(layout, batch_size, image_shape):
    verify_e2e_translation("cuda", layout, batch_size, image_shape)


def verify_extracted_tasks(target_str, layout, batch_size, image_shape):
    target = Target(target_str)
    relay_mod, params = get_resnet(batch_size, "float32", layout, image_shape)
    relax_mod = relay_translator.from_relay(
        relay_mod["main"],
        target,
        params,
        pass_config={
            "relay.backend.use_meta_schedule": True,
            "relay.FuseOps.max_depth": 1,  # Disable relay fusion
        },
    )
    relay_tasks = ms.relay_integration.extract_tasks(
        relay_mod,
        target=target,
        params=params,
        pass_config={
            "relay.backend.use_meta_schedule": True,
            "relay.FuseOps.max_depth": 1,  # Disable relay fusion
        },
    )
    relax_tasks = ms.relax_integration.extract_tasks(
        relax_mod,
        target=target,
        params=params,
    )
    # TODO (yongwww, yuchen): tophub guides relay passes, which causes inconsistent tasks
    # assert len(relay_tasks) == len(relax_tasks)
    # TODO: Can we compare extracted tasks as well?


@pytest.mark.parametrize(
    "layout, batch_size, image_shape",
    [
        ("NCHW", 1, (3, 224, 224)),
        ("NHWC", 1, (224, 224, 3)),
    ],
)
def test_verify_extracted_tasks_cpu(layout, batch_size, image_shape):
    verify_extracted_tasks("llvm --num-cores=16", layout, batch_size, image_shape)


@tvm.testing.requires_gpu
@pytest.mark.parametrize(
    "layout, batch_size, image_shape", [("NCHW", 1, (3, 224, 224)), ("NHWC", 1, (224, 224, 3))]
)
def test_verify_extracted_tasks_gpu(layout, batch_size, image_shape):
    verify_extracted_tasks("cuda", layout, batch_size, image_shape)


def translate_and_build_vms(relay_mod, target_str="llvm", translate_op_with_tir=None):
    target = tvm.target.Target(target_str)

    # build the relay IRModule and create relay vm
    relay_ex = relay.vm.compile(relay_mod, target)
    relay_vm = vm.VirtualMachine(relay_ex, tvm.cpu())

    # build the relax IRModule and create relax vm
    relax_mod = relay_translator.from_relay(
        relay_mod["main"], target, translate_op_with_tir=translate_op_with_tir
    )
    relax_ex = relax.vm.build(relax_mod, target)
    relax_vm = relax.VirtualMachine(relax_ex, tvm.cpu())

    return relay_vm, relax_vm, relax_mod


def verify_vm_outputs(
    input_shape,
    relay_vm,
    relax_vm,
    extra_args=[],
):
    input = tvm.nd.array(np.random.rand(*input_shape).astype(np.float32))

    # check correctness by comparing relax and relay result
    args = [input] + extra_args
    relax_output = relax_vm["main"](*args)
    relay_output = relay_vm.run(*args)
    tvm.testing.assert_allclose(relay_output.numpy(), relax_output.numpy())


def test_single_dynamic_dim():
    wx, wy = 64, 128
    # create relay module: y = data * weights + bias with dynamic batch dimension
    data = relay.var("data", shape=(relay.Any(), wx))
    weights = relay.var("weights", shape=(wx, wy))
    bias = relay.var("bias", shape=(wy,))
    y = relay.nn.matmul(data, weights)
    relay_mod = tvm.IRModule.from_expr(relay.Function([data, weights, bias], y + bias))

    relay_vm, relax_vm, _ = translate_and_build_vms(relay_mod)
    weights = tvm.nd.array(np.random.rand(wx, wy).astype(np.float32))
    bias = tvm.nd.array(np.random.rand(wy).astype(np.float32))
    # verify for different batch sizes
    verify_vm_outputs([10, wx], relay_vm, relax_vm, [weights, bias])
    verify_vm_outputs([32, wx], relay_vm, relax_vm, [weights, bias])


def test_multiple_dynamic_dims():
    # create relay module: y = a + a, where a has shape = (?, 5, ?)
    shape = (relay.Any(), 5, relay.Any())
    a = relay.var("a", shape=shape)

    relay_mod = tvm.IRModule.from_expr(relay.Function([a], a + a))
    relay_vm, relax_vm, _ = translate_and_build_vms(relay_mod)
    # verify for different shapes
    verify_vm_outputs([2, 5, 10], relay_vm, relax_vm)
    verify_vm_outputs([12, 5, 24], relay_vm, relax_vm)


def test_layout_transform():
    shape = (1, 3, 224, 224)
    a = relay.var("a", shape=shape)
    b = relay.layout_transform(a, "NCHW", "NHWC")
    relay_mod = tvm.IRModule.from_expr(relay.Function([a], b))

    relay_vm, relax_vm, _ = translate_and_build_vms(relay_mod)
    verify_vm_outputs([1, 3, 224, 224], relay_vm, relax_vm)


def test_translate_op_with_tir():
    @T.prim_func
    def tir_matmul(
        A: T.Buffer[(512, 512), "float32"],
        B: T.Buffer[(512, 512), "float32"],
        C: T.Buffer[(512, 512), "float32"],
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "multiply", "tir.noalias": True})
        # body
        # with T.block("root")
        for i0, i1, i2 in T.grid(512, 512, 512):
            with T.block("C"):
                i, j, k = T.axis.remap("SSR", [i0, i1, i2])
                T.reads(C[i, j], A[i, k], B[k, j])
                T.writes(C[i, j])
                with T.init():
                    C[i, j] = T.float32(0)
                C[i, j] = C[i, j] + A[i, k] * B[k, j]

    shape = (512, 512)
    a = relay.var("a", shape=shape)

    relay_mod = tvm.IRModule.from_expr(relay.Function([a], a * a))
    _, _, relax_mod = translate_and_build_vms(
        relay_mod, translate_op_with_tir={"multiply": tir_matmul}
    )
    assert_structural_equal(relax_mod["multiply"], tir_matmul)


if __name__ == "__main__":
    pytest.main([__file__])
