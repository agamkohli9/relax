from __future__ import annotations  # must import to defer parsing of annotations
import os
import numpy as np
import tvm
from tvm import relax, tir, topi
from tvm.runtime import container
from tvm.target.target import Target
from tvm.relax.testing import nn
from small import SmallModel

import tvm.script
from tvm.script import tir as T, relax as R

target = tvm.target.Target("llvm")
device = tvm.cpu()

builder = relax.BlockBuilder()

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

with builder.function(name="main"):
        # model = nn.Sequential(
        #     nn.Linear(input_size, hidden_sizes[0]),
        #     nn.ReLU(),
        #     nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        #     nn.ReLU(),
        #     nn.Linear(hidden_sizes[1], output_size),
        #     nn.LogSoftmax(),
        # )
        model = SmallModel
        
        # # n is a symbolic variable to represent a dynamic batch size
        # n = tir.Var("n", "int64")
        # data = nn.Placeholder((n, input_size), name="data")
        output = model()
        params = [None] + model.parameters()
        builder.emit_func_output(output, params=params) 

mod = builder.get()

# build and create vm executor
ex = relax.vm.build(mod, target)
vm = relax.VirtualMachine(ex, device)

# init parameters
params = nn.init_params(mod)
# the input data has a minibatch size of 3
# data = tvm.nd.array(np.random.rand(3, input_size).astype(np.float32))

res = vm["main"]([], *params)
print(res)
