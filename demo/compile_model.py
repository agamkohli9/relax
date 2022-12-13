""" Compiles the model given in model.py """ 

import tvm
from tvm import relay, relax
from tvm.relax import testing
import torch
import numpy as np
from model import Model

if __name__ == '__main__':
    # TODO: Get this to run with cuda
    # Config device
    target = tvm.target.Target('llvm')
    device = tvm.cpu()

    # Config input model
    input_size = 2 # get from Netron
    input_shape = (input_size)
    input_data = torch.randn(input_shape)
    minibatch_size = 5

    # Load model as Relax IR
    model = Model()
    model.load_state_dict(torch.load('model.pt'))
    model = torch.jit.trace(model, input_data).eval()
    mod, _ = relay.frontend.from_pytorch(model, [('input', input_data.shape)])
    mod = relax.testing.relay_translator.from_relay(mod['main'], target)

    # Write unoptimized model
    with open('model.relax', 'w') as f:
        print(mod.script(), file=f)

    # Run the unoptimized model
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, device)

    # init parameters
    params = tvm.relax.testing.nn.init_params(mod)

    data = tvm.nd.array(np.random.rand(minibatch_size, input_size).astype(np.float32))

    res = vm["main"](*params)

    # Do our optimization pass
    mod = relax.transform.FoldConstant()(mod)

    # Write optimized model
    with open('model.optimized.relax', 'w') as f:
        print(mod.script(), file=f)

    # Run the optimized model
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, device)

    # init parameters
    params = tvm.relax.testing.nn.init_params(mod)

    data = tvm.nd.array(np.random.rand(minibatch_size, input_size).astype(np.float32))

    res_optimized = vm["main"](*params)

    # Assert that unoptimizized and optimized models are structurally equal
    print(res)
    print(res_optimized)

    print('Done!')