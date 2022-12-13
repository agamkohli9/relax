""" Compiles Resnet 18 """

import tvm
from tvm import relay, relax
from tvm.relay.testing import resnet
from tvm.relax.testing import relay_translator
import numpy as np

if __name__ == '__main__':
    target = tvm.target.Target('llvm')
    device = tvm.cpu()

    # Assume inputs are RGB color images of size 224 * 224.
    batch_size = 1
    num_class = 1000
    input_shape = (3, 224, 224)
    data_shape = (batch_size,) + input_shape
    out_shape = (batch_size, num_class)
    minibatch_size = 5

    # Load Resnet 18 as Relax IR
    mod, params = resnet.get_workload(
        num_layers=18, batch_size=batch_size, input_shape=input_shape
    )
    mod = relay_translator.from_relay(mod['main'], target)

    # Write unoptimized Resnet18 model
    with open('resnet.relax', 'w') as f:
        print(mod.script(), file=f)
        pass

    # Do our Constant Folding optimization pass
    optimized_mod = relax.transform.FoldConstant()(mod)

    # Write optimized Resnet18 model
    with open('resnet.optimized.relax', 'w') as f:
        print(optimized_mod.script(), file=f)
        pass

    # Assert that unoptimizized and optimized models are structurally equal
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, device)

    # init parameters
    params = tvm.relax.testing.nn.init_params(mod)

    data = tvm.nd.array(np.random.rand(minibatch_size, 244).astype(np.float32))

    res = vm["main"](data, *params)
    print(res)

    print('Done!')