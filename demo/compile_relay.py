""" Compiles the model given in model.py """ 

import tvm
from tvm import relax, relay
from tvm.script import relax as R
from tvm.script.parser.relax.entry import Tensor
import numpy as np

if __name__ == '__main__':
    # TODO: Get this to run with cuda
    # Config device
    target = tvm.target.Target('llvm')
    device = tvm.cpu()

    with open('small.relax', 'w') as f:
        print(SmallModel, file=f)

    mod = relax.transform.FoldConstant()(SmallModel)

    with open('small.optimized.relax', 'w') as f:
        print(mod, file=f)

    # Assert that unoptimizized and optimized models are structurally equal
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, device)

    # init parameters
    params = tvm.relax.testing.nn.init_params(mod)

    data = tvm.nd.array(np.random.rand(1, 1).astype(np.float32))

    res = vm["main"](data, *params)
    print(res)

    print('Done!')