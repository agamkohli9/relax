""" Compiles the model given in model.py """ 

import tvm
from tvm import relax, relay
from tvm.script import relax as R
from tvm.script.parser.relax.entry import Tensor
import numpy as np
from small import SmallModel

if __name__ == '__main__':
    target = tvm.target.Target('llvm')
    device = tvm.cpu()

    with open('small.relax', 'w') as f:
        print(SmallModel, file=f)

    mod = relax.transform.FoldConstant()(SmallModel)

    for _ in range(10):
        mod = relax.transform.FoldConstant()(mod)

    with open('small.optimized.relax', 'w') as f:
        print(mod, file=f)

    print('Done!')