""" Compiles the model given in model.py """ 

import tvm
from tvm import relax, relay
from tvm.script import relax as R
from tvm.script.parser.relax.entry import Tensor
from small import SmallModel

if __name__ == '__main__':
    with open('small.relax', 'w') as f:
        print(SmallModel, file=f)

    optimized_mod = relax.transform.FoldConstant()(SmallModel)

    with open('small.optimized.relax', 'w') as f:
        print(optimized_mod, file=f)

print('Compilation Done!')