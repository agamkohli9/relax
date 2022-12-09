""" Compiles the model given in model.py """ 

import tvm
from tvm import relax, relay
from small import SmallModel

if __name__ == '__main__':
    with open('small.relax', 'w') as f:
        print(SmallModel, file=f)

    optimized_mod = relax.transform.CommonSubexpressionElimination()(SmallModel)

    with open('small.optimized.relax', 'w') as f:
        print(optimized_mod, file=f)
