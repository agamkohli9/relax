""" Compiles the model given in model.py """ 

import tvm
from tvm import relax, relay
from small import SmallModel

if __name__ == '__main__':
    # Load model as Relax IR
    #mod = SmallModel()

    with open('small.relax', 'w') as f:
        print(SmallModel, file=f)

    # Do some dummy optimization pass
    mod = relay.transform.EliminateCommonSubexpr()(SmallModel)
    #mod = relax.transform.FlashAttention()(mod)

    with open('small.optimized.relax', 'w') as f:
        print(mod, file=f)
