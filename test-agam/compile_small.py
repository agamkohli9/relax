""" Compiles the model given in model.py """ 

import tvm
from tvm import relax, relay
from tvm.script import relax as R
from tvm.script.parser.relax.entry import Tensor
from small import SmallModel

def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body

def small_model(x: Tensor, y: Tensor):
    a = x
    b = y
    c = R.add(a, b)
    d = R.add(a, b) # Should get removed
    e = R.add(c, d)
    return e

if __name__ == '__main__':
    with open('small.relax', 'w') as f:
        print(SmallModel, file=f)

    optimized_mod = relax.transform.FoldConstant()(SmallModel)
    optimized_mod = relax.transform.FoldConstant()(optimized_mod)

    with open('small.optimized.relax', 'w') as f:
        print(optimized_mod, file=f)

print('Compilation Done!')