""" Compiles the model given in model.py """ 

import tvm
from tvm import relax, relay
from tvm.script import relax as R
from tvm.script.parser.relax.entry import Tensor

def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body

def small_relay_model():
    a = relay.const(69)
    b = relay.const(69)
    c = relay.add(a, b)
    d = relay.add(a, b) # Should get removed
    e = relay.add(c, d)
    return e

def small_model():
    a = R.const(69)
    b = R.const(69)
    c = R.add(a, b)
    d = R.add(a, b) # Should get removed
    e = R.add(c, d)
    return e

if __name__ == '__main__':
    mod = small_relay_model()
    #mod = small_model()

    with open('small.relax', 'w') as f:
        print(mod, file=f)

    optimized_mod = run_opt_pass(mod, relay.transform.EliminateCommonSubexpr())
    #optimized_mod = run_opt_pass(mod, relax.transform.CommonSubexpressionElimination())

    with open('small.optimized.relax', 'w') as f:
        print(optimized_mod, file=f)

print('Compilation Done!')