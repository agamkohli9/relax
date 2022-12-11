import tvm
from tvm import relax
from tvm.script import relax as R
from tvm.script.parser.relax.entry import Tensor

@tvm.script.ir_module
class SmallModel:
    @R.function
    def main(x: Tensor):
        a = R.add(R.const(69), R.const(69))
        b = R.multiply(x, R.const(0))
        c = R.multiply(x, R.const(1))
        d = R.add(a, a)
        e = R.multiply(x, R.const(4))
        return e
