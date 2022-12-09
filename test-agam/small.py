import tvm
from tvm import relax
from tvm.script import relax as R
from tvm.script.parser.relax.entry import Tensor

@tvm.script.ir_module
class SmallModel:
    @R.function
    def main(x: Tensor, y: Tensor):
            a = R.const(12)
            b = R.const(12)
            c = R.add(b, b)
            d = R.add(b, b) # = c
            e = R.add(c, d)
            return e