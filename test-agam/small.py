import tvm
from tvm import relax
from tvm.script import relax as R
from tvm.script.parser.relax.entry import Tensor

@tvm.script.ir_module
class SmallModel:
    @R.function
    def main(x: Tensor, y: Tensor):
        a = x
        b = y
        c = R.add(a, b)
        d = R.add(a, b)
        e = R.add(c, d)
        # R.output(e)
        return e
