import tvm
from tvm import relax
from tvm.script import relax as R
from tvm.script.parser.relax.entry import Tensor

@tvm.script.ir_module
class SmallModel:
    @R.function
    def main(x: Tensor, y: Tensor):
            a = 12
            b = 3
            c = R.add(b, b)
            d = R.add(b, b)
            e = R.add(c, d)
            return e