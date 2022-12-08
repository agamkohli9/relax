import tvm
from tvm import relax
from tvm.script import relax as R
from tvm.script.parser.relax.entry import Tensor

@tvm.script.ir_module
class SmallModel:
    @R.function
    def main(x: Tensor, y: Tensor):
        with R.dataflow():
            a = x
            b = y
            c = relax.add(a, b)
            d = relax.add(a, b)
            e = relax.add(c, d)
            R.output(e)
        return e
