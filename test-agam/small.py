import tvm
from tvm import relax
from tvm.script import relax as R
from tvm.script.parser.relax.entry import Tensor

@tvm.script.ir_module
class SmallModel:
    @R.function
    def main(x: Tensor, y: Tensor):
        y = R.add(x, y)
        o = R.add(x, y)
        return o
