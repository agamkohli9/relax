import tvm
from tvm import relax
from tvm.script import relax as R
from tvm.script.parser.relax.entry import Tensor

@tvm.script.ir_module
class SmallModel:
    @R.function
    def main():
        a = R.add(R.const(69), R.const(69))
        a = R.multiply(R.const(69), R.const(0))
        a = R.multiply(R.const(69), R.const(1))
        a = R.add(R.const(69), R.const(0))
        return a
