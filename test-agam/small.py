import tvm
from tvm import relax
from tvm.script import relax as R
from tvm.script.parser.relax.entry import Tensor

@tvm.script.ir_module
class SmallModel:
    @R.function
    def main():
        a = R.const(69)
        b = R.const(69)
        c = R.add(a, b)
        d = R.add(a, b) # Should get removed
        e = R.add(c, d)
        return e
