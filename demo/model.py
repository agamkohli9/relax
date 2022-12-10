import tvm
from tvm.script import relax as R

@tvm.script.ir_module
class Module:
    @R.function
    def main():
        res = R.add(R.const(69), R.const(69))
        return res
