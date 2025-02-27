import sys, inspect
import tvm
from tvm.script import relax as R

@tvm.script.ir_module
class ModuleBasic:
    @R.function
    def main():
        res = R.add(R.const(30), R.const(40))
        return res

@tvm.script.ir_module
class ModuleBasic2:
    @R.function
    def main():
        a = R.const(20)
        b = R.add(a, a)
        c = R.add(a, R.const(20))
        d = R.add(R.const(20), R.const(20))
        return d

@tvm.script.ir_module
class ModuleMultipleIters:
    @R.function
    def main():
        a = R.const(20)
        b = R.const(40)
        c = R.add(a, b)
        d = R.add(R.const(40), c)
        return d

@tvm.script.ir_module
class ModuleMultPowerOfTwo:
    @R.function
    def main():
        a = R.const(50)
        b = R.const(32) # note this is a power of 2
        c = R.multiply(a, b)
        return c

@tvm.script.ir_module
class ModuleMultPowerOfTwo:
    @R.function
    def main(a):
        b = R.const(32) # note this is a power of 2
        c = R.multiply(a, b) # 'a' is an arbitrary input (scaler, tensor, etc.)
        return c

# TODO: Resolve below error
# error: module 'tvm.script.parser.relax' has no attribute 'divide'
#
# @tvm.script.ir_module
# class ModuleDivPowerOfTwo:
#     @R.function
#     def main():
#         a = R.const(64)
#         b = R.const(8) # note this is a power of 2
#         c = R.divide(a, b)
#         return c

