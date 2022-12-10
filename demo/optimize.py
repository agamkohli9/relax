import os

import tvm
from tvm import relax, tir, topi
from tvm.runtime import container
from tvm.target.target import Target
from tvm.relax.testing import nn

import tvm.script
from tvm.script import tir as T, relax as R

from utils import bcolors, log
from model import Module

OUTPUT_DIR = "output"

MODEL_RAW = "model-raw.relax"
MODEL_OPT_RELAX = "model-opt.relax"


def save_model(model, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    log(f"Saving model to {filepath}", bcolors.OKCYAN)
    with open(filepath, "w") as f:
        print(model, file=f)


def compile():
    print("Compiling")

    # Save original for reference
    mod = Module
    save_model(mod, MODEL_RAW)

    # Save optimized
    mod_opt = relax.transform.FoldConstant()(mod)
    mod_opt = relax.transform.FoldConstant()(mod_opt)
    save_model(mod_opt, MODEL_OPT_RELAX)

    builder = relax.BlockBuilder()

    with builder.function(name="main"):
        model = Module
        
        # n is a symbolic variable to represent a dynamic batch size
        n = tir.Var("n", "int64")
        data = nn.Placeholder((n, input_size), name="data")
        output = model(data)
        params = [data] + model.parameters()
        builder.emit_func_output(output, params=params) 


     
    # # Get and print the IRModule being built.
    # mod = builder.get()
    # mod.show()


    # Build and create vm executor
    log("Build and create vm executor", bcolors.OKBLUE)

    target = tvm.target.Target("cuda")
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    # Init parameters
    log("Init parameters", bcolors.OKBLUE)

    params = nn.init_params(mod)
    print("params", params)

    res = vm["main"](None, *params)
    print(res)

    log("Done compiling", bcolors.OKGREEN)


if __name__ == '__main__':
    compile()
