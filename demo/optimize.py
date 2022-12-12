import os
import sys, inspect, importlib

import tvm
from tvm import relax, tir, topi
from tvm.runtime import container
from tvm.target.target import Target
from tvm.relax.testing import nn

import tvm.script
from tvm.script import tir as T, relax as R

from utils import bcolors, log

OUTPUT_DIR = "output"

def get_modules():
    mods = []
    for name, cls in inspect.getmembers(importlib.import_module("modules")):
        if name.startswith("Module"):
            mods.append((name, cls))
    return mods

def save_model(model, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    log(f"Saving model to {filepath}", bcolors.OKCYAN)
    with open(filepath, "w") as f:
        print(model, file=f)


def optimize_and_save_model(name, mod_in):
    print("Compiling")

    # Save original for reference
    mod = mod_in
    save_model(mod, f'{name}-raw.relax')

    # Optimize
    mod_opt = mod
    for _ in range(10):
        mod_opt = relax.transform.FoldConstant()(mod_opt)
    
    # Save 
    save_model(mod_opt, f'{name}-opt.relax')
    log("Done", bcolors.OKGREEN)


def main():
    modules = get_modules()

    for mod in modules:
        print("name:", mod[0])
        print("mod:", mod[1 ])
        optimize_and_save_model(mod[0], mod[1])

if __name__ == '__main__':
    main()
