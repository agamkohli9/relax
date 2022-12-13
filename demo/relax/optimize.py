import os
import sys, inspect, importlib

import tvm
from tvm import relax, tir, topi
from tvm.runtime import container
from tvm.target.target import Target
from tvm.relax.testing import nn

import tvm.script
from tvm.script import tir as T, relax as R

from logger import bcolors, log

def get_modules():
    mods = []
    for name, cls in inspect.getmembers(importlib.import_module("relax.modules")):
        if name.startswith("Module"):
            mods.append((name, cls))
    return mods


def save_model(model, filepath):
    log(f"Saving model to {filepath}", bcolors.OKCYAN)
    with open(filepath, "w") as f:
        print(model, file=f)


def optimize_and_save_model(output_dir, mod_name, mod_in):
    
    filepath_raw = os.path.join(output_dir, f'{mod_name}-raw.relax')
    filepath_opt = os.path.join(output_dir, f'{mod_name}-opt.relax')

    # Save original for reference
    mod = mod_in
    save_model(mod, filepath_raw)

    # Optimize
    mod_opt = mod
    for _ in range(10):
        mod_opt = relax.transform.FoldConstant()(mod_opt)
    
    # Save 
    save_model(mod_opt, filepath_opt)
    log("Done", bcolors.OKGREEN)
