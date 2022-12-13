import os
import numpy as np
import time

import tvm
import tvm.testing
from tvm import relay
from tvm import te
from tvm.relay import testing
from tvm.contrib import graph_executor
from tvm.contrib import utils

from .config import BATCH_SIZE, IMAGE_SHAPE, DATA_SHAPE

from logger import log, bcolors


# Runner setup
dev = tvm.cuda()


def get_lib_path(output_dir, opt_level):
    return os.path.join(output_dir, f"deploy_lib_{opt_level}.tar")


def generate_lib(output_dir, opt_level=0):
    log(f'Generating lib with opt_level {opt_level}', bcolors.OKBLUE)

    mod, params = relay.testing.resnet.get_workload(
        num_layers=18, batch_size=BATCH_SIZE, image_shape=IMAGE_SHAPE
    )
    target = tvm.target.cuda()
    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build(mod, target, params=params)

    # Generate compiled lib
    print("generate compiled lib")
    lib_path = get_lib_path(output_dir, opt_level)
    log(f"Saving lib to {lib_path}", bcolors.OKBLUE)
    lib.export_library(lib_path)
    log("Done", bcolors.OKGREEN)


def run_lib(lib_path, opt_level, num_iters):

    lib_path = get_lib_path(lib_path, opt_level)
    
    # load the module back.
    loaded_lib = tvm.runtime.load_module(lib_path)
    log("Loaded compiled lib", bcolors.OKGREEN)

    # Experimental data
    data = np.random.uniform(-1, 1, size=DATA_SHAPE).astype("float32")
    input_data = tvm.nd.array(data)

    # Run from compiled lib
    module = graph_executor.GraphModule(loaded_lib["default"](dev))

    st = time.time()
    for i in range(num_iters):
        module.run(data=input_data)
    log(f"Done running for {num_iters} iterations", bcolors.OKGREEN)
    et = time.time()

    return et - st
