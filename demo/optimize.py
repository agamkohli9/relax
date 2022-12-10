import os
import tvm
from tvm import relay, relax
from tvm.script import relax as R
from utils import bcolors, log

OUTPUT_DIR = "output"
MODEL_RAW = "model-raw.relax"
MODEL_OPT_RELAY = "model-opt-relay.relax"
MODEL_OPT_RELAX = "model-opt-relax.relax"

def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def small_relay_model():
    a = relay.const(69)
    b = relay.const(69)
    c = relay.add(a, b)
    d = relay.add(a, b)
    e = relay.add(c, d)
    return e


def small_relax_model():
    a = R.const(69)
    b = R.const(69)
    c = R.add(a, b)
    d = R.add(a, b)
    e = R.add(c, d)
    return e


def save_model(model, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    log(f"Saving model to {filepath}", bcolors.OKCYAN)
    with open(filepath, "w") as f:
        print(model, file=f)


def compile():
    print("Compiling")

    # Save original model for reference
    small_model = small_relay_model()
    save_model(small_model, MODEL_RAW)

    # Compile using relay implementation
    print("optimizing with relay")
    small_model_opt = run_opt_pass(small_model, relay.transform.FoldConstant())
    save_model(small_model_opt, MODEL_OPT_RELAY)

    log("Done compiling", bcolors.OKGREEN)


if __name__ == '__main__':
    compile()
