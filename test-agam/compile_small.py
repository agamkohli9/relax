""" Compiles the model given in model.py """ 

import tvm
from tvm import relax, relay
from small import SmallModel

OUTPUT_RAW = "small"
OUTPUT_OPT = "small.optimized"

def save_model(model, name: str):
    filename = f"{name}.relax"
    print(f"saving {filename}")
    with open(filename, "w") as f:
        print(model, file=f)


def compile():
    # Save original model
    save_model(SmallModel, OUTPUT_RAW)

    # Common subexpression elimination
    optimized_mod = relax.transform.CommonSubexpressionElimination()(SmallModel)

    # Save optimized model
    save_model(optimized_mod, OUTPUT_OPT)


if __name__ == '__main__':
    compile()
