from definitions import RELAX_IR_DIR
from relax.optimize import get_modules, optimize_and_save_model

modules = get_modules()

for name, mod in modules:
    print("compiling model ", name)
    optimize_and_save_model(RELAX_IR_DIR, name, mod)
