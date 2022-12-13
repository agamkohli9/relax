from definitions import RELAX_IR_DIR
from relax.optimize import get_modules, optimize_and_save_model

modules = get_modules()
print("modules:", modules)

for mod in modules:
    print("compiling model ", mod[0])
    optimize_and_save_model(RELAX_IR_DIR, mod[0], mod[1])