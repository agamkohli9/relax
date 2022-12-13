import os
from pathlib import Path
from relay.lib import generate_lib
from relay.config import MAX_OPT
from definitions import LIB_DIR

# This script generates compiled libraries at all optimization levels
# available for the relay PassContext. Output is saved to ./lib.

# Kill old lib
for f in os.listdir(LIB_DIR):
    os.remove(os.path.join(LIB_DIR, f))

# Generate new
for opt_level in range(MAX_OPT + 1):
    generate_lib(LIB_DIR, opt_level=opt_level)
