# Highest optimization level supported by relay PassContext
MAX_OPT = 3

# Runtime configuration
BATCH_SIZE = 1
NUM_CLASS = 1000
IMAGE_SHAPE = (3, 224, 224)
DATA_SHAPE = (BATCH_SIZE,) + IMAGE_SHAPE
OUT_SHAPE = (BATCH_SIZE, NUM_CLASS)