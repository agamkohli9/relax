import matplotlib.pyplot as plt
from relay.lib import run_lib
from logger import log, bcolors
from definitions import LIB_DIR

fig, ax = plt.subplots()
bar_x = []
bar_y = []

for i in range(4):
    log(f"Running at opt_level {i}", bcolors.OKBLUE)
    elapsed = run_lib(LIB_DIR, opt_level=i, num_iters=1000)
    log(f"Elapsed time: {elapsed}", bcolors.WARNING)

    bar_x.append(f"Level {i}")
    bar_y.append(elapsed)

ax.bar(bar_x, bar_y, label=bar_x)
ax.set_ylabel('Execution time')
ax.set_xlabel('Optimization level')
ax.set_title('Relay optimization level vs. execution time')
# ax.legend(title='Fruit color')

plt.savefig('plot.png')

