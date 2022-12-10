""" Compiles Resnet 18 """

import tvm
from tvm import relay, relax
from tvm.relay.testing import resnet
from tvm.relax.testing import relay_translator

if __name__ == '__main__':
    target = tvm.target.Target('cuda')

    # Assume inputs are RGB color images of size 224 * 224.
    batch_size = 1
    num_class = 1000
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape
    out_shape = (batch_size, num_class)

    # Load Resnet 18 as Relax IR
    mod, params = resnet.get_workload(
        num_layers=18, batch_size=batch_size, image_shape=image_shape
    )
    mod = relay_translator.from_relay(mod['main'], target)

    # Write unoptimized Resnet18 model
    with open('resnet.relax', 'w') as f:
        print(mod.script(), file=f)
        pass

    # Do our Constant Folding optimization pass
    optimized_mod = relax.transform.FoldConstant()(mod)

    # Write optimized Resnet18 model
    with open('resnet.optimized.relax', 'w') as f:
        print(optimized_mod.script(), file=f)
        pass
