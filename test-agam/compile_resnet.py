""" Compiles Resnet 18 """

import tvm
from tvm import relay, relax
from tvm.relay import testing
from tvm.relax import testing

if __name__ == '__main__':
    target = tvm.target.Target('cuda')

    # Assume inputs are RGB color images of size 224 * 224.
    batch_size = 1
    num_class = 1000
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape
    out_shape = (batch_size, num_class)

    # Load Resnet 18 as Relax IR
    mod, params = relay.testing.resnet.get_workload(
        num_layers=18, batch_size=batch_size, image_shape=image_shape
    )

    mod = relax.testing.relay_translator.from_relay(mod['main'], target)

    with open('resnet.relax', 'w') as f:
        print(mod, file=f)

    # Do some dummy optimization pass
    mod = relay.transform.FoldConstant()(mod)

    with open('resnet.optimized.relax', 'w') as f:
        print(mod, file=f)
