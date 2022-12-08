""" Compiles the model given in model.py """ 

import tvm
from tvm import relay, relax
from tvm.relax import testing
import torch
from model import Model

if __name__ == '__main__':
    # Config device
    target = tvm.target.Target('cuda')
    device = tvm.device('cuda')

    # Config input model
    input_shape = (2) # get from Netron
    input_data = torch.randn(input_shape)

    # Load model as Relax IR
    model = Model()
    model.load_state_dict(torch.load('model.pt'))
    model = torch.jit.trace(model, input_data).eval()
    mod, _ = relay.frontend.from_pytorch(model, [('input', input_data.shape)])
    mod = relax.testing.relay_translator.from_relay(mod['main'], target)

    with open('model.relax', 'w') as f:
        print(mod, file=f)

    # Do some dummy optimization pass
    mod = relay.transform.FoldConstant()(mod)
    #mod = relax.transform.FlashAttention()(mod)

    with open('model.optimized.relax', 'w') as f:
        print(mod, file=f)
