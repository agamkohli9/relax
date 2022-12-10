import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 4)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        return x

model = Model()

# PyTorch
torch.save(model.state_dict(), 'model.pt')
