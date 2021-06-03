import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

lstm   = nn.LSTM(3, 3)

inputs        = [torch.randn(1, 3) for _ in range(5)]
inputs_concat = torch.cat(inputs).view(len(inputs), 1, -1)

h = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))

hidden = h

for i in inputs:
    out2, hidden = lstm(i.view(1, 1, -1), hidden)



out, hidden2 = lstm(inputs_concat, h)


print('all in one go size ={}'.format(out.size()))
print('iterations out size = {}'.format(out2.size()))

print(out2)

print(out[-1,:,:])
