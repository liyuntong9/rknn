import torch.nn as nn
import os

class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], 10)
        )
    def forward(self, x):
        return self.mlp(x)
#
# class MLP(nn.Module):
#     def __init__(self, dims):
#         super(MLP, self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(dims[0], 10),
#         )
#     def forward(self, x):
#         return self.mlp(x)
#



def get_logger(log_filename, display=True):
    f = open(log_filename, 'w')
    counter = [0]
    def logger(text):
        if display:
            print(text)
        f.write(text + '\n')
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())
    return logger, f.close
