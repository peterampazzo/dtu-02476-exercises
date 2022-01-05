from torch import nn
import torch.nn.functional as F


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        # Dropout
        self.dropout = nn.Dropout(p=0.2)

        self.flatten = nn.Flatten()

    def forward(self, x):
        # make sure input tensor is flattened
        # x = x.view(x.shape[0], -1)
        x = self.flatten(x)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        # Output so no dropout here
        x = F.log_softmax(self.fc4(x), dim=1)

        return x