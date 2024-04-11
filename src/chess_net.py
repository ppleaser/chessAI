import torch
import torch.nn as nn


class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 8 * 8, 256)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(256, 64)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(64, 1)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x.view(-1, 12, 8, 8))))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 256 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x