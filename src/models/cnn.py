import torch as T
import torch.nn as nn
from typing import List, Tuple

class CNN(nn.Module):
    def __init__(self, device, num_features: int=11, lags: int=10, out_dim: int=5, exogenous_dim: int=0,
                 in_channels: List[int]=[1, 16, 16, 32],
                 out_channels: List[int]=[16, 16, 32, 32],
                 kernel_sizes: List[Tuple[int, int]]=[(16, 3), (3, 5), (8, 3), (4, 3)],
                 pool_kernel_sizes: List[Tuple[int, int]]=[(2, 1)]):
        super(CNN, self).__init__()
        assert len(in_channels) == len(out_channels) == len(kernel_sizes)
        self.activation = nn.ReLU()
        self.num_lags = lags
        self.num_features = num_features

        self.conv1 = nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_sizes[0],
                               padding="same")
        self.conv2 = nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_sizes[1],
                               padding="same")
        self.conv3 = nn.Conv2d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel_sizes[2],
                               padding="same")
        self.conv4 = nn.Conv2d(in_channels=in_channels[3], out_channels=out_channels[3], kernel_size=kernel_sizes[3],
                               padding="same")

        self.pool = nn.AvgPool2d(kernel_size=pool_kernel_sizes[0])

        kernel0, kernel1 = pool_kernel_sizes[-1][0], pool_kernel_sizes[-1][1]

        self.fc = nn.Linear(
            in_features=(out_channels[3] * int(lags/kernel0) * int(num_features/kernel1)) + exogenous_dim,
            out_features=out_dim
        )

        self.conv1.apply(self._init_weights)
        self.conv2.apply(self._init_weights)
        self.conv3.apply(self._init_weights)
        self.conv4.apply(self._init_weights)
        self.fc.apply(self._init_weights)

        self.device = device
        self.to(device)

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), x.size(3), x.size(1), x.size(2))
        else:
            x = x.view(x.size(0), 1, self.num_lags, self.num_features,)

        x = self.conv1(x) #[batch_size]
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()