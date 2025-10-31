import torch.nn
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        y_hat = self.nn(x)
        return y_hat


class CNN(nn.Module):
    def __init__(self, input_dim, seq_len, conv_layer=[16, 32], fc_layer=[64, 32]):
        super(CNN, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=input_dim,
                            out_channels=conv_layer[0],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            padding_mode='zeros'),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.Conv1d(in_channels=conv_layer[0],
                            out_channels=conv_layer[1],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            padding_mode='zeros'),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(int((seq_len / 2 ** len(conv_layer)) * conv_layer[-1]), fc_layer[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(fc_layer[0], fc_layer[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(fc_layer[1], 1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
