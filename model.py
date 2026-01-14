import torch.nn
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, in_dim, window_size):
        super(CNN1D, self).__init__()
        # ---- C1 ----
        self.conv1 = nn.Conv1d(
            in_channels=in_dim, out_channels=128,
            kernel_size=17, stride=1, padding="same"
        )
        self.act1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        # ---- C2 ----
        self.conv2 = nn.Conv1d(
            in_channels=128, out_channels=256,
            kernel_size=7, stride=1, padding="same"
        )
        self.act2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        # ---- C3 ----
        self.conv3 = nn.Conv1d(
            in_channels=256, out_channels=512,
            kernel_size=7, stride=1, padding="same"
        )
        self.act3 = nn.LeakyReLU(0.1)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)

        with torch.no_grad():
            dummy = torch.zeros(1, in_dim, window_size)
            out = self.pool3(self.act3(self.conv3(
                self.pool2(self.act2(self.conv2(
                    self.pool1(self.act1(self.conv1(dummy)))
                )))
            )))
        final_len = out.shape[-1]

        # ---- Fully Connected ----
        self.fc1 = nn.Linear(512 * final_len, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.act_fc = nn.LeakyReLU(0.1)
        self.act_out = nn.Tanh()

    def forward(self, x):
        # C1
        x = self.pool1(self.act1(self.conv1(x)))
        # C2
        x = self.pool2(self.act2(self.conv2(x)))
        # C3
        x = self.pool3(self.act3(self.conv3(x)))

        # flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.act_fc(self.fc1(x))
        x = self.act_fc(self.fc2(x))
        x = self.act_out(self.fc3(x))

        return x
