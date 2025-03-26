import torch.nn
import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self, beta_TL, lambda_phy=1.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_phy = lambda_phy
        self.beta_TL = beta_TL

    def forward(self, model_output, y, A, B_tl):
        c, d = model_output[:, :18], model_output[:, 18].reshape(-1, 1)

        # A * (beta + c) + d
        tmp_value = torch.diag(torch.matmul(A, (self.beta_TL + c).T)).reshape(-1, 1) + d

        L_data = self.mse_loss(tmp_value, y)
        L_TLprior = self.mse_loss(tmp_value, B_tl)
        L_total = L_data + self.lambda_phy * L_TLprior
        return L_total


class TLMLP(nn.Module):
    def __init__(self, input_dim):
        super(TLMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 19)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class TLCNN(nn.Module):
    def __init__(self, input_dim, seq_len, conv_layer=[16, 32], fc_layer=[32, 18]):
        super(TLCNN, self).__init__()
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


if __name__ == '__main__':
    n = 100
    features = 25
    sequence_length = 20
    x = torch.randn(n, features)
    y = torch.randn(n)

    stride = 1
    x_windows = x.unfold(0, sequence_length, stride)
    y = y[sequence_length - 1:]

    print(x_windows.shape)
    print(y.shape)

    model = TLCNN(features, sequence_length)
    output = model(x_windows)
    print(output.shape)
