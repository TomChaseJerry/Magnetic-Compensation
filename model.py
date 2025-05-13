import torch.nn
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomLoss(nn.Module):
    def __init__(self, beta, std_y, lambda_phy=1):
        super(CustomLoss, self).__init__()
        self.huber_loss = nn.HuberLoss()
        self.mse_loss = nn.MSELoss()
        self.lambda_phy = lambda_phy
        self.beta = beta
        self.y_mean = torch.tensor(std_y.mean_, dtype=torch.float32, device=device)
        self.y_scale = torch.tensor(std_y.scale_, dtype=torch.float32, device=device)

    def forward(self, c, y, A, mag4uc, mag4c):
        # y_hat = mag4uc - A * (beta + c)
        # trueth: y
        # TL_mag: mag4c
        y_hat = (mag4uc - torch.diag(torch.matmul(A, (self.beta + c).T)).reshape(-1, 1) - self.y_mean) / self.y_scale

        L_data = self.mse_loss(y_hat, y)
        L_TLprior = self.mse_loss(y_hat, mag4c)
        L_total = L_data + self.lambda_phy * L_TLprior
        return L_total


class TLMLP(nn.Module):
    def __init__(self, input_dim):
        super(TLMLP, self).__init__()
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


class TLCNN(nn.Module):
    def __init__(self, input_dim, seq_len, conv_layer=[16, 32], fc_layer=[64, 32]):
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return x


class MagTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=4,
                 dim_feedforward=256, dropout=0.1, max_seq_len=20):
        super(MagTransformer, self).__init__()

        self.d_model = d_model
        self.input_embed = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 回归输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor shape [batch_size, seq_len, input_dim]
        Returns:
            output: Tensor shape [batch_size]
        """
        x = x.permute(0, 2, 1)

        # 输入嵌入
        x = self.input_embed(x) * math.sqrt(self.d_model)  # [B, T, D]

        # 位置编码
        x = self.pos_encoder(x)

        # Transformer编码
        x = self.transformer_encoder(x)  # [B, T, D]

        # 时序特征聚合（取最后一个时间步）
        x = x[:, -1, :]  # [B, D]

        # 回归输出
        output = self.output_layer(x).squeeze(-1)  # [B]
        return output


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
