import torch.nn as nn


class MagneticCompensationLoss(nn.Module):
    def __init__(self, lambda_phy=1.0):
        super(MagneticCompensationLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_phy = lambda_phy

    def forward(self, B_pred, B_real, B_total):
        L_data = self.mse_loss(B_pred, B_real)
        L_physics = self.mse_loss(B_pred, B_total)
        L_total = L_data + self.lambda_phy * L_physics
        return L_total


class PINN_TLNET(nn.Module):
    def __init__(self, input_dim):
        super(PINN_TLNET, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
