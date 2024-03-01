import torch.nn as nn
import torch.nn.functional as F

class CNNAugmentedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(CNNAugmentedLSTM, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.bn = nn.BatchNorm1d(num_features=8)
        self.fc = nn.Linear(hidden_size, 512)
        self.output = nn.Linear(512, 1)

    def forward(self, x):
        # 1D Conv expects: [batch, channels, seq_len]
        x = self.conv1d(x)
        #batch nomorlization
        x = self.bn(x)
        # LSTM expects: [batch, seq_len, features]
        x = x.permute(0, 2, 1)
        x, (ht, ct) = self.lstm(x)
        
        # Use the last hidden state
        x = ht[-1]
        
        # Pass through the fully connected layers
        x = F.leaky_relu(self.fc(x))
        x = self.output(x)
        return x


class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
