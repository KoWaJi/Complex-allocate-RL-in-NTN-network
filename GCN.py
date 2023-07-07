import torch
import torch.nn as nn




class GCN(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels):
        super(GCN, self).__init__()
        self.lin1 = nn.Linear(in_channels, hid_channels, bias=False)
        self.lin2 = nn.Linear(hid_channels, out_channels, bias=False)
        self.out_channels = out_channels

    def forward(self, input):
        # x has shape [N, in_channels]
        # edge_index = L = D - A

        x, edge_index = input
        true_ue = x.size(0) - 1
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        x = torch.relu(x)
        output = torch.matmul(edge_index, x)

        bs_embedding = torch.mul(1 / true_ue, output[0, :])
        user_embedding = output[1:, ]
        bs_embedding = bs_embedding.expand(true_ue, self.out_channels)
        embedding = torch.cat([user_embedding, bs_embedding], dim=1)
        return embedding
