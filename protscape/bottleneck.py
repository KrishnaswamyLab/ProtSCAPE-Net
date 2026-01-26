from torch import nn
from torch.nn import functional as F
import torch

class BaseBottleneck(nn.Module):
    """Basic fcn bottleneck

    Args:
        nn ([type]): [description]
    """

    def __init__(self, input_dim, hidden_dim, bottleneck_dim):
        super(BaseBottleneck, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.layer_norm2 = nn.LayerNorm(hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.layer_norm3 = nn.LayerNorm(hidden_dim//4)
        self.fc4 = nn.Linear(hidden_dim//4, bottleneck_dim)
        # self.fc5 = nn.Linear(hidden_dim//4, bottleneck_dim)
        # self.fc4 = nn.Linear(hidden_dim, bottleneck_dim)
    
    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps * std

    def forward(self, h):
        """
        b = batch size
        h = hidden dimension
        z = latent dim

        input_dim: b x s x h
        output_dim: b x z
        """

        # z_rep = F.relu(self.fc1(h))
        # z_rep = F.relu(self.fc2(z_rep))
        # z_rep = self.fc3(z_rep)
        z_rep = F.gelu(self.layer_norm1(self.fc1(h)))
        z_rep = F.gelu(self.layer_norm2(self.fc2(z_rep)))
        z_rep = F.gelu(self.layer_norm3(self.fc3(z_rep)))
        # z_rep = self.fc4(z_rep)
        z_rep = self.fc4(z_rep)
        # logvar = self.fc5(z_rep)
        # z_rep = self.reparameterize(mu, logvar)

        return z_rep
