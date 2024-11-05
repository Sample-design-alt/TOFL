import torch
from mamba_ssm import Mamba
from torch import nn
from model.layers.Embed import TokenEmbedding
import torch.nn.functional as F

class mamba(nn.Module):
    def __init__(self, config):
        super(mamba, self).__init__()
        self.value_embedding = TokenEmbedding(
            c_in=config['model_params']['in_channel'],
            d_model=config['model_params']['d_model']
        )
        self.mamba = nn.ModuleList(
            [
                Mamba(
                    d_model=config['model_params']['d_model'],  # Model dimension d_model
                    d_state=config['model_params']['d_state'],  # SSM state expansion factor
                    d_conv=config['model_params']['d_conv'],  # Local convolution width
                    expand=config['model_params']['expand'],
                    ) for i in range(config['model_params']['num_layers'])
            ]
        )

        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.flatten = nn.Flatten()

    def forward(self, t):
        x = self.value_embedding(t)
        for mamba in self.mamba:
            x = mamba(x)

        output = self.flatten(self.adaptive_avg_pool(x.transpose(2, 1).contiguous()))
        output = F.normalize(output, dim=1)

        return output



# batch, length, dim = 2, 64, 16
# x = torch.randn(batch, length, dim).to("cuda")
#
# model = .to("cuda")
# y = model(x)
# assert y.shape == x.shape