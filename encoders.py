import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormMLP(nn.Module):
    """A simple MLP block with LayerNorm and ReLU.
    This will serve as a building block for the ResNet v2 style block.
    Each block: pre-activation (LayerNorm -> ReLU) -> Linear -> Possibly next block
    """
    def __init__(self, input_dim, output_dim):
        super(LayerNormMLP, self).__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Pre-activation: LayerNorm -> ReLU
        out = self.ln(x)
        out = F.relu(out)
        out = self.fc(out)
        return out

class PreActivationResBlock(nn.Module):
    """ResNet v2 style pre-activation residual block.
    Each block:
      - layer norm + relu (pre-activation)
      - linear layer
      - layer norm + relu
      - linear layer
    Add skip connection.
    """
    def __init__(self, dim):
        super(PreActivationResBlock, self).__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        # Pre-activation for first layer
        out = self.ln1(x)
        out = F.relu(out)
        out = self.fc1(out)

        # Pre-activation for second layer
        out = self.ln2(out)
        out = F.relu(out)
        out = self.fc2(out)

        return x + out

class ResNetV2Tower(nn.Module):
    """A 10-block ResNet v2 style tower as described.
    Input -> 10 Residual Blocks -> Output
    Each linear layer output size is 256, so we assume input also 256 for simplicity.
    """
    def __init__(self, input_dim=256, num_blocks=10):
        super(ResNetV2Tower, self).__init__()
        self.input_dim = input_dim
        self.blocks = nn.ModuleList([PreActivationResBlock(input_dim) for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    

###########################
# VQ-VAE style codebook    #
###########################
# According to snippet, a codebook of size 32 is used. Each entry is a one-hot vector of size 32.
# We will store them as a fixed identity matrix.
#
# We'll simulate the VQ step by:
#  - Given some encoder output (logits), we select the code via argmax or use Gumbel Softmax trick for training.
#  - The selected code is a one-hot vector from the codebook.

class VQCodebook(nn.Module):
    """A fixed codebook of size M. Each code is a one-hot vector of dimension M.
    For M=32, codebook = I_32 (the 32x32 identity matrix).
    """
    def __init__(self, M=32):
        super(VQCodebook, self).__init__()
        # Just a fixed identity matrix of size (M,M). Each row is a code.
        self.register_buffer('codebook', torch.eye(M))

    def forward(self, logits: torch.Tensor, use_gumbel=False, tau=1.0):
        """
        logits: (batch_size, M)
        If use_gumbel: apply Gumbel-Softmax trick (Jang et al., 2016) to sample a discrete code.
        Else: argmax selection.
        """
        if use_gumbel:
            # Gumbel softmax trick
            gumbel_probs = F.gumbel_softmax(logits, tau=tau, hard=True)
            return gumbel_probs @ self.codebook  # This should yield a one-hot vector directly
        else:
            # Argmax selection
            indices = torch.argmax(logits, dim=-1)
            # Convert indices to one-hot
            one_hot = F.one_hot(indices, num_classes=self.codebook.size(0)).float()
            return one_hot