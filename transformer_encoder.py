import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL import Image
import numpy as np

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=768, mlp_dim=3072, num_heads=3):
        super().__init__()

        self.embed_dim=embed_dim
        self.mlp_dim=mlp_dim
        self.num_heads=num_heads
        self.head_dim=(embed_dim//num_heads)

        #for attention, QKV
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)

        #adding layer norms
        self.norm1=nn.LayerNorm(embed_dim)

        self.norm2=nn.LayerNorm(embed_dim)

        #MLP def
        self.mlp=nn.Sequential(nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, embed_dim))
    
    def forward(self, x):
        batch_size, num_tokens, token_dim = x.shape
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # multi-head attention
        Q = Q.reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim
        )

        K = K.reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim
        )

        V = V.reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim
        )

        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dim)
        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        attention_output = (
            attention_weights @ V
        )

        attention_output = attention_output.permute(
            0,
            2,
            1,
            3
        )

        # attention_output 
        attention_output = attention_output.reshape(
            batch_size,
            num_tokens,
            self.embed_dim
        )

        # print(attention_output.shape)

        x = self.norm1(
            x + attention_output
        )

        mlp_output = self.mlp(x)

        x = self.norm2(
            x + mlp_output
        )

        return x


encoder_block = TransformerEncoderBlock()
#print(encoder_block)


img = Image.open("lion.jpg")
img = img.resize((224, 224))

# --------------------
# Extract Patch P27
# --------------------
patch_size = 28

all_patches = []

# -------------------
# Extract all 64 patches
# -------------------
for row in range(8):

    for col in range(8):

        x = col * patch_size
        y = row * patch_size

        patch = img.crop(
            (x,
             y,
             x + patch_size,
             y + patch_size)
        )

        patch_array = np.array(patch)

        flat = patch_array.flatten()

        all_patches.append(flat)

# --------------------
# Convert to tensor
# --------------------
all_patches = torch.tensor(
    np.array(all_patches),
    dtype=torch.float32
)

# --------------------
# Patch Embedding Layer
# --------------------
patch_embed = nn.Linear(
    2352,
    768
)

patch_tokens = patch_embed(all_patches)

# --------------------
# Positional Embeddings
# --------------------
num_patches = 64
embed_dim = 768

pos_embed = nn.Parameter(
    torch.randn(
        num_patches,
        embed_dim
    )
)


final_tokens = (
    patch_tokens +
    pos_embed
)

tokens = final_tokens.unsqueeze(0)

# print(tokens.shape)

output = encoder_block(tokens)

# print(output.shape)
