from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------
# Load image
# --------------------
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

print("All Patches Shape:")
print(all_patches.shape)

# --------------------
# Patch Embedding Layer
# --------------------
patch_embed = nn.Linear(
    2352,
    768
)

patch_tokens = patch_embed(all_patches)

print("\nPatch Tokens Shape:")
print(patch_tokens.shape)

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
print("\nPosition embeddings shape:")
print(pos_embed.shape)


# # Get positional embedding for P27, it's a random values of the position and not the correct value as training is needed for getting the correct value.
# position_27 = pos_embed[:, 27, :]

# print("\nPosition 27 Shape:")
# print(position_27.shape)

# --------------------
# Final Token
# --------------------
# final_tokens = (
#     patch_token +
#     position_27
# )

final_tokens = (
    patch_tokens +
    pos_embed
)

print("\nFinal Token Shape:")
print(final_tokens.shape)

print("\nFirst 10 Values:")
print(final_tokens[0][:10])

#calculating the attention part
import torch
import math

# X = your final tokens
# Shape: (64,768)

X = final_tokens

# Learnable weight matrices
Wq = torch.randn(768, 768, requires_grad=True)
Wk = torch.randn(768, 768, requires_grad=True)
Wv = torch.randn(768, 768, requires_grad=True)

# Create Q, K, V
Q = X @ Wq
K = X @ Wk
V = X @ Wv

print("Q Shape:", Q.shape)
print("K Shape:", K.shape)
print("V Shape:", V.shape)

scores = Q @ K.T

print(scores.shape)

scores = scores / math.sqrt(768)

print("After changing score")
print(scores.shape)

#softmax makes the values between 0 and 1 and also the sum is 1.
attention_weights = F.softmax(scores, dim=1)

#attention_output = scores @ V
attention_output = attention_weights @ V

print("Attention output dimension")
print(attention_output.shape)