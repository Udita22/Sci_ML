from transformer_encoder import TransformerEncoderBlock
import torch
import torch.nn as nn


# for CIFAR10, image size is 32x32 and patch size is 4x4
class VisionTransformer(nn.Module):

    def __init__(self, image_size=32, patch_size=4, embed_dim=128, num_heads=4, mlp_dim=512, num_classes=10
    ):

        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (
            image_size // patch_size
        ) ** 2

        self.patch_dim = (
            patch_size
            * patch_size
            * 3
        )

        self.patch_embed = nn.Linear(
            self.patch_dim,
            embed_dim
        )

        self.pos_embed = nn.Parameter(torch.randn(self.num_patches,embed_dim))

        self.encoder = TransformerEncoderBlock(embed_dim=embed_dim,num_heads=num_heads,mlp_dim=mlp_dim)

        self.classifier = nn.Linear(embed_dim,num_classes)

    def forward(self, x):

        batch_size = x.shape[0]
        #unfold()

        patches = x.unfold(2,self.patch_size,self.patch_size).unfold(3,self.patch_size,self.patch_size)

        patches = patches.permute(0,2,3,1,4,5)

        patches = patches.reshape(batch_size,self.num_patches,self.patch_dim)

        #patch embedding
        tokens = self.patch_embed(patches)

        tokens = (tokens + self.pos_embed)

        tokens = self.encoder(tokens)

        # print(tokens.shape)

        tokens = tokens.mean( dim=1)

        logits = self.classifier(tokens)

        return logits
'''
vit = VisionTransformer()
dummy = torch.randn(
    8,
    3,
    32,
    32
)

output = vit(dummy)

print("output shape is")
print(output.shape)
'''




