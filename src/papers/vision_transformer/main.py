import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # Formula is: (Height * Width) / (Patch_size * Patch_size). Simplifies to: (img_size / patch_size)^2 when H=W=img_size
        self.n_patches = (img_size // patch_size) ** 2
        # A convolution with kernel size = patch size and stride = patch size acts as the patch projector.
        # This splits the image into patches and projects them to the embedding dimension.
        # It's mathematically equivalent to splitting the image into patches, flattening each patch, 
        # and then projecting the flattened patches to the embedding dimension.
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (Batch, Channel, Height, Width)
        x = self.proj(x)  # (Batch, Embed_dim, Height/patch_size, Width/patch_size)
        # Flatten the last two dimensions of the output tensor
        x = x.flatten(2)  # (Batch, Embed_dim, n_patches)
        # Transpose the output tensor to have the shape (Batch, n_patches, Embed_dim)
        x = x.transpose(1, 2)  # (Batch, n_patches, Embed_dim)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        # MLP: two linear layers with GELU nonlinearity in between.
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Apply LayerNorm before attention (Pre-Norm)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # Apply LayerNorm before MLP and add residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
        num_classes=1000,
    ):
        super().__init__()
        # Create patch embeddings
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Learnable positional embeddings for patches + class token
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Stack Transformer blocks
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        # Final classification head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (Batch, Channel, Height, Width)
        x = self.patch_embed(x)  # (Batch, n_patches, Embed_dim)
        B = x.shape[0]
        # Prepend the class token to the patch tokens for each batch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (Batch, 1, Embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (Batch, n_patches+1, Embed_dim)
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        # Classification: take the class token (first token)
        cls_out = x[:, 0]
        x = self.head(cls_out)
        return x


# Example usage:
if __name__ == "__main__":
    model = VisionTransformer()
    dummy_input = torch.randn(1, 3, 224, 224)  # (B, C, H, W)
    logits = model(dummy_input)
    print("Logits shape:", logits.shape)  # Expected: (1, 1000)
