import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class CropwiseLinearProjection(nn.Module):
    def __init__(self, embed_dim, num_crops = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.projection = nn.Parameter(torch.randn(num_crops, embed_dim, embed_dim))

    def forward(self, x):
        x = torch.matmul(x, self.projection.transpose(1,2))
        return x

class CropwiseLinearMultiheadProjection(nn.Module):
    def __init__(self, heads, embed_dim, head_dim, crops = 1):
        super().__init__()
        self.heads = heads
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.crops = crops
        self.projection = nn.Parameter(torch.randn(heads, crops, head_dim, embed_dim))
    
    def forward (self, x):
        return torch.matmul(x, self.projection.transpose(2,3))
    
class MultiheadAttention(nn.Module):
    def __init__(self, heads, num_crops, embed_dim, cropwise = True):
        super().__init__()
        self.heads = heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // heads
        self.num_crops = num_crops
        self.cropwise = cropwise
        if cropwise:
            self.q_linear = CropwiseLinearMultiheadProjection(heads, embed_dim, self.head_dim, num_crops)
            self.k_linear = CropwiseLinearMultiheadProjection(heads, embed_dim, self.head_dim, num_crops)
            self.v_linear = CropwiseLinearMultiheadProjection(heads, embed_dim, self.head_dim, num_crops)
            self.linear_projection = CropwiseLinearProjection(embed_dim=embed_dim, num_crops=num_crops)
        else:
            pass
            self.q_linear = CropwiseLinearMultiheadProjection(heads, embed_dim, self.head_dim)
            self.k_linear = CropwiseLinearMultiheadProjection(heads, embed_dim, self.head_dim)
            self.v_linear = CropwiseLinearMultiheadProjection(heads, embed_dim, self.head_dim)
            self.linear_projection = CropwiseLinearProjection(embed_dim=embed_dim)
    
    def forward(self, x, mask = None):
        Q, K, V = self.projection_into_heads(x)     
        multihead_output = self.scaled_dot_product_attention(Q, K, V, mask)
        catenated_output = multihead_output.permute(0,2,3,1,4).flatten(-2)
        return self.linear_projection(catenated_output)
        
    def projection_into_heads(self, x):
        # input shape: (batch, crops, num_patches, embed_dim)
        x = torch.unsqueeze(x, dim = 1)
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        return Q, K, V
    
    def scaled_dot_product_attention(self, Q, K, V, mask = None):
        b, h, c, p, head_dim = Q.shape
        if self.cropwise:
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
            attn_probs = torch.softmax(attn_scores, dim=-1)
            return torch.matmul(attn_probs, V) 
        else:
            # Extreme bottleneck when global attention
            # attn_scores = torch.matmul(Q.view(b, h, -1, head_dim), K.view(b, h, -1, head_dim).transpose(-2, -1)) / math.sqrt(head_dim)
            # Patial solution - attention over crops dim instead of patch dim
            attn_scores = torch.matmul(Q.transpose(-2,-3), K.permute(0,1,3,4,2)) / math.sqrt(head_dim)
            attn_probs = torch.softmax(attn_scores, dim=-1)
            return torch.matmul(attn_probs, V.transpose(-2,-3)).transpose(-3,-2)

class CropwiseFFN(nn.Module):
    def __init__(self, num_crops, embed_dim, hidden_dim, cropwise = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.cropwise = cropwise
        if cropwise:
            self.linear1 = nn.Parameter(torch.randn(num_crops, hidden_dim, embed_dim))
            # dim 1 of bias is for sequence
            self.bias1 = nn.Parameter(torch.randn(num_crops, 1, hidden_dim))
            self.linear2 = nn.Parameter(torch.randn(num_crops, embed_dim, hidden_dim))
            self.bias2 = nn.Parameter(torch.randn(num_crops, 1, embed_dim))
        else:
            self.linear1 = nn.Parameter(torch.randn(1, hidden_dim, embed_dim))
            # dim 1 of bias is for sequence
            self.bias1 = nn.Parameter(torch.randn(1, hidden_dim))
            self.linear2 = nn.Parameter(torch.randn(1, embed_dim, hidden_dim))
            self.bias2 = nn.Parameter(torch.randn(1, embed_dim))
            

    def forward(self, x):
        x = F.relu(torch.matmul(x, self.linear1.transpose(-1,-2)) + self.bias1)
        x = torch.matmul(x, self.linear2.transpose(-1,-2)) + self.bias2
        return x

class CropwiseLayernorm(nn.Module):
    def __init__(self, num_crops, num_patches, embed_dim, cropwise = True):
        super().__init__()
        # Include normalization over crops?
        self.layer_norm = nn.LayerNorm([num_patches, embed_dim])

    def forward(self, x):
        return self.layer_norm(x)    

class GroupTransformerBlock(nn.Module):
    def __init__(self, num_heads, num_crops, num_patches, embed_dim, cropwise = True):
        super().__init__()
        self.multihead_attention = MultiheadAttention(num_heads, num_crops, embed_dim, cropwise=cropwise)
        self.layernorm = CropwiseLayernorm(num_crops, num_patches, embed_dim)
        self.ffn = CropwiseFFN(num_crops, embed_dim, 4 * embed_dim, cropwise=cropwise)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        ln_x = self.layernorm(x)
        mh_x = self.multihead_attention(ln_x)
        x = x + mh_x
        ln_x = self.layernorm(x)
        ff_x = self.ffn(ln_x)
        output = self.dropout(x + ff_x )
        return output       

class PatchEmbedding(nn.Module):
    def __init__(self, crop_size, num_crops, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = crop_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_crops = num_crops
        self.num_patches = (crop_size//patch_size)**2
        self.projection = nn.Conv2d(in_channels * num_crops,
                                    embed_dim * num_crops,
                                    kernel_size=patch_size,
                                    stride=patch_size,
                                    groups=num_crops,
                                    bias = False)
    
    def forward(self, x):
        flattened_multicrop = torch.flatten(self.projection(x), start_dim=-2)
        output = flattened_multicrop.view(flattened_multicrop.shape[0], self.num_crops, self.embed_dim, self.num_patches).transpose(3,2)
        # grouped_multicrop = flattened_multicrop.view(flattened_multicrop.shape[0], self.num_crops, self.embed_dim, self.num_patches)
        # output = grouped_multicrop.transpose(-1,-2)
        return output

class FinalProjection(nn.Module):
    def __init__(self, crops, embed_dim, num_patches):
        super().__init__()
        # TODO: Cropwise?
        self.linear = nn.Parameter(torch.randn([crops, 2, embed_dim * num_patches]))
        
    def forward(self, x):
        # batch, crops, num_patches, embed_dim
        flatten_patches = torch.flatten(x, start_dim = -2).unsqueeze(-2)
        output = torch.matmul(flatten_patches, self.linear.transpose(-1,-2)).flatten(-2)
        return output
    
    
class GroupedTransformer(nn.Module):
    def __init__(self, crop_size, num_crops, patch_size, embed_dim, num_heads, num_blocks):
        super().__init__()
        self.num_patches = (crop_size // patch_size)**2
        self.patch_embedding = PatchEmbedding(crop_size=crop_size,
                                              num_crops=num_crops,
                                              patch_size=patch_size,
                                              in_channels=3,
                                              embed_dim=embed_dim)
        self.patch_pos_embedding = nn.Parameter(torch.randn(1, 1, self.num_patches, embed_dim))
        self.crop_pos_embedding = nn.Parameter(torch.randn(1, num_crops, 1, embed_dim))
        
        self.transformer_blocks = nn.ModuleList([
            GroupTransformerBlock(num_heads, num_crops, self.num_patches, embed_dim, cropwise=True)
            for _ in range(num_blocks)
        ])
        
        self.final_projection = FinalProjection(num_crops, embed_dim, self.num_patches)

    def forward(self, x):
        patches = self.patch_embedding(x)
        x = patches + self.patch_pos_embedding + self.crop_pos_embedding
        for block in self.transformer_blocks:
            x = block(x)
        output = self.final_projection(x)
        output = torch.flatten(output.transpose(-1,-2), start_dim=-2)
        return output / 50e7 # due to enormous loss error