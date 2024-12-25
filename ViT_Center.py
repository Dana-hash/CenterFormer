import random
import numpy as np
import torch
import cv2
import math
import torch.nn.functional as F
from torch import nn
from skimage import transform
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers
from torch.nn import init


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class OurFE(nn.Module):
    def __init__(self, channel, dim):
        super(OurFE, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(3 * channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(in_channels=channel, out_channels=dim, kernel_size=1)

    def forward(self, x):
        feature1 = self.conv_block1(x)
        feature2 = self.conv_block2(feature1)
        feature3 = self.conv_block3(feature2)
        fused_features = self.fusion_conv(torch.cat((feature1, feature2, feature3), dim=1))
        pooled_features = self.avg_pool(fused_features)
        output = self.final_conv(pooled_features)
        return output


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class DEPTHWISECONV(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, padding=0, stride=1, is_fe=False):
        super(DEPTHWISECONV, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            DEPTHWISECONV(dim, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=dim, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x):
        spatial_dim = int(math.sqrt(x.size(1)))  # 计算空间维度
        x = x + rearrange(
            self.net(
                rearrange(x, 'b (h w) c -> b c h w', h=spatial_dim, w=spatial_dim)),
            'b c h w -> b (h w) c'
        )
        return x


class CenterAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0., num_patches=10):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.to_k_v = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if not (heads == 1 and dim_head == dim) else nn.Identity()
        self.to_qkv_spec = nn.Linear(num_patches, num_patches * 3, bias=False)
        self.attend_spec = nn.Softmax(dim=-1)

    def forward(self, x):
        # Spatial attention
        median_idx = x.size(1) // 2  # Optimized median index computation
        x_center = x[:, median_idx:median_idx + 1, :].expand(-1, x.size(1), -1)
        q = self.to_q(x_center)
        k, v = self.to_k_v(x).chunk(2, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        attn_weights = self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        out = torch.matmul(attn_weights, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        spatial_out = self.to_out(out)
        # Spectral attention
        x_t = x.transpose(-2, -1)
        q_spec, k_spec, v_spec = self.to_qkv_spec(x_t).chunk(3, dim=-1)
        q_spec, k_spec, v_spec = map(
            lambda t: rearrange(t, 'b n d -> b 1 n d'),
            (q_spec, k_spec, v_spec)
        )
        spec_weights = self.attend_spec(torch.matmul(q_spec, k_spec.transpose(-1, -2)) * self.scale).squeeze(dim=1)
        # Combine spatial and spectral
        return torch.matmul(spatial_out, spec_weights)



class CenterTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0., num_patches=25, is_granularity=False):
        super().__init__()
        self.is_granularity = is_granularity
        self.layers = nn.ModuleList([])
        if is_granularity:
            self.conv1d = nn.Conv1d(in_channels=num_patches, out_channels=1, kernel_size=1)
            self.max_pool = nn.MaxPool1d(kernel_size=num_patches)
            self.avg_pool = nn.AvgPool1d(kernel_size=num_patches)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim,
                        CenterAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, num_patches=num_patches)),
                PreNorm(dim, FeedForward(dim)),
            ]))

    def forward(self, x):
        outputs = []
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            outputs.append(x)
        if self.is_granularity:
            # Granularity-specific behavior
            output_result = rearrange(outputs[-1], 'b n d -> b d n')
            max_token = self.max_pool(output_result)
            avg_token = self.avg_pool(output_result)
            cls_token = max_token + avg_token
            cls_token = rearrange(cls_token, 'b d n -> b n d')
            return cls_token
        else:
            # Center-specific behavior
            return x, outputs


class SubNet(nn.Module):
    def __init__(self, num_patches, dim, emb_dropout, depth, heads, dim_head, dropout, is_granularity=False):
        super(SubNet, self).__init__()
        self.is_granularity = is_granularity

        if is_granularity:
            # Granularity-specific embedding layer
            self.embedding = nn.Sequential(
                Rearrange('b c w h -> b (h w) c'),
            )

        # Shared positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        #  transformer
        self.transformer = CenterTransformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head, dropout=dropout,
            num_patches=num_patches, is_granularity=is_granularity
        )

    def forward(self, x):
        if self.is_granularity:
            # Granularity-specific embedding
            x = self.embedding(x)

        # Add positional embedding
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]

        # Apply dropout and transformer
        x = self.dropout(x)

        if not self.is_granularity:
            _, outputs = self.transformer(x)
            return outputs[-1]
        else:
            x = self.transformer(x)
            return x



def get_num_patches(ps, ks):
    return int((ps - ks) / ks) + 1


class ViT_Center(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, depth_granularity, heads, heads_granularity,
                 mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super(ViT_Center, self).__init__()
        self.ournet = OurFE(channels, dim)
        self.image_size = image_size
        self.patch_size = patch_size
        self.weight = torch.ones(len(patch_size))  # Initialize weights for patch sizes
        self.net = nn.ModuleList()
        self.granularity_net = nn.ModuleList()
        self.mlp_head = nn.ModuleList()

        for ps in patch_size:
            num_patches = get_num_patches(image_size, ps) ** 2
            patch_dim = dim * num_patches
            # Normal and granularity mode subnets
            self.net.append(
                SubNet(num_patches, dim, emb_dropout, depth, heads, dim_head, dropout, is_granularity=False))
            self.granularity_net.append(
                SubNet(ps**2, dim, emb_dropout, depth_granularity, heads_granularity, dim_head, dropout,
                       is_granularity=True))
            self.mlp_head.append(nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim, num_classes)))

    def forward(self, img):
        img = self.ournet(img)  # Feature extraction from image
        all_granularity_results = []

        # Generate granularity data
        for ps in self.patch_size:
            stride = ps if self.image_size % ps == 0 else ps // 2
            granularity_size = (self.image_size - ps) // stride + 1
            unfolded = img.unfold(2, ps, stride).unfold(3, ps, stride)

            granularity_data = [unfolded[:, :, i, j] for i in range(granularity_size) for j in range(granularity_size)]
            all_granularity_results.append(granularity_data)

        all_branch_outputs = []

        # Process each subnet and granularity data
        for i, sub_net in enumerate(self.net):
            granularity_results = [
                self.process_granularity_data(granularity_img, self.granularity_net[i]) for granularity_img in all_granularity_results[i]
            ]
            # Concatenate granularity results and apply transformer
            granularity_output = torch.cat(granularity_results, dim=-2)
            transformer_output = sub_net(granularity_output)
            all_branch_outputs.append(transformer_output)

        # Softmax for patch size weights
        self.weight = F.softmax(self.weight, dim=0)
        final_output = 0

        # Classification using weighted results
        for i, mlp in enumerate(self.mlp_head):
            output = all_branch_outputs[i].flatten(start_dim=1)
            final_output += mlp(output) * self.weight[i]

        return final_output

    def process_granularity_data(self, granularity_img, granularity_sub_net):
        """
        Process granularity data through the granularity subnet.
        Since SubNet has its own forward method, we don't need to manually process it here.
        """
        return granularity_sub_net(granularity_img)  # Directly call the forward method of SubNet
