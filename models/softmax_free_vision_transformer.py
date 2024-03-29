import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
from functools import partial
import numpy as np
from einops import rearrange

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from .softmax_free_transformer import SoftmaxFreeTransformerBlock, SoftmaxFreeNormTransformerBlock


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, kernel_size=7, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        if kernel_size == 7:
            self.proj = nn.Sequential(nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(32), nn.ReLU(),
                                      nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(32), nn.ReLU(),
                                      nn.Conv2d(32, embed_dim, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(embed_dim), nn.ReLU())
        else:
            self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(embed_dim), nn.ReLU())

    def forward(self, x):
        x = self.proj(x)
        H, W = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        return x, (H, W)


class SoftmaxFreeVisionTransformer(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], 
                 mlp_ratios=[4, 4, 4, 4], 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0.,
                 attn_drop_rate=0., 
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], 
                 sr_ratios=[8, 4, 2, 1], 
                 newton_max_iter=20, 
                 kernel_method="cuda"):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = PatchEmbed(img_size=img_size, kernel_size=7, patch_size=patch_size, in_chans=in_chans,
                                       embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(img_size=img_size // 4, kernel_size=3, patch_size=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(img_size=img_size // 8, kernel_size=3, patch_size=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 16, kernel_size=3, patch_size=2, in_chans=embed_dims[2],
                                       embed_dim=embed_dims[3])

        # pos_embed
        self.pos_embed1 = nn.Parameter(torch.zeros(1, 3136, embed_dims[0]))
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, 784, embed_dims[1]))
        self.pos_drop2 = nn.Dropout(p=drop_rate)
        self.pos_embed3 = nn.Parameter(torch.zeros(1, 196, embed_dims[2]))
        self.pos_drop3 = nn.Dropout(p=drop_rate)
        self.pos_embed4 = nn.Parameter(torch.zeros(1, 49 + 1, embed_dims[3]))
        self.pos_drop4 = nn.Dropout(p=drop_rate)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([SoftmaxFreeTransformerBlock(
            dim=embed_dims[0], num_heads=num_heads[0], drop_path=dpr[cur + i], ratio=sr_ratios[0], conv_size=9,
            max_iter=newton_max_iter, kernel_method=kernel_method)
            for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.ModuleList([SoftmaxFreeTransformerBlock(
            dim=embed_dims[1], num_heads=num_heads[1], drop_path=dpr[cur + i], ratio=sr_ratios[1], conv_size=5,
            max_iter=newton_max_iter, kernel_method=kernel_method)
            for i in range(depths[1])])

        cur += depths[1]
        self.block3 = nn.ModuleList([SoftmaxFreeTransformerBlock(
            dim=embed_dims[2], num_heads=num_heads[2], drop_path=dpr[cur + i], ratio=sr_ratios[2], conv_size=3,
            max_iter=newton_max_iter, kernel_method=kernel_method)
            for i in range(depths[2])])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm = norm_layer(embed_dims[3])

        # cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        trunc_normal_(self.pos_embed4, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        # stage 1
        x, (H, W) = self.patch_embed1(x)
        Bp, Np, Cp = self.pos_embed1.shape
        pos_embed1 = rearrange(self.pos_embed1, 'b (h w) c -> b c h w', h=int(Np ** 0.5), w=int(Np ** 0.5))
        pos_embed1 = F.interpolate(pos_embed1, size=(H, W), mode='bilinear', align_corners=False)
        pos_embed1 = pos_embed1.flatten(2).transpose(-1, -2)
        x = x + pos_embed1
        x = self.pos_drop1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 2
        x, (H, W) = self.patch_embed2(x)
        Bp, Np, Cp = self.pos_embed2.shape
        pos_embed2 = rearrange(self.pos_embed2, 'b (h w) c -> b c h w', h=int(Np ** 0.5), w=int(Np ** 0.5))
        pos_embed2 = F.interpolate(pos_embed2, size=(H, W), mode='bilinear', align_corners=False)
        pos_embed2 = pos_embed2.flatten(2).transpose(-1, -2)
        x = x + pos_embed2
        x = self.pos_drop2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3
        x, (H, W) = self.patch_embed3(x)
        Bp, Np, Cp = self.pos_embed3.shape
        pos_embed3 = rearrange(self.pos_embed3, 'b (h w) c -> b c h w', h=int(Np ** 0.5), w=int(Np ** 0.5))
        pos_embed3 = F.interpolate(pos_embed3, size=(H, W), mode='bilinear', align_corners=False)
        pos_embed3 = pos_embed3.flatten(2).transpose(-1, -2)
        x = x + pos_embed3
        x = self.pos_drop3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 4
        x, (H, W) = self.patch_embed4(x)
        Bp, Np, Cp = self.pos_embed4.shape
        pos_embed4 = rearrange(self.pos_embed4[:, 1:, :], 'b (h w) c -> b c h w', h=int(Np ** 0.5), w=int(Np ** 0.5))
        pos_embed4 = F.interpolate(pos_embed4, size=(H, W), mode='bilinear', align_corners=False)
        pos_embed4 = rearrange(pos_embed4, 'b c h w -> b (h w) c')
        pos_embed4 = torch.cat((self.pos_embed4[:, 0, :].unsqueeze(1), pos_embed4), dim=1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed4
        for blk in self.block4:
            x = blk(x, H, W)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x



class SoftmaxFreeNormVisionTransformer(SoftmaxFreeVisionTransformer):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], 
                 mlp_ratios=[4, 4, 4, 4], 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0.,
                 attn_drop_rate=0., 
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], 
                 sr_ratios=[8, 4, 2, 1], 
                 newton_max_iter=20, 
                 kernel_method="cuda"):
        super(SoftmaxFreeNormVisionTransformer, self).__init__(img_size, 
                                                                  patch_size, 
                                                                  in_chans, 
                                                                  num_classes, 
                                                                  embed_dims,
                                                                  num_heads, 
                                                                  mlp_ratios, 
                                                                  qkv_bias, 
                                                                  qk_scale, 
                                                                  drop_rate,
                                                                  attn_drop_rate, 
                                                                  drop_path_rate, 
                                                                  norm_layer,
                                                                  depths, 
                                                                  sr_ratios, 
                                                                  newton_max_iter, 
                                                                  kernel_method)
        
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([SoftmaxFreeNormTransformerBlock(
            dim=embed_dims[0], num_heads=num_heads[0], drop_path=dpr[cur + i], ratio=sr_ratios[0], conv_size=9,
            max_iter=newton_max_iter, kernel_method=kernel_method)
            for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.ModuleList([SoftmaxFreeNormTransformerBlock(
            dim=embed_dims[1], num_heads=num_heads[1], drop_path=dpr[cur + i], ratio=sr_ratios[1], conv_size=5,
            max_iter=newton_max_iter, kernel_method=kernel_method)
            for i in range(depths[1])])

        cur += depths[1]
        self.block3 = nn.ModuleList([SoftmaxFreeNormTransformerBlock(
            dim=embed_dims[2], num_heads=num_heads[2], drop_path=dpr[cur + i], ratio=sr_ratios[2], conv_size=3,
            max_iter=newton_max_iter, kernel_method=kernel_method)
            for i in range(depths[2])])



@register_model
def soft_tiny(pretrained=False, **kwargs):
    model = SoftmaxFreeVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[2, 4, 10, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 2, 3, 2], sr_ratios=[8, 4, 2, 1], 
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def soft_small(pretrained=False, **kwargs):
    model = SoftmaxFreeVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[2, 4, 10, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 3, 20, 4], sr_ratios=[8, 4, 2, 1], 
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def soft_medium(pretrained=False, **kwargs):
    model = SoftmaxFreeVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 288, 512], num_heads=[2, 4, 9, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 3, 29, 5], sr_ratios=[8, 4, 2, 1], 
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def soft_large(pretrained=False, **kwargs):
    model = SoftmaxFreeVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[2, 4, 10, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 3, 40, 5], sr_ratios=[8, 4, 2, 1], 
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def soft_huge(pretrained=False, **kwargs):
    model = SoftmaxFreeVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 352, 512], num_heads=[2, 4, 11, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 5, 49, 5], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def soft_norm_tiny(pretrained=False, **kwargs):
    model = SoftmaxFreeNormVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[2, 4, 10, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 2, 3, 2], sr_ratios=[8, 4, 2, 1], 
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def soft_norm_small(pretrained=False, **kwargs):
    model = SoftmaxFreeNormVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[2, 4, 10, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 3, 20, 4], sr_ratios=[8, 4, 2, 1], 
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def soft_norm_medium(pretrained=False, **kwargs):
    model = SoftmaxFreeNormVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 288, 512], num_heads=[2, 4, 9, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 3, 29, 5], sr_ratios=[8, 4, 2, 1], 
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def soft_norm_large(pretrained=False, **kwargs):
    model = SoftmaxFreeNormVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[2, 4, 10, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 3, 40, 5], sr_ratios=[8, 4, 2, 1], 
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def soft_norm_huge(pretrained=False, **kwargs):
    model = SoftmaxFreeNormVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 352, 512], num_heads=[2, 4, 11, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 5, 49, 5], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model
