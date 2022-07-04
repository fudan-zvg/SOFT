import torch
import torch.nn as nn
import math
import numpy as np
from SOFT.kernel.subtraction import subtraction_gaussian_kernel
from SOFT.kernel.inverse import newton_inverse_kernel

class Approx_GeLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_checkpointing = True

    def func(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def forward(self, x):
        x = self.func(x)
        return x


def subtraction_gaussian_kernel_torch(q, k):
    # [B, H, H1*W1, C] @ [C, H2*W2] -> [B, H, H1*W1, H2*W2]
    matA_square = q ** 2. @ torch.ones(k.shape[-2:]).cuda()
    # [H1*W1, C] @ [B, H, C, H2*W2] -> [B, H, H1*W1, H2*W2]
    matB_square = torch.ones(q.shape[-2:]).cuda() @ k ** 2.
    return matA_square + matB_square - 2. * (q @ k)


class SoftmaxFreeAttentionKernel(nn.Module):
    def __init__(self, dim, num_heads, ratio, use_conv, max_iter=20, kernel_method="cuda"):
        super().__init__()

        self.head_dim = int(dim // num_heads)
        self.num_head = num_heads
        self.ratio = ratio
        self.max_iter = max_iter

        if kernel_method == "torch":
            self.kernel_function = subtraction_gaussian_kernel_torch
        elif kernel_method == "cuda":
            self.kernel_function = subtraction_gaussian_kernel
        else:
            assert False, "please choose kernel method from torch and cuda"

        if ratio == 1:
            self.Qlandmark_op = nn.Linear(self.head_dim, self.head_dim, bias=False)
            self.Qnorm_act = nn.Sequential(nn.LayerNorm(self.head_dim), nn.GELU())
        else:
            self.Qlandmark_op = nn.Conv2d(self.head_dim, self.head_dim, kernel_size=ratio, stride=ratio, bias=False)
            self.Qnorm_act = nn.Sequential(nn.LayerNorm(self.head_dim), nn.GELU())

        self.use_conv = use_conv
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels=self.num_head, out_channels=self.num_head,
                kernel_size=(self.use_conv, self.use_conv), padding=(self.use_conv // 2, self.use_conv // 2),
                bias=False,
                groups=self.num_head)

    def forward(self, Q, V, H, W):
        b, nhead, num_sequence, headdim = Q.size()
        # Q: [b, num_head, N, head_dim]
        Q = Q / math.sqrt(math.sqrt(headdim))
        K=Q
        if self.ratio == 1:
            Q_landmarks = Q.reshape(b * nhead, H * W + 1, headdim)
            Q_landmarks = self.Qlandmark_op(Q_landmarks)
            Q_landmarks = self.Qnorm_act(Q_landmarks).reshape(b, nhead, self.num_landmarks + 1, headdim)
            K_landmarks = Q_landmarks
            attn = self.kernel_function(Q_landmarks, K_landmarks.transpose(-1, -2).contiguous())
            attn = torch.exp(-attn / 2)
            X = torch.matmul(attn, V)

            if self.use_conv:
                V_ = V[:, :, 1:, :]
                cls_token = V[:, :, 0, :].unsqueeze(2)
                V_ = V_.reshape(b, nhead, H, W, headdim)
                V_ = V_.permute(0, 4, 1, 2, 3).reshape(b * headdim, nhead, H, W)
                out = self.conv(V_).reshape(b, headdim, nhead, H, W).flatten(3).permute(0, 2, 3, 1)
                out = torch.cat([cls_token, out], dim=2)
                X += out
        else:
            Q_landmarks = Q.reshape(b * nhead, H * W, 
                                    headdim).reshape(b * nhead, 
                                                           H, W, headdim).permute(0, 3, 1, 2)
            Q_landmarks = self.Qlandmark_op(Q_landmarks)
            Q_landmarks = Q_landmarks.flatten(2).transpose(1, 2).reshape(b, nhead, -1, headdim)
            Q_landmarks = self.Qnorm_act(Q_landmarks)
            K_landmarks = Q_landmarks

            kernel_1_ = self.kernel_function(Q, K_landmarks.transpose(-1, -2).contiguous())
            kernel_1_ = torch.exp(-kernel_1_/2)

            kernel_2_ = self.kernel_function(Q_landmarks, K_landmarks.transpose(-1, -2).contiguous())
            kernel_2_ = torch.exp(-kernel_2_/2)

            kernel_3_ = kernel_1_.transpose(-1, -2)

            X = torch.matmul(torch.matmul(kernel_1_, newton_inverse_kernel(kernel_2_, self.max_iter)), torch.matmul(kernel_3_, V))

            if self.use_conv:
                V = V.reshape(b, nhead, H, W, headdim)
                V = V.permute(0, 4, 1, 2, 3).reshape(b*headdim, nhead, H, W)
                X += self.conv(V).reshape(b, headdim, nhead, H, W).flatten(3).permute(0, 2, 3, 1)

        return X


class SoftmaxFreeAttention(nn.Module):
    def __init__(self, dim, num_heads, ratio, conv_size, max_iter=20, kernel_method="cuda"):
        super().__init__()

        self.grad_checkpointing = True
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_head = num_heads

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

        self.attn = SoftmaxFreeAttentionKernel(dim, num_heads, ratio, conv_size, max_iter, kernel_method)

        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, X, H, W):

        Q = self.split_heads(self.W_q(X))
        V = self.split_heads(self.W_v(X))
        attn_out = self.attn(Q, V, H, W)
        attn_out = self.combine_heads(attn_out)

        out = self.ff(attn_out)
        return out

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X


class SoftmaxFreeTransformer(nn.Module):
    def __init__(self, dim, num_heads, ratio, conv_size, drop_path=0., max_iter=20, kernel_method="torch"):
        super().__init__()
        self.dim = dim
        self.hidden_dim = int(4*dim)

        self.mha = SoftmaxFreeAttention(dim, num_heads, ratio, conv_size, max_iter, kernel_method)

        self.dropout1 = torch.nn.Dropout(p=drop_path)
        self.norm1 = nn.LayerNorm(self.dim)

        self.ff1 = nn.Linear(self.dim, self.hidden_dim)
        self.act = Approx_GeLU()
        self.ff2 = nn.Linear(self.hidden_dim, self.dim)

        self.dropout2 = torch.nn.Dropout(p=drop_path)
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(self, X, H, W):
        mha_out = self.mha(X, H, W)
        mha_out = self.norm1(X + self.dropout1(mha_out))
        ff_out = self.ff2(self.act(self.ff1(mha_out)))
        mha_out = self.norm2(mha_out + self.dropout2(ff_out))
        return mha_out


class SoftmaxFreeTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ratio, drop_path=0., conv_size=3, max_iter=20, kernel_method="cuda"):
        super().__init__()
        self.att = SoftmaxFreeTransformer(dim, num_heads, ratio, conv_size, drop_path, max_iter, kernel_method)

    def forward(self, x, H, W):
        x = self.att(x, H, W)
        return x
