import torch
import torch.nn as nn
import math
import numpy as np
from SOFT.kernel.subtraction import subtraction_gaussian_kernel

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
    def __init__(self, dim, num_heads, q_len, k_len, num_landmark, use_conv, max_iter=20, kernel_method="cuda"):
        super().__init__()

        self.head_dim = int(dim // num_heads)
        self.num_head = num_heads

        self.num_landmarks = num_landmark
        self.q_seq_len = q_len
        self.k_seq_len = k_len
        self.max_iter = max_iter

        if kernel_method == "torch":
            self.kernel_function = subtraction_gaussian_kernel_torch
        elif kernel_method == "cuda":
            self.kernel_function = subtraction_gaussian_kernel
        else:
            assert False, "please choose kernel method from torch and cuda"

        ratio = int(np.sqrt(self.q_seq_len // self.num_landmarks))
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

    def forward(self, Q, V):
        b, nhead, N, headdim, = Q.size()
        # Q: [b, num_head, N, head_dim]
        Q = Q / math.sqrt(math.sqrt(self.head_dim))
        K=Q
        if self.num_landmarks == self.q_seq_len:
            Q_landmarks = Q.reshape(b * self.num_head, int(np.sqrt(self.q_seq_len)) * int(np.sqrt(self.q_seq_len)) + 1,
                                     self.head_dim)
            Q_landmarks = self.Qlandmark_op(Q_landmarks)
            Q_landmarks = self.Qnorm_act(Q_landmarks).reshape(b, self.num_head, self.num_landmarks + 1, self.head_dim)
            K_landmarks = Q_landmarks
            attn = self.kernel_function(Q_landmarks, K_landmarks.transpose(-1, -2).contiguous())
            attn = torch.exp(-attn / 2)
            X = torch.matmul(attn, V)

            h = w = int(np.sqrt(N))
            if self.use_conv:
                V_ = V[:, :, 1:, :]
                cls_token = V[:, :, 0, :].unsqueeze(2)
                V_ = V_.reshape(b, nhead, h, w, headdim)
                V_ = V_.permute(0, 4, 1, 2, 3).reshape(b * headdim, nhead, h, w)
                out = self.conv(V_).reshape(b, headdim, nhead, h, w).flatten(3).permute(0, 2, 3, 1)
                out = torch.cat([cls_token, out], dim=2)
                X += out
        else:
            Q_landmarks = Q.reshape(b * self.num_head, int(np.sqrt(self.q_seq_len)) * int(np.sqrt(self.q_seq_len)),
                                    self.head_dim).reshape(b * self.num_head, int(np.sqrt(self.q_seq_len)),
                                                           int(np.sqrt(self.q_seq_len)),
                                                           self.head_dim).permute(0, 3, 1, 2)
            Q_landmarks = self.Qlandmark_op(Q_landmarks)
            Q_landmarks = Q_landmarks.flatten(2).transpose(1, 2).reshape(b, self.num_head, self.num_landmarks,
                                                                         self.head_dim)
            Q_landmarks = self.Qnorm_act(Q_landmarks)
            K_landmarks = Q_landmarks

            kernel_1_ = self.kernel_function(Q, K_landmarks.transpose(-1, -2).contiguous())
            kernel_1_ = torch.exp(-kernel_1_/2)

            kernel_2_ = self.kernel_function(Q_landmarks, K_landmarks.transpose(-1, -2).contiguous())
            kernel_2_ = torch.exp(-kernel_2_/2)

            kernel_3_ = kernel_1_.transpose(-1, -2)

            X = torch.matmul(torch.matmul(kernel_1_, self.newton_inv(kernel_2_)), torch.matmul(kernel_3_, V))

            h = w = int(np.sqrt(N))
            if self.use_conv:
                V = V.reshape(b, nhead, h, w, headdim)
                V = V.permute(0, 4, 1, 2, 3).reshape(b*headdim, nhead, h, w)
                X += self.conv(V).reshape(b, headdim, nhead, h, w).flatten(3).permute(0, 2, 3, 1)

        return X

    def newton_inv(self, mat):
        P = mat
        I = torch.eye(mat.size(-1), device=mat.device)
        alpha = 2 / (torch.max(torch.sum(mat, dim=-1)) ** 2)
        beta = 0.5
        V = alpha * P
        pnorm = torch.max(torch.sum(torch.abs(I - torch.matmul(P, V)), dim=-2))
        err_cnt = 0
        while pnorm > 1.01 and err_cnt < 10:
            alpha *= beta
            V = alpha * P
            pnorm = torch.max(torch.sum(torch.abs(I - torch.matmul(P, V)), dim=-2))
            err_cnt += 1

        for i in range(self.max_iter):
            V = 2 * V - V @ P @ V
        return V


class SoftmaxFreeAttention(nn.Module):
    def __init__(self, dim, num_heads, q_len, k_len, num_landmark, conv_size, max_iter=20, kernel_method="cuda"):
        super().__init__()

        self.grad_checkpointing = True
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_head = num_heads

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

        self.attn = SoftmaxFreeAttentionKernel(dim, num_heads, q_len, k_len, num_landmark, conv_size, max_iter, kernel_method)

        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, X, return_QKV = False):

        Q = self.split_heads(self.W_q(X))
        V = self.split_heads(self.W_v(X))
        attn_out = self.attn(Q, V)
        attn_out = self.combine_heads(attn_out)

        out = self.ff(attn_out)

        if return_QKV:
            return out, (Q, V)
        else:
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
    def __init__(self, dim, num_heads, q_len, k_len, num_landmark, conv_size, drop_path=0., max_iter=20, kernel_method="torch"):
        super().__init__()
        self.dim = dim
        self.hidden_dim = int(4*dim)

        self.mha = SoftmaxFreeAttention(dim, num_heads, q_len, k_len, num_landmark, conv_size, max_iter, kernel_method)

        self.dropout1 = torch.nn.Dropout(p=drop_path)
        self.norm1 = nn.LayerNorm(self.dim)

        self.ff1 = nn.Linear(self.dim, self.hidden_dim)
        self.act = Approx_GeLU()
        self.ff2 = nn.Linear(self.hidden_dim, self.dim)

        self.dropout2 = torch.nn.Dropout(p=drop_path)
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(self, X, return_QKV = False):

        if return_QKV:
            mha_out, QKV = self.mha(X, return_QKV = True)
        else:
            mha_out = self.mha(X)

        mha_out = self.norm1(X + self.dropout1(mha_out))
        ff_out = self.ff2(self.act(self.ff1(mha_out)))
        mha_out = self.norm2(mha_out + self.dropout2(ff_out))

        if return_QKV:
            return mha_out, QKV
        else:
            return mha_out


class SoftmaxFreeTrasnformerBlock(nn.Module):
    def __init__(self, dim, num_heads, H, W, drop_path=0., conv_size=3, max_iter=20, kernel_method="cuda"):
        super().__init__()
        seq_len = 49
        self.att = SoftmaxFreeTransformer(dim, num_heads, int(H*W), int(H*W), seq_len, conv_size, drop_path, max_iter, kernel_method)

    def forward(self, x):
        x = self.att(x)
        return x
