import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import MLP, PointNetConv, fps, global_mean_pool, radius, knn
from torch import nn

class OverlapSamplingModule(torch.nn.Module):
    def __init__(self, ratio, k, overlap_ratio=0.2):
        super().__init__()
        self.ratio = ratio
        self.k = k
        self.overlap_ratio = overlap_ratio

    def forward(self, pos, batch):
        # 基础采样
        base_idx = fps(pos, batch, ratio=self.ratio)

        # 计算重叠区域
        overlap_size = int(len(base_idx) * self.overlap_ratio)
        extra_idx = fps(pos, batch, ratio=self.overlap_ratio)

        # 合并采样点
        combined_idx = torch.cat([base_idx, extra_idx[:overlap_size]])

        # kNN压缩
        row, col = knn(pos, pos[combined_idx], k=self.k, batch_x=batch, batch_y=batch[combined_idx])

        return combined_idx, (row, col)

class TransformerBlock(torch.nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads)
        self.mlp = MLPWithoutBatchNorm([dim, dim*2, dim])
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        att_out, _ = self.attention(x, x, x)
        x = self.norm1(x + att_out)
        x = self.norm2(x + self.mlp(x))
        return x
class MLPWithoutBatchNorm(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class EnhancedSAModule(torch.nn.Module):
    def __init__(self, ratio, k, nn, transformer_dim=128):
        super().__init__()
        self.sampling = OverlapSamplingModule(ratio, k)
        self.conv = PointNetConv(nn, add_self_loops=False)
        self.transformer = TransformerBlock(transformer_dim)
        self.feature_reduction = nn.Linear(256, transformer_dim)  # Add Linear layer to reduce dimension

    def forward(self, x, pos, batch):
        # 重叠采样
        idx, (row, col) = self.sampling(pos, batch)
        edge_index = torch.stack([col, row], dim=0)

        # 特征提取
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)

        # Reduce feature dimension from 256 to 128
        x = self.feature_reduction(x)  # This step reduces the feature dimension to match transformer_dim

        # Transformer处理
        batch_size = x.size(0)
        x = x.view(batch_size, -1, x.size(-1))
        x = self.transformer(x)
        x = x.reshape(batch_size, -1)

        return x, pos[idx], batch[idx]



class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_mean_pool(x, batch)  # 改为全局平均池化
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class PointNetKnn(torch.nn.Module):
    def __init__(self, info=None):
        super().__init__()
        self.num_classes = info['num_classes']

        # 使用增强的SA模块
        self.sa1_module = EnhancedSAModule(0.5, 16, MLP([3 * 2, 64, 64, 128]))
        self.sa2_module = EnhancedSAModule(0.25, 16, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        if self.num_classes is None:
            self.num_points = info['num_keypoints']
            point_branches = {}
            for i in range(self.num_points):
                point_branches[f'branch_{i}'] = MLP([1024, 512, 256, 3])
            self.mlp = torch.nn.ModuleDict(point_branches)
        else:
            self.mlp = MLP([1024, 512, 256, self.num_classes], dropout=0.5, norm=None)

    def forward(self, data):
        batchsize = data.shape[0]
        npoints = data.shape[1]
        x = data.reshape((batchsize * npoints, 3))
        batch = torch.arange(batchsize).repeat_interleave(npoints).to(x.device)

        # 前向传播
        sa0_out = (x, x, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        x, pos, batch = sa3_out

        if self.num_classes is None:
            y = []
            for i in range(self.num_points):
                y.append(self.mlp[f'branch_{i}'](x))
            y = torch.stack(y, dim=1)
        else:
            y = self.mlp(x)
        return y