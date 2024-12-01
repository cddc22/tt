
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import MLP, PointNetConv, fps, global_mean_pool, radius, knn


import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.in_proj = nn.Linear(d_model, expand * d_model)
        self.conv1d = nn.Conv1d(expand * d_model, expand * d_model,
                                kernel_size=d_conv, groups=expand * d_model,
                                padding=d_conv//2)

        # SSM (State Space Model) parameters
        self.x_proj = nn.Linear(d_model, d_state + d_conv, bias=False)
        self.dt_proj = nn.Linear(d_state + d_conv, d_model, bias=True)


        self.out_proj = nn.Linear(expand * d_model, d_model)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if 1D

        # Project input
        z = self.in_proj(x)

        # Reshape for convolution (batch, channels, length)
        batch_size, _ = x.shape
        z = z.view(batch_size, self.expand * self.d_model, -1)

        # Convolution
        z = self.conv1d(z)
        z = z.permute(0, 2, 1)  # Back to (B, L, C)
        z = z.reshape(-1, z.size(-1))  # Flatten z if necessary

        # State Space Model component
        z_state = self.x_proj(x)
        dt = F.softplus(self.dt_proj(z_state))

        # Ensure z and dt have the same batch size dimension
        if z.size(0) != dt.size(0):
            if z.size(0) > dt.size(0):
                # 重复 dt 的批次维度，直到匹配
                dt = dt.repeat(z.size(0) // dt.size(0), 1)
            else:
                # 如果 z 的批次维度小于 dt，则可能需要裁剪
                dt = dt[:z.size(0)]


        # Now multiply z with dt, ensuring dimensions match
        dt = dt.repeat(1,2)

        out = self.out_proj(z * dt)
        return out



class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn, use_mamba=False, mamba_dim=None):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

        # 可选的Mamba层
        self.use_mamba = use_mamba
        if use_mamba:
            assert mamba_dim is not None, "Must provide mamba_dim when use_mamba is True"
            self.mamba = MambaBlock(d_model=mamba_dim)

    def forward(self, x, pos, batch):
        # FPS下采样
        idx = fps(pos, batch, ratio=self.ratio)

        # 半径图构建
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)

        # 目标点特征
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)

        # 可选的Mamba特征增强
        if self.use_mamba:
            x = self.mamba(x)

        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn, use_mamba=False, mamba_dim=None):
        super().__init__()
        self.nn = nn

        # 可选的Mamba层
        self.use_mamba = use_mamba
        if use_mamba:
            assert mamba_dim is not None, "Must provide mamba_dim when use_mamba is True"
            self.mamba = MambaBlock(d_model=mamba_dim)

    def forward(self, x, pos, batch):
        # 拼接特征
        x = self.nn(torch.cat([x, pos], dim=1))

        # 可选的Mamba特征增强
        if self.use_mamba:
            x = self.mamba(x)

        # 全局平均池化
        x = global_mean_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class TransitionDown(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16, use_mamba=False, mamba_dim=None):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

        # 可选的Mamba层
        self.use_mamba = use_mamba
        if use_mamba:
            assert mamba_dim is not None, "Must provide mamba_dim when use_mamba is True"
            self.mamba = MambaBlock(d_model=mamba_dim)

    def forward(self, x, pos, batch):
        # FPS下采样
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # K近邻采样
        sub_batch = batch[id_clusters] if batch is not None else None
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch)

        # MLP特征变换
        x = self.mlp(x)

        # 特征聚合
        x_out = torch_geometric.utils.scatter(x[id_k_neighbor[1]], id_k_neighbor[0], dim=0,
                                              dim_size=id_clusters.size(0), reduce='max')

        # 可选的Mamba特征增强
        if self.use_mamba:
            x_out = self.mamba(x_out)

        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


class PointNetMamba(torch.nn.Module):
    def __init__(self, info=None):
        super().__init__()
        self.num_classes = info['num_classes']

        # 引入Mamba增强的模块
        self.sa1_module = SAModule(
            0.5, 0.2,
            MLP([3 * 2, 64, 64, 128]),
            use_mamba=True,
            mamba_dim=128
        )
        self.transition_down1 = TransitionDown(
            in_channels=128,
            out_channels=256,
            use_mamba=False,
            mamba_dim=256
        )
        self.sa2_module = SAModule(
            0.25, 0.4,
            MLP([256 + 3, 128, 128, 256]),
            use_mamba=True,
            mamba_dim=256
        )
        self.transition_down2 = TransitionDown(
            in_channels=256,
            out_channels=512,
            use_mamba=False,
            mamba_dim=512
        )
        self.sa3_module = GlobalSAModule(
            MLP([512 + 3, 256, 512, 1024]),
            use_mamba=False,
            mamba_dim=1024
        )

        # 分类头
        if self.num_classes is None:
            self.num_points = info['num_keypoints']
            point_branches = {}
            for i in range(self.num_points):
                point_branches[f'branch_{i}'] = MLP([1024, 512, 256, 3])
            self.mlp = torch.nn.ModuleDict(point_branches)
        else:
            self.mlp = MLP([1024, 512, 256, self.num_classes], dropout=0.5, norm=None)

    def forward(self, data):
        # 数据预处理
        batchsize = data.shape[0]
        npoints = data.shape[1]
        x = data.reshape((batchsize * npoints, 3))
        batch = torch.arange(batchsize).repeat_interleave(npoints).to(x.device)

        # 特征提取
        sa0_out = (x, x, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa1_down = self.transition_down1(*sa1_out)
        sa2_out = self.sa2_module(*sa1_down)
        sa2_down = self.transition_down2(*sa2_out)
        sa3_out = self.sa3_module(*sa2_down)

        x, pos, batch = sa3_out

        # 分类/回归
        if self.num_classes is None:
            y = []
            for i in range(self.num_points):
                y.append(self.mlp[f'branch_{i}'](x))
            y = torch.stack(y, dim=1)
        else:
            y = self.mlp(x)
        return y

# 使用示例
def test_pointnet_mamba():
    # 模拟输入数据
    batch_size = 128
    num_points = 110
    data = torch.randn(batch_size, num_points, 3)

    # 配置信息
    info = {
        'num_classes': 10,
        'num_keypoints': None
    }

    # 创建模型
    model = PointNetMamba(info)

    # 前向传播
    output = model(data)
    print("模型输出形状:", output.shape)

if __name__ == "__main__":
    test_pointnet_mamba()