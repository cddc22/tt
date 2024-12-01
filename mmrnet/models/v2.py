import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import MLP, PointNetConv, fps, global_mean_pool, radius, knn, ChebConv

class DeformConv(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=1):
        super(DeformConv, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # 添加BatchNorm和Dropout
        self.offset_net = torch.nn.Sequential(
            torch.nn.Linear(3, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(32, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 3)
        )

    def forward(self, pos):
        original_shape = pos.shape
        if len(original_shape) == 2:
            pos = pos.unsqueeze(0)

        batch_size, num_points, _ = pos.shape

        # 重塑以适应BatchNorm1d
        pos_reshaped = pos.reshape(-1, 3)
        offset = self.offset_net(pos_reshaped)
        offset = offset.reshape(batch_size, num_points, 3)

        # 限制offset的范围
        offset = torch.tanh(offset) * 0.1

        # 残差连接
        deformed_pos = pos + offset

        if len(original_shape) == 2:
            deformed_pos = deformed_pos.squeeze(0)

        return deformed_pos

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)
        self.deform_conv = DeformConv()
        #self.bn = torch.nn.BatchNorm1d(nn.layers[-1].out_features)

    def forward(self, x, pos, batch):
        identity = x  # 保存输入用于残差连接

        deformed_pos = self.deform_conv(pos)
        idx = fps(deformed_pos, batch, ratio=self.ratio)
        row, col = radius(deformed_pos, deformed_pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (deformed_pos, deformed_pos[idx]), edge_index)
        #
        # # 应用BatchNorm
        # if x is not None:
        #     x = self.bn(x)

        # 残差连接（如果特征维度匹配）
        if identity is not None and identity.shape == x.shape:
            x = x + identity

        pos, batch = deformed_pos[idx], batch[idx]
        return x, pos, batch

class TransitionDown(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)
        self.deform_conv = DeformConv()
        self.bn = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x, pos, batch):
        identity = x  # 保存输入用于残差连接

        deformed_pos = self.deform_conv(pos)
        id_clusters = fps(deformed_pos, ratio=self.ratio, batch=batch)
        sub_batch = batch[id_clusters] if batch is not None else None
        id_k_neighbor = knn(deformed_pos, deformed_pos[id_clusters], k=self.k,
                            batch_x=batch, batch_y=sub_batch)

        x = self.mlp(x)
        x = self.bn(x)  # 应用BatchNorm

        x_out = torch_geometric.utils.scatter(x[id_k_neighbor[1]], id_k_neighbor[0],
                                              dim=0, dim_size=id_clusters.size(0), reduce='max')

        sub_pos, out = deformed_pos[id_clusters], x_out
        return out, sub_pos, sub_batch

class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn
        #self.bn = torch.nn.BatchNorm1d(nn.layers[-1].out_features)

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        #x = self.bn(x)  # 应用BatchNorm
        x = global_mean_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class PointNetChebDconv(torch.nn.Module):
    def __init__(self, info=None):
        super().__init__()
        self.num_classes = info['num_classes']

        # 主干网络
        self.sa1_module = SAModule(0.5, 0.2, MLP([3 * 2, 64, 64, 128]))
        self.transition_down1 = TransitionDown(in_channels=128, out_channels=256)
        self.sa2_module = SAModule(0.25, 0.4, MLP([256 + 3, 128, 128, 256]))
        self.transition_down2 = TransitionDown(in_channels=256, out_channels=512)
        self.sa3_module = GlobalSAModule(MLP([512 + 3, 256, 512, 1024]))

        # ChebConv层
        self.cheb_conv1 = ChebConv(256, 256, K=3)
        self.cheb_conv2 = ChebConv(512, 512, K=3)

        # BatchNorm层
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(512)

        if self.num_classes is None:
            self.num_points = info['num_keypoints']
            point_branches = {}
            for i in range(self.num_points):
                point_branches[f'branch_{i}'] = MLP([1024, 512, 256, 3], dropout=0.5)
            self.mlp = torch.nn.ModuleDict(point_branches)
        else:
            self.mlp = MLP([1024, 512, 256, self.num_classes], dropout=0.5, norm=None)

    def forward(self, data):
        batchsize = data.shape[0]
        npoints = data.shape[1]
        x = data.reshape((batchsize * npoints, 3))
        batch = torch.arange(batchsize).repeat_interleave(npoints).to(x.device)

        sa0_out = (x, x, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa1_down = self.transition_down1(*sa1_out)

        x, pos, batch = sa1_down
        edge_index = knn(pos, pos, k=16, batch_x=batch, batch_y=batch)
        identity1 = x  # 保存用于残差连接
        x = self.cheb_conv1(x, edge_index)
        x = self.bn1(x)
        if x.shape == identity1.shape:  # 残差连接
            x = x + identity1

        sa2_out = self.sa2_module(x, pos, batch)
        sa2_down = self.transition_down2(*sa2_out)

        x, pos, batch = sa2_down
        edge_index = knn(pos, pos, k=16, batch_x=batch, batch_y=batch)
        identity2 = x  # 保存用于残差连接
        x = self.cheb_conv2(x, edge_index)
        x = self.bn2(x)
        if x.shape == identity2.shape:  # 残差连接
            x = x + identity2

        sa3_out = self.sa3_module(x, pos, batch)
        x, pos, batch = sa3_out

        if self.num_classes is None:
            y = []
            for i in range(self.num_points):
                y.append(self.mlp[f'branch_{i}'](x))
            y = torch.stack(y, dim=1)
        else:
            y = self.mlp(x)
        return y

