import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from torch import einsum
from torch_geometric.nn import MLP, PointNetConv, fps, global_mean_pool, radius, knn
import math






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
class TransitionDown(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
            super().__init__()
            self.k = k
            self.ratio = ratio
            self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch):
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)
        sub_batch = batch[id_clusters] if batch is not None else None
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch)
        x = self.mlp(x)
        x_out = torch_geometric.utils.scatter(x[id_k_neighbor[1]], id_k_neighbor[0], dim=0,
                                              dim_size=id_clusters.size(0), reduce='max')
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn, feature_dim):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)


    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)


        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class PointNetgaussSsm(torch.nn.Module):
    def __init__(self, info=None):
        super().__init__()
        self.num_classes = info['num_classes']

        # Modified SA modules with hybrid processing
        self.sa1_module = SAModule(0.5, 0.2, MLP([3 * 2, 64, 64, 128]), 128)
        self.transition_down1 = TransitionDown(in_channels=128, out_channels=256)
        self.sa2_module = SAModule(0.25, 0.4, MLP([256 + 3, 128, 128, 256]), 256)
        self.transition_down2 = TransitionDown(in_channels=256, out_channels=512)
        self.sa3_module = GlobalSAModule(MLP([512 + 3, 256, 512, 1024]))

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

        sa0_out = (x, x, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa1_down = self.transition_down1(*sa1_out)
        sa2_out = self.sa2_module(*sa1_down)
        sa2_down = self.transition_down2(*sa2_out)
        sa3_out = self.sa3_module(*sa2_down)

        x, pos, batch = sa3_out

        if self.num_classes is None:
            y = []
            for i in range(self.num_points):
                y.append(self.mlp[f'branch_{i}'](x))
            y = torch.stack(y, dim=1)
        else:
            y = self.mlp(x)
        return y