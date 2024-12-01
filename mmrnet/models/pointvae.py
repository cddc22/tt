import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MLP, PointNetConv, fps, global_mean_pool, radius, knn

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = MLP([input_dim, 512, 256])
        self.fc2_mean = torch.nn.Linear(256, latent_dim)
        self.fc2_logvar = torch.nn.Linear(256, latent_dim)

    def forward(self, x):
        x = self.fc1(x)
        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mean, logvar

class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc1 = MLP([latent_dim, 256, 512])
        self.fc2 = torch.nn.Linear(512, output_dim)

    def forward(self, z):
        z = self.fc1(z)
        out = self.fc2(z)
        return out

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_mean_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class TransitionDown(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels, out_channels], norm='batch_norm', dropout=0.3)

    def forward(self, x, pos, batch):
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)
        sub_batch = batch[id_clusters] if batch is not None else None
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch)
        x = self.mlp(x)
        x_out = torch_geometric.utils.scatter(x[id_k_neighbor[1]], id_k_neighbor[0], dim=0,
                                              dim_size=id_clusters.size(0), reduce='max')
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


class PointNetVAE(torch.nn.Module):
    def __init__(self, info=None, latent_dim=128):
        super().__init__()
        self.num_classes = info['num_classes']
        self.latent_dim = latent_dim

        # PointNet++ layers
        self.sa1_module = SAModule(0.5, 0.2, MLP([3 * 2, 64, 64, 128], norm='batch_norm', dropout=0.3))
        self.transition_down1 = TransitionDown(in_channels=128, out_channels=256)
        self.sa2_module = SAModule(0.25, 0.4, MLP([256 + 3, 128, 128, 512], norm='batch_norm', dropout=0.3))
        self.transition_down2 = TransitionDown(in_channels=512, out_channels=1024)
        self.sa3_module = GlobalSAModule(MLP([1024 + 3, 512, 512, 1024], norm='batch_norm', dropout=0.3))

        # Encoder and Decoder for VAE
        self.encoder = Encoder(1024, self.latent_dim)
        self.decoder = Decoder(self.latent_dim, 1024)

        if self.num_classes is None:
            self.num_points = info['num_keypoints']
            point_branches = {}
            for i in range(self.num_points):
                point_branches[f'branch_{i}'] = MLP([1024, 512, 256, 3], norm='batch_norm', dropout=0.3)
            self.mlp = torch.nn.ModuleDict(point_branches)
        else:
            self.mlp = MLP([1024, 512, 256, self.num_classes], dropout=0.5, norm='batch_norm')

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, data):
        batchsize = data.shape[0]
        npoints = data.shape[1]
        x = data.reshape((batchsize * npoints, 3))
        batch = torch.arange(batchsize).repeat_interleave(npoints).to(x.device)

        # PointNet++ forward pass
        sa0_out = (x, x, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa1_down = self.transition_down1(*sa1_out)
        sa2_out = self.sa2_module(*sa1_down)
        sa2_down = self.transition_down2(*sa2_out)
        sa3_out = self.sa3_module(*sa2_down)

        x, pos, batch = sa3_out

        # Flatten and encode
        x_flat = x.view(x.size(0), -1)  # Flatten the features before passing to the encoder
        mean, logvar = self.encoder(x_flat)
        z = self.reparameterize(mean, logvar)

        # Decode the latent vector
        decoded_x = self.decoder(z)

        # Return only the predicted values (y_hat)
        if self.num_classes is None:
            y = []
            for i in range(self.num_points):
                y.append(self.mlp[f'branch_{i}'](decoded_x))
            y = torch.stack(y, dim=1)
        else:
            y = self.mlp(decoded_x)

        return y  # Only return predictions (y)

