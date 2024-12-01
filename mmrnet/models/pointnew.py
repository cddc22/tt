import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch_geometric.nn import global_mean_pool, knn_graph
import math

class PointTransformerLayer(nn.Module):
 def __init__(self, d_model):
  super(PointTransformerLayer, self).__init__()
  self.d_model = d_model
  self.linear_q = nn.Linear(d_model, d_model)
  self.linear_k = nn.Linear(d_model, d_model)
  self.linear_v = nn.Linear(d_model, d_model)
  self.linear_out = nn.Linear(d_model, d_model)

 def forward(self, x):
  batch_size, seq_length, _ = x.size()

  # Generate Q, K, V matrices
  Q = self.linear_q(x)
  K = self.linear_k(x)
  V = self.linear_v(x)

  # Calculate attention scores
  scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.d_model)
  attn = F.softmax(scores, dim=-1)

  # Attend to values
  context = torch.matmul(attn, V)
  out = self.linear_out(context)

  return out, attn

class LFPointTransformer(nn.Module):
 def __init__(self, input_dim=3, d_model=5):
  super(LFPointTransformer, self).__init__()
  self.input_dim = input_dim
  self.d_model = d_model
  self.input_linear = nn.Linear(input_dim, d_model)
  self.transformer_layer = PointTransformerLayer(d_model)

 def select_points(self, in_mat, attn):
  batch_size, num_points, _ = in_mat.size()
  selected_points = torch.zeros(batch_size, 110, 3, device=in_mat.device)

  for i in range(batch_size):
   # Get the attention scores for each point (assuming it's from self-attention's diagonal)
   point_attn = attn[i].diagonal(dim1=-2, dim2=-1)  # Take the diagonal to get [96]
   # Divide the attention scores of 96 points into 96 groups, each with 6 points
   attn_sums = point_attn.view(-1, 5).sum(dim=1)  # [96]
   # Select the index of the group with the highest sum of attention
   _, max_group_idx = attn_sums.max(dim=0)
   # Get the indices of the points in that group
   group_indices = torch.arange(num_points, device=in_mat.device).view(-1, 5)[max_group_idx]
   # Retrieve the coordinates of this group of points
   group_points = in_mat[i][group_indices]
   # Calculate the centroid
   centroid = group_points.mean(dim=0)
   # Calculate the distance of all points to the centroid
   all_points = in_mat[i][:, :3]  # Only take the data from the first three dimensions (the coordinates of the point cloud)
   distances = torch.norm(all_points - centroid[:3], dim=1)
   # Select the 96 closest points
   _, closest_indices = distances.topk(110, largest=False)
   # Save the selected points
   selected_points[i] = in_mat[i][closest_indices]

  return selected_points

 def forward(self, in_mat):
  # Convert input to d_model size
  x = self.input_linear(in_mat)
  # Pass through the transformer layer
  transformed, attn = self.transformer_layer(x)
  # Select important points
  important_points = self.select_points(in_mat, attn)
  # Return the indices of the most and least important points
  return important_points

class BasePointTiNet(nn.Module):
 def __init__(self):
  super(BasePointTiNet, self).__init__()
  self.conv1 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=1)
  self.cb1 = nn.BatchNorm1d(8)
  self.caf1 = nn.ReLU()

  self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=1)
  self.cb2 = nn.BatchNorm1d(16)
  self.caf2 = nn.ReLU()

  self.conv3 = nn.Conv1d(in_channels=16, out_channels=24, kernel_size=1)
  self.cb3 = nn.BatchNorm1d(24)
  self.caf3 = nn.ReLU()

 def forward(self, in_mat): # in_mat:(400, 96, 5) (batchsize * length_size, pc_num_ti, [x,y,z,velocity,intensity])
  x = in_mat.transpose(1,2)   #convert       # x:(400, 5, 96) point(x,y,z,range,intensity,velocity)

  x = self.caf1(self.cb1(self.conv1(x)))  # x:(400, 8, 96)
  x = self.caf2(self.cb2(self.conv2(x)))  # x:(400, 16, 96)
  x = self.caf3(self.cb3(self.conv3(x)))  # x:(400, 24, 96)

  x = x.transpose(1,2)  # x:(400, 96, 24)
  x = torch.cat((in_mat[:,:,:3], x), -1)   # x:(400, 96, 29)  拼接了x,y,z,range
  return x

class GlobalPointTiNet(nn.Module):
 def __init__(self):
  super(GlobalPointTiNet, self).__init__()

  self.conv1 = nn.Conv1d(in_channels= 24 + 3,   out_channels=48,  kernel_size=1)
  self.cb1 = nn.BatchNorm1d(48)
  self.caf1 = nn.ReLU()

  self.conv2 = nn.Conv1d(in_channels=48,   out_channels=72,  kernel_size=1)
  self.cb2 = nn.BatchNorm1d(72)
  self.caf2 = nn.ReLU()

  self.conv3 = nn.Conv1d(in_channels=72, out_channels=110, kernel_size=1)
  self.cb3 = nn.BatchNorm1d(110)
  self.caf3 = nn.ReLU()

  self.attn=nn.Linear(110, 1)
  self.softmax=nn.Softmax(dim=1)

 def forward(self, x):
  # x:(128, 110, 27)
  x = x.transpose(1,2)   # x:(128, 27, 110)
  #print("x shape before gpointnet:", x.shape)

  x = self.caf1(self.cb1(self.conv1(x)))   # x:(128, 48, 110)
  x = self.caf2(self.cb2(self.conv2(x)))   # x:(128, 72, 110)
  x = self.caf3(self.cb3(self.conv3(x)))   # x:(128, 96, 110)
 # print("x shape before gpointnet:", x.shape)

  x = x.transpose(1,2)   # x:(128, 110, 110)

  attn_weights=self.softmax(self.attn(x))   # attn_weights:(128, 110, 1)
  #print('attn_weights',attn_weights.shape)
  attn_vec=torch.sum(x*attn_weights, dim=1)  # attn_vec:(128, 96)   * times

  return attn_vec

class GlobalTiRNN(nn.Module):
 def __init__(self):
  super(GlobalTiRNN, self).__init__()
  self.rnn=nn.LSTM(input_size=110, hidden_size=110//2, num_layers=3, batch_first=True, dropout=0.1, bidirectional=True)
  self.fc1 = nn.Linear(110, 16)
  self.faf1 = nn.ReLU()
  self.fc2 = nn.Linear(16, 4)

 def forward(self, x, h0, c0):
  g_vec, (hn, cn)=self.rnn(x, (h0, c0))

  return g_vec

class GlobalTiModule(nn.Module):
 def __init__(self):
  super(GlobalTiModule, self).__init__()
  # self.lfpointtransformaernet=LFPointTransformer()
  self.bpointnet=BasePointTiNet()
  self.gpointnet=GlobalPointTiNet()
  self.grnn=GlobalTiRNN()

 def forward(self, x, h0, c0,  batch_size, length_size):
  # important_points = self.lfpointtransformaernet(x)
  # print('important_points',important_points.shape)

  x=self.bpointnet(x)

  x=self.gpointnet(x)


  x=x.view(batch_size, length_size, 110)

  g_vec=self.grnn(x, h0, c0)
  return g_vec



class PointNEW(nn.Module):
 def __init__(self, info):
  super(PointNEW, self).__init__()
  # Support both classification and keypoint detection
  self.num_classes = info.get('num_classes')
  self.num_keypoints = info.get('num_keypoints')
  self.hidden_size = 110
  self.num_layers = 2

  # Point feature extraction
  self.point_transformer = GlobalTiModule()

  # LSTM for temporal features if needed
  self.lstm = nn.LSTM(
   input_size=110,
   hidden_size=self.hidden_size//2,
   num_layers=self.num_layers,
   batch_first=True,
   bidirectional=True
  )

  # Output layer
  if self.num_classes is not None:
   # Classification
   self.output_layer = nn.Linear(self.hidden_size, self.num_classes)
  else:
   # Keypoint detection
   self.output_layer = nn.ModuleDict({
    f'branch_{i}': nn.Linear(self.hidden_size, 3)
    for i in range(self.num_keypoints)
   })

 def forward(self, data):
  # Input shape: (batch_size, num_points, 3)
  # // 向下取整
  length_size = 1
  batchsize = data.shape[0]

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  h0 = torch.zeros((6, batchsize, 110//2), dtype=torch.float32,device=device)  # 3，16，96
  c0 = torch.zeros((6, batchsize, 110//2), dtype=torch.float32,device=device)



  # Point feature extraction
  g_vec_parsing0 = self.point_transformer(data, h0, c0, batchsize, length_size)


  # LSTM processing
  lstm_out, _ = self.lstm(g_vec_parsing0)

  # Take the last time step
  x = lstm_out[:, -1, :]

  # Output layer
  if self.num_classes is not None:
   # Classification
   out = self.output_layer(x)
   return F.log_softmax(out, dim=1)
  else:
   # Keypoint detection
   y = []
   for i in range(self.num_keypoints):
    y.append(self.output_layer[f'branch_{i}'](x))
   return torch.stack(y, dim=1)


if __name__ == '__main__':
 # Example usage for classification
 classification_info = {
  'num_classes': 49,
  'num_keypoints': None
 }

 # Example usage for keypoint detection
 keypoint_info = {
  'num_classes': None,
  'num_keypoints': 10
 }

 # Classification model
 classification_model = PointNEW(classification_info)

 # Keypoint detection model
 keypoint_model = PointNEW(keypoint_info)

 # Generate dummy input data
 data_points = torch.rand((128, 110, 3), dtype=torch.float32)

 # Test classification
 print("Classification Output:")
 classification_output = classification_model(data_points)
 print(classification_output.shape)

 # # Test keypoint detection
 # print("\nKeypoint Detection Output:")
 # keypoint_output = keypoint_model(data_points)
 # print(keypoint_output.shape)