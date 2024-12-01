from .dgcnn import DGCNN
from .mlp import MLP
from .pointmaba import PointNetMamba

from .pointnet import PointNet
from .point_transformer import PointTransformer
from .pointmlp import PointMLP
from .pointnetchet import PointNetCheb
from .pointnetddm import PointNetWithDDM
from .pointnew import PointNEW
from .pointdconv import PointNetChebDconv
from .pointvae import PointNetVAE

model_map = {
    'dgcnn': DGCNN,
    'mlp': MLP,
    'pointnet': PointNet,
    'pointtransformer': PointTransformer,
    'pointmlp': PointMLP,
    'pointnew': PointNEW,
    'pointnetvae': PointNetVAE,
    'pointmamba': PointNetMamba,
    'pointddm': PointNetWithDDM,
    'pointcheb': PointNetCheb,
    'pointall': PointNetChebDconv,

}
