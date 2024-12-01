import torch


from mmrnet.models.v2 import PointNetChebDconv
# from mmrnet.models import PointNetMamba
# from mmrnet.models.pointnetddm import PointNetWithDDM
# from mmrnet.models.pointnetgauss import PointNetGauss

from pointnetgauss import PointNetGauss
from pointnetchet import PointNetCheb
from pointknn import PointNetKnn

if __name__ == "__main__":
    # 模拟输入数据
    batch_size = 128
    num_points = 110
    data = torch.randn(batch_size, num_points, 3)

    # 配置信息
    info = {
        'num_classes': 49,
        'num_keypoints': None
    }

    # 创建模型
    model = PointNetChebDconv(info)

    # 前向传播
    output = model(data)
    print(model)
    print(output.shape)
