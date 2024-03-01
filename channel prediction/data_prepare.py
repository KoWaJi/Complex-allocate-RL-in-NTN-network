import torch
import scipy.io
import numpy as np
import random
from torch.utils.data import Dataset
import torch.nn.functional as F
import pickle
from torch.nn.utils.rnn import pad_sequence

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.data[idx]
        target = self.labels[idx]

        # 将数据转换为torch.Tensor（如果它们还不是）
        feature = torch.tensor(feature, dtype=torch.float32)  # 转换为Float
        target = torch.tensor(target, dtype=torch.float32)    # 同样转换为Float

        return feature, target
    
# 数据加载和预处理
def LoadAndCreate_data(channel_file, angel_file):
    channel_data = scipy.io.loadmat(channel_file)
    angle_data = scipy.io.loadmat(angel_file)
    CSI = channel_data['pow'].reshape(1,len(channel_data['pow'])).flatten()
    ANGLE = angle_data['angel'].flatten()
    ANGLE = np.degrees(ANGLE)          #弧度转换为度
    data = np.vstack([CSI, ANGLE])
    features = []
    targets = []
    # 遍历数据并构建特征和目标
    for i in range(data.shape[1]):
        elevation = data[1, i]
        seq_len = get_sequence_length(elevation)
        if seq_len > 0 and i >= seq_len:
            seq_feature = data[:, (i-seq_len):i]  # 提取序列
            seq_target = data[0, i]  # 当前的CSI值
            features.append(seq_feature)
            targets.append(seq_target)
    train_features, train_targets, test_features, test_targets = creat_dataset(features, targets)
    data = {'train_features': train_features, 'train_targets': train_targets, 'test_features': test_features, 'test_targets': test_targets}
    with open('channel prediction/datasets.pkl', 'wb') as f:
        pickle.dump(data, f)
    # 将特征和目标转换为NumPy数组
    return train_features, train_targets, test_features, test_targets


# 确定序列长度的函数
def get_sequence_length(elevation):
    if 20 <= elevation < 40:
        return 10
    elif 40 <= elevation < 50:
        return 9
    elif 50 <= elevation < 60:
        return 8
    elif 60 <= elevation < 70:
        return 7
    elif 70 <= elevation < 80:
        return 6
    elif 80 <= elevation <= 90:
        return 5
    else:
        return 0   

def creat_dataset(features, targets):

    # 数据点的总数量
    total_samples = len(features)

    # 生成索引列表并随机打乱
    indices = list(range(total_samples))
    random.shuffle(indices)

    # 确定测试集大小（10%）
    test_size = int(19999)

    # 分割索引为训练集和测试集
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    # 根据索引构建训练集和测试集
    train_features = [features[i] for i in train_indices]
    train_targets = [targets[i] for i in train_indices]
    test_features = [features[i] for i in test_indices]
    test_targets = [targets[i] for i in test_indices]

    return train_features, train_targets, test_features, test_targets


#用0填充序列长度函数
def collate_fn(batch):
    # batch中的每个元素形式为(data, label)
    data, labels = zip(*batch)
    
    # 找到最长的序列
    max_length = max([s.shape[1] for s in data])  # 假设s的形状为[2, seq_len]
    
    data = [torch.tensor(s) for s in data]
    # 填充序列
    padded_data = [F.pad(s, (0, max_length - s.shape[1])) for s in data]
    
    # 将填充后的数据转换为张量
    padded_data = torch.stack(padded_data)
    
    return padded_data, torch.tensor(labels)

