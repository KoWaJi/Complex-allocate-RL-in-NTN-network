import torch
from networkmodel import CNNAugmentedLSTM, LSTMNet
from data_prepare import CustomDataset, LoadAndCreate_data, collate_fn
from train import train_model
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import os
import pickle
from torch.utils.tensorboard import SummaryWriter

def main(args):
    device = torch.device('cuda')
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    
    batch_size = args.batch_size
    input_size = args.input_size
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    dropout_rate = args.dropout_rate
    epochs = args.epochs
    lr = args.lr
    model_type = args.model_type

    DataSet_path = 'channel prediction/datasets.pkl'
    ChannelData_path = 'channel prediction/CSI_data/Channel.mat'
    AngelData_path = 'channel prediction/CSI_data/angel.mat'

    if os.path.exists(DataSet_path):
        print("datasets exist. Proceeding to load data...")
        with open(DataSet_path, 'rb') as f:
            datasets = pickle.load(f)

        # 从字典中取回四个列表
        train_features = datasets['train_features']
        train_targets = datasets['train_targets']
 
    else:
        print("testdata do not exist. Proceeding to creat datasets...")

        train_features, train_targets, _, _ = LoadAndCreate_data(ChannelData_path, AngelData_path)
        
        # 创建训练集和自定义数据集实例
    train_dataset = CustomDataset(train_features, train_targets)

        # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size, collate_fn=collate_fn, shuffle=True)    
    

    # 实例化模型
    #model = LSTMNet(input_size=1, hidden_size=256, output_size=1)

    # 实例化模型
    if model_type == 'CNN+LSTM':
        model = CNNAugmentedLSTM(input_size, hidden_size, num_layers, dropout_rate)
    else:
        model = LSTMNet(input_size=2, hidden_size=64, output_size=1, num_layers=2, dropout_rate=0.25)
    model  = model.to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, epochs, device, model_type)
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for channel prediction")
    parser.add_argument("--batch_size", type=int, default=int(100), help=" Batch size of dataset")
    parser.add_argument("--input_size", type=int, default=8, help="Input size of LSTM layer")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size of LSTM layer")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM")
    parser.add_argument("--dropout_rate", type=float, default=0.25, help="Dropout rate of LSTM")
    parser.add_argument("--lr", type = float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=150, help="Epoch of the training process")
    parser.add_argument("--model_type", type=str, default="LSTM", help="Decide which model will be used")
    args = parser.parse_args()
    main(args)