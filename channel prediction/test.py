import torch
import pickle
import argparse
from data_prepare import CustomDataset, collate_fn
from networkmodel import CNNAugmentedLSTM, LSTMNet
from torch.utils.data import DataLoader

def test_model(args):
    # 加载模型
    batch_size = args.batch_size
    input_size = args.input_size
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    dropout_rate = args.dropout_rate
    model_type = args.model_type

    device = torch.device('cuda')
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    
    with open('channel prediction/datasets.pkl', 'rb') as f:
            datasets = pickle.load(f)

    model_path = 'channel prediction/lstm_network/model_epoch_150.pth'

    test_features = datasets['test_features']
    test_targets = datasets['test_targets']
  
    test_dataset = CustomDataset(test_features, test_targets)
    test_loader = DataLoader(test_dataset, batch_size, collate_fn=collate_fn, shuffle=False)
    if model_type == 'CNN+LSTM':
        model = CNNAugmentedLSTM(input_size, hidden_size, num_layers, dropout_rate)
    else:
        model = LSTMNet(input_size=2, hidden_size=64, output_size=1, num_layers=2, dropout_rate=0.25)
    model  = model.to(device)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的模型权重
    model.to(device)
    model.eval()  # 切换到评估模式
    with torch.no_grad():  # 在测试过程中不计算梯度
        nmse = 0.0
        for data in test_loader:
            sequences, targets = data
            sequences = sequences.to(device)
            targets = targets.to(device)
            outputs = model(sequences)
            outputs = torch.squeeze(outputs) 
            assert torch.all(targets != 0), "Target tensor contains zero values, which would lead to division by zero."
            error = ((outputs - targets) ** 2) / (targets ** 2)
            mean_error = torch.mean(error)  #计算每个batch中的NMSE（真的是NMSE吗？）
            nmse += mean_error.item()

    # 计算归一化均方误差 (NMSE)
    loss = nmse / len(test_loader)
    print(f'Normalized Mean Squared Error (NMSE) on the test set: {loss:.8f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for channel prediction testing")
    parser.add_argument("--batch_size", type=int, default=int(100), help=" Batch size of dataset")
    parser.add_argument("--input_size", type=int, default=8, help="Input size of LSTM layer")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size of LSTM layer")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM")
    parser.add_argument("--dropout_rate", type=float, default=0.25, help="Dropout rate of LSTM")
    parser.add_argument("--model_type", type=str, default="LSTM", help="Decide which model will be used")
    args = parser.parse_args()
    test_model(args)
