# 10/31　最新　モデルを学習させる用のコード　l
import csv
import os
import math
from re import M
import numpy as np
import torch
import torch.nn.functional
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
#from torchmetrics.functional import r2_score
import sys
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
#from torch.optim import LBFGS 
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import flwr as fl
import data_loading
from sklearn.utils import shuffle
from data_set import base, load

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self,):
        #  親クラスnn.Modulesの初期化呼び出し
        super().__init__()

        # 出力層の定義
        self.l1 = nn.Linear(16, 30)
        self.l2 = nn.Linear(30, 1)
        
    # 予測関数の定義
    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        #x = x.unsqueeze(1) 
        return x

def data_load(all_data, test_data):
    df = pd.read_csv(all_data)
    X = df.drop(["time","accumulated_time", 'ave_move_pace','name'], axis = 1)#axisって何？
    Y = df["time"]
    train_rate = 0.75
    val_rate = 0.15
    test_rate = 0.10
    
    #標準化処理
    scaler_train = StandardScaler()
    X = scaler_train.fit_transform(X)
    mean = scaler_train.mean_
    var = scaler_train.var_
    std = np.sqrt(var)

    X = torch.from_numpy(X).float()
    Y = torch.tensor(Y.values)
    Y = Y.to(device)
    standard = [mean,var,std]

    #訓練データとテストデータに分割
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=train_rate, random_state = 0)
    X_val, X_test, Y_val, Y_test = train_test_split(X, Y, train_size=val_rate / (val_rate + test_rate), random_state = 0)

    trainset = TensorDataset(X_train, Y_train)
    validset = TensorDataset(X_val, Y_val)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(validset, batch_size=32, shuffle=True), X_train, X_val, X_test, Y_train, Y_val, Y_test


def train(net, trainloader,optimizer, epochs):
    loss_func = torch.nn.functional.cross_entropy 
    #定義
    net.train()
    net.to(device)
    criterion = nn.MSELoss()
    
    # 学習
    for epoch in range(epochs):
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            # Zero the gradients
            optimizer.zero_grad()
            # Perform forward pass
            outputs = net(inputs)
            # Compute loss
            loss = criterion(outputs, targets)
            # Perform backward pass
            loss.backward()
            optimizer.step()

        #torch.save(net.state_dict(), 'model.pth')

def test(net, testloader):
    #定義
    net.eval()
    net.to(device)
    MSE_criterion = nn.MSELoss()
    #Test
    test_loss = 0
    epoch = 0
    correct = 0
    mae_loss = 0
    mse_loss = 0
    rmse_loss = 0
    loss = 0
    
    with torch.no_grad():
        pred_y = torch.Tensor()
        for i, data in enumerate(testloader, 0):
            inputs, targets = data
            #inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            outputs = net(inputs)
            MSE_loss = MSE_criterion(outputs, targets)
            loss += MSE_loss

            #平均絶対誤差
            mae = mean_absolute_error(targets, outputs)
            mae_loss += mae
            #平均２乗誤差
            mse= mean_squared_error(targets, outputs)
            mse_loss += mse
            #二乗平均平方根誤差
            rmse = np.sqrt(mean_squared_error(targets, outputs))
            rmse_loss+= rmse
            #print('\nTest loss (avg)', loss)
            #pred = outputs.argmax(dim=1, keepdim=True)
            epoch= epoch+1

    return  loss, mae_loss/epoch, mse_loss/epoch, rmse_loss/epoch

# インスタンス生成
def load_model():
    return Net().to(device)

def model_to_parameters(model):
    from flwr.common.parameter import ndarrays_to_parameters

    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(ndarrays)
    print("Extracted model parameters!")
    return parameters

#trainloader, testloader = load_data()
if __name__ == "__main__":
    net = load_model()
    criterion = nn.MSELoss()
    loss_func = torch.nn.functional.cross_entropy
    optimizer = torch.optim.Adam(net.parameters(),lr=0.001)

    folders = os.listdir('data_of_client_folders')
    merged_data = pd.DataFrame()
    for folder in folders:
        # データを保持する空のDataFrameを作成
        # 選択されたフォルダーのパス
        input_folder = os.path.join('data_of_client_folders', folder)
        # フォルダ内の各ファイルに対して処理
        for file in os.listdir(input_folder):
            file_path = "data_of_client_folders/"+folder+"/"+file
            data = pd.read_csv(file_path)             
            merged_data = pd.concat([merged_data, data], ignore_index=True)
          # マージされたデータをCSVファイルとして保存
    csvdata = "garbage/garbage.csv"
    merged_data.to_csv(csvdata, index=False)
    
    X,Y, df = base(csvdata, csv_data = 'csv_data/'+folder+'.csv')

    train_rate = 0.75
    val_rate = 0.15
    test_rate = 0.10
    
    #標準化処理
    scaler_train = StandardScaler()
    X = scaler_train.fit_transform(X)
    mean = scaler_train.mean_
    var = scaler_train.var_
    std = np.sqrt(var)

    X = torch.from_numpy(X).float()
    Y = torch.tensor(Y.values)
    Y = Y.to(device)
    standard = [mean,var,std]

    #訓練データとテストデータに分割
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=train_rate, random_state = 0)
    X_val, X_test, Y_val, Y_test = train_test_split(X, Y, train_size=val_rate / (val_rate + test_rate), random_state = 0)

    trainset = TensorDataset(X_train, Y_train)
    validset = TensorDataset(X_val, Y_val)
    testset = TensorDataset(X_test, Y_test)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)  
    valloader = DataLoader(validset, batch_size=16, shuffle=True)
    testloader = DataLoader(testset, batch_size=16, shuffle=True)
    n_epochs = 500
    for epoch in range(n_epochs):
        #train 
        train(net, trainloader,optimizer, epochs=1)
        torch.save(net.state_dict(), 'model.pth')
        #val loss
        loss, mae_loss, mse_loss, rmse_loss = test(net, valloader) 

        #if (epoch+1) % 100 == 0:
        #print(f"epochs{epoch+1:.0f},mae_Loss:{mae_loss:.5f}, mse_oss:{mse_loss:.5f}, rmse_loss:{rmse_loss:.5f}")
        print(f"{epoch+1:.0f},{mae_loss:.5f},{mse_loss:.5f},{rmse_loss:.5f}")

    #test loss
    loss, mae_loss, mse_loss, rmse_loss = test(net, testloader)
    print(f"testloss : mae_Loss:{mae_loss:.5f}, mse_oss:{mse_loss:.5f}, rmse_loss:{rmse_loss:.5f}")

