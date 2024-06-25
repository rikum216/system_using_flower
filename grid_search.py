from re import M
import torch
import torch.nn.functional
import torch.utils.data
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import data_loading
import warnings
from data_set import datasets_gridsearch
warnings.simplefilter('error', category=RuntimeWarning) 


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

def test(net,data,X,Y, max, min, dynamics_number, filename):

    prediction_time = 10000
    best_time = 1000000
    best_dynamics_number = 0
    number = 0
    best_dynamics = X[dynamics_number]
    # 空のDataFrameを作成
    df = pd.DataFrame()
    #データを変更しながら、よいものを探していく
    #値を変更していくプログラム(3kmを1kmずつに分解するため、どうやって出すべきか、できれば足し合わせたくない)
    #値の条件があるから、最初にマイナスした状態から初めて、一定の上限値まで(純粋にこれまでのデータ等の最大値から最小値までを計算していけばよいのかも)

    for dynamics in range(int(min), int(max)):
        if dynamics_number == 8:
            dynamics = float(dynamics)*0.1
            X[dynamics_number] = dynamics

        elif dynamics_number == 7:
            dynamics = float(dynamics)*0.01
            X[dynamics_number] = dynamics

        elif dynamics_number == 3 or dynamics_number == 4:
            X[dynamics_number] = dynamics

        inputs, targets = X, Y
        inputs, targets = inputs.float(), targets.float()
        #targets = targets.reshape((targets.shape[0], 1))
        outputs = net(inputs)
        #print('\ndynamics:{} targets:{} output: {}'.format(dynamics, targets, outputs))
        prediction_time = outputs
        prediction_time_numpy = prediction_time.detach().numpy().item()
        # DataFrameに追加
        if dynamics_number == 8 or dynamics_number==7:
            csv_data = {'dynamics': [dynamics*100], 'prediction_time': [prediction_time_numpy]}
        else:
            csv_data = {'dynamics': [dynamics], 'prediction_time': [prediction_time_numpy]}                 
        # listを作成
        df = pd.concat([df, pd.DataFrame(csv_data)], ignore_index=True)
        if best_time > prediction_time:
            best_time = prediction_time
            best_dynamics = X[dynamics_number]
            #このbest_dynamics_numberって何のための変数？
            best_dynamics_number = number
            number = number + 1
            df.to_csv(filename, index=False)
    return best_time, best_dynamics, best_dynamics_number

def GCT_balance_test(net, X, Y, max, min, dynamics_number, mean, std, filename):
        input_data = TensorDataset(X, Y)
        data_loader = DataLoader(input_data, batch_size=1)
        prediction_time = 10000
        best_time = 1000000
        best_dynamics_number = 0
        number = 0
        #print("mean", mean)
        #print("\nstd", std)
        best_dynamics = X[0][dynamics_number]
        # 空のDataFrameを作成
        df = pd.DataFrame()
        #データを変更しながら、よいものを探していく
        #値を変更していくプログラム(3kmを1kmずつに分解するため、どうやって出すべきか、できれば足し合わせたくない)
        #値の条件があるから、最初にマイナスした状態から初めて、一定の上限値まで(純粋にこれまでのデータ等の最大値から最小値までを計算していけばよいのかも)
        for dynamics in range(int(min), int(max)):#いったん適当に範囲を決めて変更してみる
            dynamics = float(dynamics)
            #mean[dynamics_number] =  mean[dynamics_number]*0.01
            X[0][dynamics_number] = dynamics
            if dynamics_number==5:
                dynamics2 = float(100-dynamics)
                X[0][dynamics_number+1] = dynamics2
            elif dynamics_number==6:
                dynamics2 = float(100-dynamics)
                X[0][dynamics_number-1] = dynamics2

            with torch.no_grad():                
                for e, data in enumerate(data_loader, 0):
                    inputs, targets = data
                    #標準化
                    #print("\ninputs",inputs)
                    if dynamics_number==5:
                        inputs[0][dynamics_number] = (inputs[0][dynamics_number] - mean[dynamics_number])/std[dynamics_number]
                        inputs[0][dynamics_number+1] = (inputs[0][dynamics_number+1] - mean[dynamics_number+1])/std[dynamics_number+1]
                    elif dynamics_number==6:
                        inputs[0][dynamics_number] = (inputs[0][dynamics_number] - mean[dynamics_number])/std[dynamics_number]
                        inputs[0][dynamics_number-1] = (inputs[0][dynamics_number-1] - mean[dynamics_number-1])/std[dynamics_number-1]
                    #print("\ninputs標準化",inputs)
                    inputs, targets = inputs.float(), targets.float()
                    targets = targets.reshape((targets.shape[0], 1))
                    outputs = net(inputs)
                    #print('\ndynamics1:{} dyamics2:{} targets:{} output: {}'.format(dynamics, dynamics2, targets, outputs))
                    prediction_time = outputs
                    prediction_time_numpy = prediction_time.numpy().item()
                    # DataFrameに追加するデータ
                    data = {'dynamics': [dynamics], 'prediction_time': [prediction_time_numpy]}
                    # DataFrameを作成
                    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

                    # csvファイルに追記モードで書き込み

                if best_time > prediction_time:
                    best_time = prediction_time
                    best_dynamics = X[0][dynamics_number]
                    best_dynamics_number = number
            number = number + 1
        # csvファイルに追記モードで書き込み
        df.to_csv(filename, index=False)
        #if dynamics_number==5:
            #print("右")
        #elif dynamics_number==6:
            #print("左")
        return net, best_time, best_dynamics, best_dynamics_number

def main():
    #datasets.pyからデータをとってくる
    datasets, X, Y = datasets_gridsearch()

    #保存したモデルをよび起こす
    model = Net()
    model.load_state_dict(torch.load('model_fl.pth'))

    #どのランニングダイナミクスを変更するのか選択（左右バランスは除いている）
    dynamics_list = np.array([["pitch", 3], ["ground_time",4],["stride",7],["vertical_motion",8]])
    dynamics_number = 4
    
    #---------------for文で回して　それぞれのダイナミクスを変更させたい--------------------------------
    #for 一人一人にデータを区分けする
    for dataset_number in range(len(datasets)):
        dataset = datasets[dataset_number]
        #for 一つ一つのデータに分ける
        for i in range(len(dataset)):
            data = dataset[i]
            x = data[0].clone()
            y = data[1].clone()
            print(f"元のデータ time:{y:.3f},dynamics:{x[dynamics_number]:.3f}")
            #最速タイム＋最適な説明変数＝グリッドサーチする関数（モデル、データセット）(pitch)
            max = 300
            min = 0
            #test(model, data, max, min, dynamics_number, filename)
            file_name = 'results_grid_search/'+str(dynamics_number)+'/'+ str(dataset_number)+'-'+ str(i)+ '.csv'
            best_time, best_dynamics, best_dynamics_number = test(model, data, x, y, max, min, dynamics_number, file_name)
            #たぶんdatasetsがグローバル変数になっている　下の出力と上の出力とはことなってくる
            print(f"元のデータ time:{data[1]:.0f},dynamics:{data[0][dynamics_number]:.0f}")
            print(f"最適データ time:",best_time,"dynamics:",best_dynamics)

            #ある人のあるデータの最適な説明変数と目的変数と元の説明、目的変数を保存（csvファイル？）
        #個人ごとに一つのファイルにまとめる

    #全員分のファイルを求める

    #ファイルを使うかどう変わらなないけど、結果をmain.pyに送って、ログとして記録
    return best_time, best_dynamics


if __name__ == "__main__":
    best_time, best_dynamics = main()



