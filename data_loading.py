import csv
from re import M
import numpy as np
import torch
import torch.nn.functional
import torch.utils.data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def base(all_data, csv_data):
        #変数
    i=0
    k=0 #pitchとかの箱に何もなかった時の検出に仕様
    l=0 #pitchとかの箱に何もなかったとき、その分だけ、iの値をずらさないといけないから、それの調整に使用
    lost_count = 0 #何も入ってない列が何個あったかを格納して、pitchとstep以外（最初にやってるやつ以外）のやつのデータの箱を調整するために使用
    time = []
    accumulated_time = []#累積時間
    pace = []
    pitch = []
    gro_time = []
    right_GCT = []
    left_GCT = []
    step = []
    ver_move = []
    ver_move_ratio = []
    run_weight = []
    heart_rate = []
    max_heart_rate = []
    dis_a = []
    data_color = []
    lap = []
    calorie = []
    blanck=[]
    high_pace = []#
    high_pitch = []#
    move_time = []#
    ave_move_pace = []#
    hight = []
    weight = []
    vo2max = []
    #データの読み込み
    df = pd.read_csv(all_data,encoding='utf-8', sep=",")
    df.head()
    count = 0
    with open(all_data,encoding='utf-8') as f:#ここでいくつデータがあるかをカウントしてる
        for line in f:
            count += 1
            count_copy = count
    f.close()
    #データを変換
    while i<count-1:
        dis_a = df['距離'][i]
        if dis_a >= 0.2:
            lap.append(float(df['ラップ数'][i]))
            time_min = float(df['タイム'][i][1])
            time_sec = float(df['タイム'][i][3])
            time_sec1 = float(df['タイム'][i][4])
            time_sec2 = float(df['タイム'][i][6])
            time.append(time_min*60 + time_sec*10 + time_sec1 + time_sec2*0.1)
            if df["累積時間"][i][1] == ":":
                accumulated_time_min = float(df["累積時間"][i][0])
                accumulated_time_ten_sec = float(df["累積時間"][i][2])
                accumulated_time_sec = float(df["累積時間"][i][3])
                accumulated_time.append(accumulated_time_min*60 + accumulated_time_ten_sec*10 + accumulated_time_sec)
            else:
                accumulated_time_ten_min = float(df["累積時間"][i][0])
                accumulated_time_min = float(df["累積時間"][i][1])
                accumulated_time_ten_sec = float(df["累積時間"][i][3])
                accumulated_time_sec = float(df["累積時間"][i][4])
                #if df["累積時間"][i][5] is None:
                    #accumulated_time_sec2 = float(df["累積時間"][i][6])
                    #accumulated_time.append(accumulated_time_ten_min*600 + accumulated_time_min*60 + accumulated_time_ten_sec*10 + accumulated_time_sec + accumulated_time_sec2*0.1)
                #else:
                    #accumulated_time.append(accumulated_time_ten_min*600 + accumulated_time_min*60 + accumulated_time_ten_sec*10 + accumulated_time_sec)
                accumulated_time.append(accumulated_time_ten_min*600 + accumulated_time_min*60 + accumulated_time_ten_sec*10 + accumulated_time_sec)
            if df['平均ペース'][i][2] == ":":
                pace_min = float(df['平均ペース'][i][0])
                pace_min1 = float(df['平均ペース'][i][1])
                pace_sec = float(df['平均ペース'][i][3])
                pace_sec1 = float(df['平均ペース'][i][4])
                pace.append(pace_min*600 + pace_min*60 + pace_sec*10 + pace_sec1)
                pitch.append(float(df['平均ピッチ'][i]))
            else:
                pace_min = float(df['平均ペース'][i][0])
                pace_sec = float(df['平均ペース'][i][2])
                pace_sec1 = float(df['平均ペース'][i][3])
                pace.append(pace_min*60 + pace_sec*10 + pace_sec1)
                pitch.append(float(df['平均ピッチ'][i]))

            gro_time.append(float(df['平均接地時間'][i]))

            if df['平均GCTバランス'][i][0] == "左":
                left_GCT10 = float(df['平均GCTバランス'][i][2])
                left_GCT1 = float(df['平均GCTバランス'][i][3])
                left_GCT01 = float(df['平均GCTバランス'][i][5])
                left_GCT.append(left_GCT10*10 + left_GCT1 + left_GCT01*0.1)
                right_GCT10 = float(df['平均GCTバランス'][i][10])
                right_GCT1 = float(df['平均GCTバランス'][i][11])
                right_GCT01 = float(df['平均GCTバランス'][i][13])
                right_GCT.append(right_GCT10*10 + right_GCT1 + right_GCT01*0.1)         
            else:
                left_GCT10 = float(df['平均GCTバランス'][i][0])
                left_GCT1 = float(df['平均GCTバランス'][i][1])
                left_GCT01 = float(df['平均GCTバランス'][i][3])
                left_GCT.append(left_GCT10*10 + left_GCT1 + left_GCT01*0.1)
                right_GCT10 = float(df['平均GCTバランス'][i][10])
                right_GCT1 = float(df['平均GCTバランス'][i][11])
                right_GCT01 = float(df['平均GCTバランス'][i][13])
                right_GCT.append(right_GCT10*10 + right_GCT1 + right_GCT01*0.1)
            step.append(float(df['平均歩幅'][i]))
            ver_move.append(float(df['平均上下動'][i]))
            ver_move_ratio.append(float(df['平均上下動比'][i]))
            #run_weight.append(float(df['ランニング強度'][i]))
            heart_rate.append(float(df['平均心拍数'][i]))
            max_heart_rate.append(float(df["最大心拍数"][i]))
            calorie.append(float(df['カロリー'][i]))
            high_pace_min = float(df['最高ペース'][i][0])
            high_pace_sec = float(df['最高ペース'][i][2])
            high_pace_sec1 = float(df['最高ペース'][i][3])
            high_pace.append(high_pace_min*60 + high_pace_sec*10 + high_pace_sec1)
            high_pitch.append(float(df['最高ピッチ'][i])) 
            ave_move_pace_min = float(df['平均移動ペース'][i][0])
            ave_move_pace_sec = float(df['平均移動ペース'][i][2])
            ave_move_pace_sec1 = float(df['平均移動ペース'][i][3])
            ave_move_pace.append(ave_move_pace_min*60 + ave_move_pace_sec*10 + ave_move_pace_sec1)
            hight.append(float(df['身長'][i]))
            weight.append(float(df['体重'][i]))
            #vo2max.append(float(df['VO2Max'][i]))

        else:
            k=1

        if k == 1:
            k=0
            l+=1
        i+=1

     #1~3kmのデータをまとめる
    csv_count = 0
    count = count_copy
    print(count)
    print(l)
    with open(csv_data,'w',newline="",encoding="utf-8") as ff:
        w = csv.writer(ff)
        w.writerow(['lap','time', 'pace',"accumulated_time",'ave_heart_rate','max_heart_rate','ave_pitch','ave_grond','right_GCT','left_GCT','ave_stride','ave_vertical_motion','ave_Vertical_movement_ratio','calorie','high_pace','high_pitch','ave_move_pace', 'height','weight'])
        while csv_count <= (count-l-2):    
            for i in range(16):
                if lap[csv_count]==i:
                    w.writerow([lap[csv_count],time[csv_count],pace[csv_count],accumulated_time[csv_count], heart_rate[csv_count],max_heart_rate[csv_count],pitch[csv_count],gro_time[csv_count],right_GCT[csv_count],left_GCT[csv_count],step[csv_count],ver_move[csv_count],ver_move_ratio[csv_count],calorie[csv_count],high_pace[csv_count],high_pitch[csv_count],ave_move_pace[csv_count], hight[csv_count],weight[csv_count]])
            csv_count += 1
    ff.close()
    df = pd.read_csv(csv_data)
    #目的変数と説明変数の作成
    #X = df.drop(["time","accumulated_time", 'ave_move_pace','ave_Vertical_movement_ratio'], axis = 1)#axisって何？
    X = df.drop(["time","accumulated_time", 'ave_move_pace'], axis = 1)#axisって何？
    Y = df["time"]
    return X, Y, csv_data

def load(all_data, csv_data):
    X, Y, df = base(all_data,csv_data)
    #標準化処理
    scaler_train = StandardScaler()
    X = scaler_train.fit_transform(X)
    mean = scaler_train.mean_
    var = scaler_train.var_
    std = np.sqrt(var)
    X = torch.from_numpy(X).float()
    Y = torch.tensor(Y.values) 
    Y = Y.to(device)
    standard_list = [mean,var,std]

    train_rate = 0.75
    val_rate = 0.15
    test_rate = 0.10
    
    #訓練データとテストデータに分割
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=train_rate, random_state = 0)
    X_val, X_test, Y_val, Y_test = train_test_split(X, Y, train_size=val_rate / (val_rate + test_rate), random_state = 0)

    trainset = TensorDataset(X_train, Y_train)
    validset = TensorDataset(X_val, Y_val)
    testset = TensorDataset(X_test, Y_test)

    return DataLoader(trainset, batch_size=32, shuffle=True),DataLoader(validset), DataLoader(testset), standard_list

def load_train(all_data, test_data):
    X, Y, all_data = base(all_data, test_data)

    train_rate = 0.80
    val_rate = 0.10
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

def load_test(all_data, test_data):
    X, Y, df = base(all_data, test_data)
    #標準化処理
    scaler_train = StandardScaler()
    X = scaler_train.fit_transform(X)
    mean = scaler_train.mean_
    var = scaler_train.var_
    std = np.sqrt(var)
    X = torch.from_numpy(X).float()
    Y = torch.tensor(Y.values) 
    Y = Y.to(device)
    standard_list = [mean,var,std]

    #訓練データとテストデータに分割
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,  random_state = 0)
    
    trainset = TensorDataset(X_train, Y_train)
    testset = TensorDataset(X_test, Y_test)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset), standard_list

def load2(all_data, test_data):
    X, Y, df = base(all_data, test_data)
    #標準化処理
    scaler_train = StandardScaler()
    X = scaler_train.fit_transform(X)
    mean = scaler_train.mean_
    var = scaler_train.var_
    std = np.sqrt(var)
    X = torch.from_numpy(X).float()
    Y = torch.tensor(Y.values) 
    Y = Y.to(device)
    standard_list = [mean,var,std]

    #訓練データとテストデータに分割
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,  random_state = 0)
    
    trainset = TensorDataset(X_train, Y_train)
    testset = TensorDataset(X_test, Y_test)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset), standard_list


if __name__ == '__main__':
    X, Y, standard_train, standard_test = load(all_data = 'personal_data/all_members_data1.csv', test_data = '3km_data_to_sdv.csv')

















