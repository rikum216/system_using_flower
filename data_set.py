import csv
from re import M
import numpy as np
import torch
import torch.nn.functional
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import os

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
    #print(count)
    #print(l)
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

    train_rate = 0.9
  
    
    #訓練データとテストデータに分割
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_rate, random_state = 0)

    trainset = TensorDataset(X_train, Y_train)
    testset = TensorDataset(X_test, Y_test)

    return trainset,testset, standard_list

def merge(folder_path,output_file):#選択したクライアントのフォルダー内のファイルを結合する関数
  # まとめたいCSVファイルのリスト
  csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

  # データを保持する空のDataFrameを作成
  merged_data = pd.DataFrame()

  # 各CSVファイルを読み込んでマージ
  for file in csv_files:
      file_path = os.path.join(folder_path, file)
      data = pd.read_csv(file_path)
      merged_data = pd.concat([merged_data, data], ignore_index=True)

  # マージされたデータをCSVファイルとして保存
  merged_data.to_csv(output_file, index=False)

def prepare_dataset(num_partitions: int,
                    batch_size: int,
                    val_ratio: float = 0.1):
    #clientの選択
    # フォルダーの名前とインデックスを表示
    folders = os.listdir('data_of_client_folders')
    for i, folder in enumerate(folders, 1):
        print(f"{i}: {folder}")

    # 使用者にフォルダーを選択させる
    selected_index = int(input("クライアントを選択してください（インデックスをカンマ区切りで入力）: "))-1
    selected_folders = folders[selected_index]
    #print(selected_folders)

    #選択したクライアントのフォルダー内のファイルを結合
    folder_path='data_of_client_folders/'+selected_folders
    client_file = 'data_of_client/'+selected_folders+'.csv'
    merge(folder_path,client_file)

    trainloader, valloader, testloader, standard_list = load(all_data = client_file, csv_data = 'csv_data/'+selected_folders +'.csv')

    return trainloader, valloader, testloader


def prepare_dataset_beta(num_partitions: int,
                    batch_size: int,
                    val_ratio: float = 0.1):
    
    folders = os.listdir('data_of_client_folders')

    if folders == num_partitions:
        print("num_clientの数が正しくない")
        exit()
    trainsets = []
    for folder in folders:
        # データを保持する空のDataFrameを作成

        merged_data = pd.DataFrame()
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

        #標準化処理をどうするか問題！！！！！！！！！！！！！！！、いったんしない方向で考える
        X,Y,df = base(csvdata, csv_data = 'csv_data/'+folder+'.csv')
        X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        Y = torch.tensor(Y.to_numpy(), dtype=torch.float64)
        #訓練データとテストデータに分割
        train_rate = 0.9
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_rate, random_state = 0)
        trainset = TensorDataset(X_train, Y_train)
        testset = TensorDataset(X_test, Y_test)
        trainsets.append(trainset)

    

    #dataloaderの作成
    trainloaders = []
    valloaders = []

    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio*num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size,shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size,shuffle=False, num_workers=2))

        #for batch in DataLoader(for_train, batch_size=batch_size,shuffle=True, num_workers=2):
            #for tensor in batch:
                #print(tensor.dtype)  # Tensorのdtypeを表示
               # break  # 最初のバッチのみを調べるため、ループを終了

        #for batch in DataLoader(for_val, batch_size=batch_size,shuffle=True, num_workers=2):
            #for tensor in batch:
                #print(tensor.dtype)  # Tensorのdtypeを表示
                #break  # 最初のバッチのみを調べるため、ループを終了
    
    testloader = DataLoader(testset, batch_size=16)
    #for batch in testloader:
           # for tensor in batch:
                #print(tensor.dtype)  # Tensorのdtypeを表示
                #break  # 最初のバッチのみを調べるため、ループを終了


    return trainloaders, valloaders, testloader

# データの理解と前処理、データの可視化、データの分析の処理を行うための関数を定義
def process_alldata():
    # フォルダーの名前とインデックスを表示
    folders = os.listdir('data_of_client_folders')

    # データを保持する空のDataFrameを作成
    merged_data = pd.DataFrame()

    for folder in folders:
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
    
    return csvdata

def prepare_dataset_alldata(num_partitions: int,
                    batch_size: int,
                    val_ratio: float = 0.1):
    
    csvdata = process_alldata()

    trainset, testset, standard_list = load(all_data = csvdata, csv_data = 'garbage/all_data.csv')

    #trainsetを 'num_partitions' 個のtrainsetに分割する
    num_samples = len(trainset)
    num_answer, remainder = divmod(num_samples, num_partitions)
    partition_len = [num_answer] * num_partitions
    # 余りを使わず、最後の部分の長さを調整する
    for i in range(remainder):
        partition_len[i] += 1

    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))
    print("---------------------------------")
    print("trainsets",type(trainsets),trainsets)
    print("---------------------------------")


    #dataloaderの作成
    trainloaders = []
    valloaders = []

    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio*num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size,shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size,shuffle=False, num_workers=2))
    
    testloader = DataLoader(testset, batch_size=128)

    for batch in DataLoader(for_train, batch_size=batch_size,shuffle=True, num_workers=2):
        for tensor in batch:
                print(tensor.dtype)  # Tensorのdtypeを表示
                break  # 最初のバッチのみを調べるため、ループを終了

    for batch in DataLoader(for_val, batch_size=batch_size,shuffle=True, num_workers=2):
        for tensor in batch:
                print(tensor.dtype)  # Tensorのdtypeを表示
                break  # 最初のバッチのみを調べるため、ループを終了
    for batch in testloader:
            for tensor in batch:
                print(tensor.dtype)  # Tensorのdtypeを表示
                break  # 最初のバッチのみを調べるため、ループを終了
    print("trainloaders, valloaders, testloader", type(trainloaders), type(valloaders), type(testloader))

    return trainloaders, valloaders, testloader