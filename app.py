import pandas as pd
from pandas import Series, DataFrame
import numpy as np
# from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from tensorflow.keras import regularizers
# from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import load_model
def readCSV(x):
    train = pd.read_csv(x,header=None)
    #去掉nan的值
    # train = train.fillna(144)
    return train
def normalize(train):
    train_norm = train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return train_norm
def denormalize(train):
    denorm = train.apply(lambda x: x*(np.max(origin_train.iloc[:,0])-np.min(origin_train.iloc[:,0]))+np.min(origin_train.iloc[:,0]))
    return denorm
def buildModel(shape):
    model = Sequential()
    # model.add(GRU(30,input_length=shape[1], input_dim=shape[2],return_sequences=True,bias_regularizer=regularizers.l2(1e-4)))
    # model.add(GRU(30,input_length=shape[1], input_dim=shape[2],return_sequences=True))
    model.add(Dropout(0.1))
    model.add(GRU(50,return_sequences=True))
    model.add(Dropout(0.1))
    model.add(GRU(70,return_sequences=True))
    model.add(Dropout(0.1))
    model.add(GRU(100))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def buildTrain(train, past=30, future=1):
     X_train, Y_train ,last_dataX= [], [], []
     for i in range(train.shape[0]-future-past):
         X_train.append(np.array(train.iloc[i:i+past]))
         Y_train.append(np.array(train.iloc[i+past:i+past+future][0]))
         if i == train.shape[0]-future-past-1:
             last_dataX.append(train.iloc[i+1:i+past+future])
     return np.array(X_train), np.array(Y_train), np.array(last_dataX)

def buildTestX(test):
    x_test = []
    x_test.append(np.array(test.iloc[200:230]))
    return np.array(x_test)
def buildTestY(test):
    y_test = []
    y_test.append(np.array(test.iloc[230:231][0]))
    return np.array(y_test)
def shuffle(X,Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]
def splitData(X,Y,rate):
    X_train = X[int(X.shape[0]*rate):]
    Y_train = Y[int(Y.shape[0]*rate):]
    X_val = X[:int(X.shape[0]*rate)]
    Y_val = Y[:int(Y.shape[0]*rate)]
    return X_train, Y_train, X_val, Y_val
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--training',
                        default='training.csv',
                        help='input training data file name')
    parser.add_argument('--testing',
                        default='testing.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    #============================訓練=========================== 
    train = readCSV(args.training)
    train_norm = normalize(train)
    X_train, Y_train ,Last_dataX= buildTrain(train_norm,30,1)
    X_train, Y_train = shuffle(X_train, Y_train)
    X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)
    model = buildModel(X_train.shape)
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    model.fit(X_train, Y_train, epochs=1000, batch_size=32, validation_data=(X_val, Y_val), callbacks=[callback])
    # model.save('test.h5')
    #=============取出最後三十天==================
    # model = load_model('test.h5')
    testing =readCSV(args.testing)
    origin_train = train
    y_test = []
    hat_action = []
    have = 0
    
    y_pre = []
    #=============================ACTION==============================
    for i in range(len(testing.iloc[:,0])):
        print(hat_action)
        y_test.append(testing.iloc[i][0])
        # pred = pd.DataFrame(pred)
        
        train = train.append(testing.iloc[i])#將今天資料加入training data
        train_norn = normalize(train)
        X_train,Y_train,Last_dataX = buildTrain(train_norn,30,1)#提出最後三十筆，包含新的第一天
        pred = model.predict(Last_dataX)#得到明天開盤預測
        pred = pd.DataFrame(pred)
        a = denormalize(pred)
        y_pre.append(a.iloc[0][0])
        print(np.sqrt((a-y_test[i])**2))
        if i < len(testing.iloc[:,0])-1:
            if a[0][0]>testing.iloc[i][0]:#明天>今天
                if have == 1:
                    hat_action.append(0)
                    have = 1
                    continue
                if have == 0:
                    hat_action.append(-1)
                    have = -1
                    continue
                if have == -1:
                    hat_action.append(0)
                    have = -1
                    continue
            if a[0][0]<testing.iloc[i][0]:
                if have == 1:
                    hat_action.append(-1)
                    have = 0
                    continue
                if have == 0:
                    # print("fuck")
                    hat_action.append(1)
                    # print(type(hat_action))
                    have = 1
                    continue
                if have == -1:
                    hat_action.append(1)
                    have = 0
                    continue
            if a[0][0] == testing.iloc[i][0]:
                hat_action.append(0)
                have = have
                continue
#     plt.xlabel('20 days', fontsize = 16)  
# # 繪圖並設定線條顏色、寬度、圖例
#     line1, = plt.plot(y_pre, color = 'red', linewidth = 3, label = 'predict')             
#     line2, = plt.plot(y_test, color = 'blue', linewidth = 3, label = 'ground true')
#     plt.legend(handles = [line1, line2])
    # plt.savefig('Fe_r_plot.svg')                              # 儲存圖片
    # plt.savefig('Fe_r_plot.png')
    # plt.show()  
    #============================================================================
    hat_action = pd.DataFrame(hat_action,columns = ['Action'])
    # hat.rename(index = {"0":'Action'})
    hat_action.to_csv(args.output,index = False)