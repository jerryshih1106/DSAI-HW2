import pandas as pd
from pandas import Series, DataFrame
import numpy as np
# from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
# from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import load_model
def readCSV(x):
    train = pd.read_csv(x,header=None)
    #去掉nan的值
    train = train.fillna(144)
    return train
def normalize(train):
    train_norm = train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return train_norm
def denormalize(train):
    denorm = train.apply(lambda x: x*(np.max(datatest.iloc[:,0])-np.min(datatest.iloc[:,0]))+np.min(datatest.iloc[:,0]))
    return denorm
def buildManyToManyModel(shape):
    
    model = Sequential()
    model.add(GRU(70,return_sequences=True,input_length=shape[1], input_dim=shape[2]))
    model.add(Dropout(0.1))
    model.add(GRU(100,return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(20))
    model.compile(loss='mse', optimizer='adam')

    print ('model compiled')

    return model

def buildTrain(train, past=30, future=20):
     X_train, Y_train = [], []
     for i in range(train.shape[0]-future-past):
         X_train.append(np.array(train.iloc[i:i+past]))
         Y_train.append(np.array(train.iloc[i+past:i+past+future][0]))
     return np.array(X_train), np.array(Y_train)

def buildTestX(test):
    x_test = []
    x_test.append(np.array(test.iloc[200:230]))
    return np.array(x_test)
def buildTestY(test):
    y_test = []
    y_test.append(np.array(test.iloc[230:250][0]))
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

train = readCSV(x = "training.csv")
# train = pd.read_csv("training.csv",header=None)
train_norm = normalize(train)

X_train, Y_train = buildTrain(train_norm,30,20)

X_train, Y_train = shuffle(X_train, Y_train)


X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)


# model = buildManyToManyModel(X_train.shape)
# callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
# model.fit(X_train, Y_train, epochs=1000, batch_size=32, validation_data=(X_val, Y_val), callbacks=[callback])
# model.save('my_model.h5')

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


    # datatest = pd.read_csv(args.training)
    model = load_model('my_model.h5')

    datatest = readCSV(x = "testing.csv")
    # datatest_Aug = augFeatures(datatest)
    datatest_norm = normalize(datatest)
#原始測試資料集
    X_test = buildTestX(datatest_norm)

    predicted_data = model.predict(X_test)
    predicted_data = pd.DataFrame(np.concatenate(predicted_data))#(1,20)--->(20,1)
    denorn_pre = denormalize(predicted_data)

    #畫圖===============================================================================
    # plt.xlabel('2021/03/09~2021/03/15', fontsize = 16)  
    # Y_test = np.reshape(Y_test,(20,1)) 
    # plt.xticks(fontsize = 12)                                 # 設定坐標軸數字格式
    # plt.yticks(fontsize = 12)
    # plt.grid(color = 'red', linestyle = '--', linewidth = 1)  # 設定格線顏色、種類、寬度
    # plt.ylim(2000, 5000)                                          # 設定y軸繪圖範圍
# 繪圖並設定線條顏色、寬度、圖例
    # line1, = plt.plot(y_hat, color = 'red', linewidth = 3, label = 'predict')             
    # line2, = plt.plot(Y_test, color = 'blue', linewidth = 3, label = 'ground true')
    # plt.legend(handles = [line1, line2])
    # plt.savefig('Fe_r_plot.svg')                              # 儲存圖片
    # plt.savefig('Fe_r_plot.png')
    # plt.show()  
    #================================================================================
    # Y_test = pd.DataFrame(np.concatenate(Y_test))
    # realdata = denormalize(Y_test)
    # print(rmse(y_hat,Y_test))
    # model.train(df_training)
    # df_result = model.predict(n_step=7)
    # y_hat1 = DataFrame(y_hat,index = ['20210323','20210324','20210325','20210326','20210327','20210328','20210329'],columns=['0'])
    # y_hat.index = Series(['2021-03-23','2021-03-24','2021-03-25','2021-03-26','2021-03-27','2021-03-28','2021-03-29'])
    # a
    list_pre = denorn_pre.values.tolist()
    # list_pre = list(denorn_pre)
    list_pre = np.reshape(list_pre,(20))
    hat = []
    # hat = list(hat)
    for i in range(len(list_pre)-1):
        # if i == 0:
        #     if list_pre[i] == np.min(denorn_pre.iloc[:,0]):
        #         hat.append(1)
        #     if list_pre[i] == np.max(denorn_pre.iloc[:,0]):
        #         hat.append(-1)
        #     else:
        #         hat.append(0)
        # if i < len(list_pre):
        # print(np.min(denorn_pre.iloc[:,0]))
        # print('第i次:',i)
        if list_pre[i+1] == np.min(denorn_pre.iloc[:,0]):
            hat.append(1)
        if list_pre[i+1] == np.max(denorn_pre.iloc[:,0]):
            # print(i)
            hat.append(-1)
        if list_pre[i+1] != np.min(denorn_pre.iloc[:,0]):
            if list_pre[i+1] != np.max(denorn_pre.iloc[:,0]):
                hat.append(0)
        # print(hat)

        # if i == len(list_pre):

    hat = pd.DataFrame(hat,columns = ['Action'])
    # hat.rename(index = {"0":'Action'})
    hat.to_csv(args.output,index = False)