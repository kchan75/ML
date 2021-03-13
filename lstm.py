import time
import warnings
import numpy as np
from keras.layers import Dense, Activation, Dropout, LeakyReLU
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import rmsprop
from os import path
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import tensorflow as tf
from tensorflow.python.client import device_lib
from stock.services import investment_service
from statistics import mean, median, stdev
from stock.services import market_data_service

#전역변수
from setuptools.msvc import winreg
warnings.filterwarnings("ignore")
first_data = []
X_train = []
y_train = []
X_test = []
y_test = []
mean_data = []
stdev_data = []

def plot_results(last_date, true_data, predicted_data_1, predicted_data_2, window_size):
    period = w_size + ((w_size / 5) * 2)

    startDate = dt.datetime.strptime(last_date, '%Y.%m.%d') + dt.timedelta(((-1)*period) + 1)
    dateList = getBizDay(startDate, period)
    fDateList = getBizDay(str(dateList[-1]), period)

    dateList = list(map(str, dateList))
    fDateList = list(map(str, fDateList))

    fig_sec = plt.figure(facecolor='white', figsize=[10,8])

    b = plt.subplot(2,2,1)
    b.plot(np.array(dateList), np.array(true_data)[:,1], 'y', lw=1, label='real')
    b.plot(np.array(dateList), np.array(predicted_data_1)[:,1], 'b', lw=1)
    plt.gcf().autofmt_xdate()
    plt.title('GOLD')

    c = plt.subplot(2,2,2)
    c.plot(np.array(dateList), np.array(true_data)[:,2], 'y', lw=1)
    c.plot(np.array(dateList), np.array(predicted_data_1)[:,2], 'b', lw=1)
    plt.gcf().autofmt_xdate()
    plt.title('Dollar')


    a = plt.subplot(2,2,(3,4))
    a.plot(np.array(dateList), np.array(true_data)[:,0], 'y', lw=1)
    a.plot(np.array(dateList), np.array(predicted_data_1)[:,0], 'b', lw=1)
    a.plot(np.array(fDateList), np.array(predicted_data_2)[:,0], 'r', lw=1)
    plt.gcf().autofmt_xdate()
    plt.title('KOSPI')

    plt.savefig("./image/" + code + ".png", dpi=300)
    plt.show()

def load_data_db(code, seq_len, nomalize_window):

    sequence_length = seq_len
    r_data = []
    result = []
    first_data = []

    date_data = []
    p_data = []
    k_data = []
    f_data = []

    dataList = investment_service.getdata(code)

    for data in dataList:
        date_data.append(data['date'])

        #거래량 0 보정
        if nomalize_window:
            price = data['price'].replace(",","")
            korea = data['korea'].replace(",", "")
            foreign = data['foreign'].replace(",", "")

            if korea == '0':    korea = 1
            if foreign == '0':    foreign = 1

            p_data.append(price)
            k_data.append(korea)
            f_data.append(foreign)
        else:
            p_data.append(data['price'].replace(",",""))
            k_data.append(data['korea'].replace(",",""))
            f_data.append(data['foreign'].replace(",",""))

    for index in range(len(date_data) - sequence_length + 1):
        result.append([p_data[index:index + sequence_length],
                       k_data[index:index + sequence_length],
                       f_data[index:index + sequence_length]])

    if nomalize_window:
        result, first_data = normalise_windows(result)

    result = np.array(result)
    print("result shape : ", result.shape)
    row = result.shape[0] - seq_len
    train = result[:int(row),:,:]
    print("train shpae : ", train.shape)
    print("row : ", row)
    #np.random.shuffle(train)
    x_train = train[:-1,:,:]
    y_train = train[1:,:,-1]
    x_test = result[int(row)-1:-1,:,:]
    y_test = result[int(row):,:,-1]

    print("x_train shpae : ", x_train.shape)
    print("y_train shpae : ", y_train.shape)
    print("x_test shpae : ", x_test.shape)
    print("y_test shpae : ", y_test.shape)
    print(len(x_train), len(x_test))

    return [date_data[-1], first_data, x_train, y_train, x_test, y_test]


#시장지표
def load_market_data_db(date_val, seq_len, nomalize_window):

    sequence_length = seq_len
    r_data = []
    result = []
    first_data = []

    date_data = []
    k_data = []
    g_data = []
    d_data = []

    res_kospi, res_gold, res_dollars = market_data_service.get_data(date_val)

    for data in res_kospi:
        date_data.append(data['date'])
        kospi = float(data['kospi'].replace(",",""))
        k_data.append(kospi)

    for data in res_gold:
        gold = float(data['gold'].replace(",", ""))
        g_data.append(gold)

    for data in res_dollars:
        dollar = float(data['dollar'].replace(",", ""))
        d_data.append(dollar)

    for index in range(len(date_data) - sequence_length + 1):
        result.append([k_data[index:index + sequence_length],
                       g_data[index:index + sequence_length],
                       d_data[index:index + sequence_length]])

    if nomalize_window:
        result, mean_data, stdev_data = normalise_windows(result)

    result = np.array(result)
    print("result shape : ", result.shape)
    row = result.shape[0] - seq_len
    train = result[:int(row),:,:]
    print("train shpae : ", train.shape)
    print("row : ", row)
    #np.random.shuffle(train)
    x_train = train[:-1,:,:]
    y_train = train[1:,:,-1]
    x_test = result[int(row)-1:-1,:,:]
    y_test = result[int(row):,:,-1]

    print("x_train shpae : ", x_train.shape)
    print("y_train shpae : ", y_train.shape)
    print("x_test shpae : ", x_test.shape)
    print("y_test shpae : ", y_test.shape)
    print(len(x_train), len(x_test))

    return date_data[-1], mean_data, stdev_data, x_train, y_train, x_test, y_test


def load_data_file(filename, seq_len, nomalize_window):
    f = open(filename, 'r').read()
    data = f.split('\n')

    sequence_length = seq_len
    r_data = []
    result = []
    first_data = []

    date_data = []
    t_data = []
    p_min_data = []
    p_max_data = []

    for i in range(len(data)):
        r_data = data[i].split('\t')
        date_data.append(r_data[0])
        t_data.append(r_data[1].replace(",",""))
        p_min_data.append(r_data[2].replace(",",""))
        p_max_data.append(r_data[3].replace(",",""))

    for index in range(len(date_data) - sequence_length + 1):
        result.append([t_data[index:index + sequence_length],
                       p_min_data[index:index + sequence_length],
                       p_max_data[index:index + sequence_length]])

    if nomalize_window:
        result, mean_data, stdev_data = normalise_windows(result)

    result = np.array(result)
    print("result shape : ", result.shape)
    row = result.shape[0] - seq_len
    train = result[:int(row),:,:]
    print("train shpae : ", train.shape)
    print("row : ", row)
    #np.random.shuffle(train)
    x_train = train[:-1,:,:]
    y_train = train[1:,:,-1]
    x_test = result[int(row)-1:-1,:,:]
    y_test = result[int(row):,:,-1]

    print("x_train shpae : ", x_train.shape)
    print("y_train shpae : ", y_train.shape)
    print("x_test shpae : ", x_test.shape)
    print("y_test shpae : ", y_test.shape)
    print(len(x_train), len(x_test))

    return date_data[-1], first_data, x_train, y_train, x_test, y_test


def normalise_windows_backup(window_data):
    nomalised_data = []
    first_data = []

    for t,min_p,max_p in window_data:
        first_data.append((t[0],min_p[0],max_p[0]))
        nomalised_t = [((float(p) / float(t[0])) - 1) for p in t ]
        nomalised_min_t = [((float(p) / float(t[0])) - 1) for p in min_p ]
        nomalised_max_t = [((float(p) / float(t[0])) - 1) for p in max_p ]
        nomalised_data.append([nomalised_t, nomalised_min_t, nomalised_max_t])
    return nomalised_data,first_data


def normalise_windows(window_data):
    nomalised_data = []
    mean_data = []
    stdev_data = []

    for t,min_p,max_p in window_data:
        mean_data.append((mean(t),mean(min_p),mean(max_p)))
        stdev_data.append((stdev(t),stdev(min_p),stdev(max_p)))

        nomalised_p = [((p - mean(t)) / stdev(t)) for p in t]
        nomalised_min_p = [((p - mean(min_p)) / stdev(min_p)) for p in min_p ]
        nomalised_max_p = [((p - mean(max_p)) / stdev(max_p)) for p in max_p ]
        nomalised_data.append([nomalised_p, nomalised_min_p, nomalised_max_p])
    return nomalised_data, mean_data, stdev_data


def predict_real_data(model, data, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    print(len(data), prediction_len)

    for i in range(len(data)):
        curr_frame = data[i]
        curr_frame = np.array(curr_frame)
        curr_frame = np.reshape(curr_frame, (1, curr_frame.shape[0], curr_frame.shape[1]))
        prediction_seqs.append(model.predict((curr_frame)))

    return prediction_seqs


def predict_future_data(model, mean_data, stdev_data, y_data, prediction_len, window_size):

    gap_check = True

    curr_frame = y_data
    curr_frame = np.array(curr_frame)
    curr_frame = np.transpose(curr_frame)
    curr_frame = np.reshape(curr_frame, (1, curr_frame.shape[0], curr_frame.shape[1]))

    predicted = []
    f_list = []
    real_y_data = []

    t_mean = mean_data
    t_stdev = stdev_data

    #y_data 복구
    for j in range(len(y_data)):
        real_y_data.append(((y_data[j][0] * t_stdev[0]) + t_mean[0],
                            (y_data[j][1] * t_stdev[1]) + t_mean[1],
                            (y_data[j][2] * t_stdev[2]) + t_mean[2]))

    last_val = real_y_data[-1]
    
    #(정규화 하고자 하는 값 - 데이터의 평균) / 데이터의 표준편차
    for j in range(prediction_len):

        predicted.append(model.predict(curr_frame))
        curr_frame[0][0] = np.insert(curr_frame[0][0][1:], window_size -1 , predicted[j][0][0])
        curr_frame[0][1] = np.insert(curr_frame[0][1][1:], window_size -1 , predicted[j][0][1])
        curr_frame[0][2] = np.insert(curr_frame[0][2][1:], window_size -1 , predicted[j][0][2])

        f_list.append(((predicted[j][0][0] * t_stdev[0]) + t_mean[0],
                       (predicted[j][0][1] * t_stdev[1]) + t_mean[1],
                       (predicted[j][0][2] * t_stdev[2]) + t_mean[2]))
        
        # y_data에 future 추가
        real_y_data = real_y_data[1:]
        real_y_data.append(f_list[-1])

        # mean 확보
        t_mean = [mean(np.array(real_y_data)[:,0]), mean(np.array(real_y_data)[:,1]), mean(np.array(real_y_data)[:,2])]

        # stdev 확보
        t_stdev = [stdev(np.array(real_y_data)[:, 0]), stdev(np.array(real_y_data)[:, 1]), stdev(np.array(real_y_data)[:, 2])]

    if gap_check is True:
        gap_k_val = last_val[0] - f_list[0][0]
        gap_g_val = last_val[1] - f_list[0][1]
        gap_d_val = last_val[2] - f_list[0][2]

        new_f_list = []

        for i in range(len(f_list)):
            new_f_list.append( (f_list[i][0] + gap_k_val, f_list[i][1] + gap_g_val, f_list[i][2] + gap_d_val))

        return new_f_list

    return f_list


def make_model(code, data_val, w_size, nomalize, fromDb):
    model = Sequential();
    model.add(LSTM(6, input_shape=(3,w_size), activation='relu', return_sequences=True))
    model.add(LSTM(12, input_shape=(3,w_size), activation='relu', return_sequences=True))
    model.add(LSTM(12, input_shape=(3,w_size), activation='relu', return_sequences=True))
    model.add(LSTM(6, input_shape=(3, w_size), activation='relu', return_sequences=False))
    model.add(Dense(3))

    file_name = "./model/" + code + "_" + str(w_size) + ".h5"

    global mean_data, stdev_data, X_train, y_train, X_test, y_test

    if fromDb:
        last_date, mean_data, stdev_data, X_train, y_train, X_test, y_test = load_market_data_db(data_val, w_size, nomalize)
    else:
        last_date, mean_data, stdev_data, X_train, y_train, X_test, y_test = load_data_file('./csvdata' + code + '.csv', w_size, nomalize)

    #Model Exist
    if path.isfile(file_name) is True:
        print(file_name + "is Exist")
        model.load_weights(file_name)
    # Model is Not Exist
    else:
        model.summary()
        rms = rmsprop(lr=0.0001)
        model.compile(loss='mse', optimizer=rms, metrics=['accuracy'])
        start = time.time()
        print('compile time : ', time.time() - start)
        print(np.shape(X_train), "------", np.shape(y_train))

        hist = model.fit(X_train, y_train
                        ,batch_size=w_size
                        ,nb_epoch=300)
        print(device_lib.list_local_devices())
        plt.figure(figsize=(8,4))
        plt.subplot(1,1,1)
        plt.plot(hist.history['loss'],'y')
        plt.title("loss function")
        plt.ylabel("loss value")
        plt.plot(hist.history['accuracy'],'g')
        plt.legend()
        plt.tight_layout()
        plt.show()
        model.save_weights(file_name)
        print("save model : ", file_name)
    return last_date, model


def getBizDay(lastDate, period):
    bizDay = []

    #'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN',
    allDay = [6,5,4,3,2,1,0]
    print("PERIOD : ", period)
    out = pd.date_range(lastDate, periods=period)
    for i in range(len(out)):
        year = int(str(out[i])[:4])
        mon = int(str(out[i])[5:7])
        day = int(str(out[i])[8:10])

        idx = dt.date(year, mon, day).weekday()
        if allDay[idx] > 1:
            bizDay.append(str(out[i])[:10])

    return bizDay

#전역변수
#code = '051910' #LG CHEM
code = '051930'  #SAMSUNG ELE
w_size = 20
nomalize = True
fromDb = True
#data_val = '2016.01.18'
data_val = '2020.04.16'

#모델을 코드에 따라 생성하거나 불러오기
last_date, model_sec = make_model(code, data_val, w_size, nomalize, fromDb)

print("#### last_date : ", last_date)

predicted_real_data = predict_real_data(model_sec, X_test, w_size)
predicted_real_data = np.array(predicted_real_data)
predicted_real_data = np.reshape(predicted_real_data, (w_size, 3))
print("predicted_real_data shape : ", np.shape(predicted_real_data))


## predict_sz
ps = len(y_test)
t_cnt = len(mean_data) - ps

real_data = []
predicted_real_data_list = []
predicted_future_data_list = []


for i in range(len(y_test)):
    if nomalize:
        idx = i + t_cnt
        real_data.append([(y_test[i][0] * stdev_data[idx][0]) + mean_data[idx][0],
                          (y_test[i][1] * stdev_data[idx][1]) + mean_data[idx][1],
                          (y_test[i][2] * stdev_data[idx][2]) + mean_data[idx][2]])
    else:
        real_data.append(y_test[i])

for i in range(len(predicted_real_data)):
    if nomalize:
        idx = i + t_cnt
        predicted_real_data_list.append([(predicted_real_data[i][0] * stdev_data[idx][0]) + mean_data[idx][0],
                          (predicted_real_data[i][1] * stdev_data[idx][1]) + mean_data[idx][1],
                          (predicted_real_data[i][2] * stdev_data[idx][2]) + mean_data[idx][2]])
    else:
        predicted_real_data_list.append([float(predicted_real_data[i][0]), float(predicted_real_data[i][1]),float(predicted_real_data[i][2])])

predicted_future_data_list = predict_future_data(model_sec, mean_data[-1], stdev_data[-1], y_test, w_size, w_size)

#마지막 실제데이터 일부와 + 실제데이터의 예측데이터 + 미래 예측데이터를 확인
plot_results(last_date, real_data, predicted_real_data_list, predicted_future_data_list, w_size)

print("REAL : ", real_data)
print("REAL_PRE : ", predicted_real_data_list)
print("FUTURE_PRE : ", predicted_future_data_list)

#데이터 가져오는
# http://marketdata.krx.co.kr/mdi#document=040204