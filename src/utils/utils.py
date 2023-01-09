import pandas as pd
import numpy as np
import random
import os
import natsort
from sklearn.metrics import mean_absolute_error
import torch
from config import *


def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# 데이터셋 생성 함수
def build_train_dataset(time_series, window_size):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - window_size):
        _x = time_series[i : i + window_size, :]
        _y = time_series[i + window_size, [0, 1]]
        # print(_x, "-->",_y)
        dataX.append(_x)
        dataY.append(_y)

    return np.array(dataX), np.array(dataY)


def build_test_dataset(time_series, window_size):
    dataX = []
    for i in range(0, len(time_series), window_size):
        _x = time_series[i : i + window_size, :]
        # print(_x, "-->",_y)
        dataX.append(_x)

    return np.array(dataX)


def result(test, name="", path="../dataset/", to="../submit/"):
    submit = pd.read_csv(f"{path}submission_sample.csv")

    for idx, col in enumerate(submit.columns):
        if col == "TIME":
            continue
        submit[col] = test[:, idx - 1]

    submit.to_csv(f"{to}{name}.csv", index=False)
    print("Done.")

    return f"{to}{name}.csv"


def mae_score(y_true, y_pred):
    score = 0
    for i in [0, 1]:
        score += mean_absolute_error(y_true[:, i], y_pred[:, i])

    return score


def load_weather(path="../dataset/weather/"):
    # Load weather data
    weather_2019 = pd.read_csv(f"{path}OBS_ASOS_TIM_2019.csv", encoding="cp949")
    weather_2020 = pd.read_csv(f"{path}OBS_ASOS_TIM_2020.csv", encoding="cp949")
    weather_2021 = pd.read_csv(f"{path}OBS_ASOS_TIM_2021.csv", encoding="cp949")

    weather = pd.concat([weather_2019, weather_2020, weather_2021])

    return weather


def load_test_data(path="../dataset/test/"):
    file_list = os.listdir(path)
    sorted_file_list = natsort.natsorted(file_list)
    # print(sorted_file_list[:10])
    test_list = [
        file for file in sorted_file_list if file.endswith(".csv")
    ]  # 파일명 끝이 .csv인 경우

    # csv 파일들을 DataFrame으로 불러와서 concat

    test = pd.DataFrame()
    for i in test_list:
        data = pd.read_csv(path + i)
        test = pd.concat([test, data])

    test = test.reset_index(drop=True)

    return test


def set_pytorch_dataset(
    processed_train,
    ratio=0.8,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    # 데이터를 정렬하여 전체 데이터의 80% 학습, 10% 검증, 10% 테스트에 사용
    train_size = int(len(processed_train) * ratio)
    # valid_size = int((len(processed_train)*ratio + len(processed_train))/2)
    train_set = processed_train[0:train_size]
    valid_set = processed_train[train_size - config.window_size :]
    # valid_set = processed_train[train_size-config.window_size:valid_size]
    # score_set = processed_train[valid_size-config.window_size:]

    trainX, trainY = build_train_dataset(np.array(train_set), config.window_size)
    validX, validY = build_train_dataset(np.array(valid_set), config.window_size)
    # scoreX, scoreY = build_train_dataset(np.array(score_set), config.window_size)

    # 텐서로 변환
    trainX_tensor = torch.FloatTensor(trainX).to(device)
    trainY_tensor = torch.FloatTensor(trainY).to(device)

    validX_tensor = torch.FloatTensor(validX).to(device)
    validY_tensor = torch.FloatTensor(validY).to(device)

    # scoreX_tensor = torch.FloatTensor(scoreX).to(device)
    # scoreY_tensor = torch.FloatTensor(scoreY).to(device)

    return trainX_tensor, trainY_tensor, validX_tensor, validY_tensor
