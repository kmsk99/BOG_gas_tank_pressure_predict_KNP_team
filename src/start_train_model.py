import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings
import random
import os
import datetime
import torch

from torch.utils.data import TensorDataset  # 텐서데이터셋
from torch.utils.data import DataLoader  # 데이터로더

# Import user libraries
from config import *
from utils.utils import *
from utils.plot import *
from pipelines.weather_pipeline import *
from pipelines.train_pipeline import *
from utils.train_model import *
from models.LD import *
from utils.model_test import *

warnings.filterwarnings("ignore")

lstm_node = 16
dense_node = 2
learning_rate = config.lr
masking = 31
start_train = bool(int(sys.argv[1])) if len(sys.argv) > 1 else True

seed_everything()

print(f"{'Weather Pipeline Started':=^40}")

weather = load_weather()

weather_pipeline = get_weather_pipeline()

processed_weather = weather_pipeline.fit_transform(weather)

print(f"{'Train Pipeline Started':=^40}")

train = pd.read_csv("../dataset/train/train.csv")

train_pipeline = load_train_pipeline(processed_weather, config.threhold, masking)

processed_train = pd.DataFrame(
    train_pipeline.fit_transform(train),
    columns=train_pipeline["final_pipe"].get_feature_names_out(),
)

scaler = train_pipeline["final_pipe"].named_transformers_["y_scaler"]

print(f"{'Test Pipeline Started':=^40}")

# make test dataset
test = load_test_data()

processed_test = pd.DataFrame(
    train_pipeline.transform(test),
    columns=train_pipeline["final_pipe"].get_feature_names_out(),
)

print(f"{'Setting Dataset Started':=^40}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터를 정렬하여 전체 데이터의 70% 학습, 30% 테스트에 사용
trainX_tensor, trainY_tensor, validX_tensor, validY_tensor = set_pytorch_dataset(
    processed_train, device=device
)

testX = build_test_dataset(np.array(processed_test), config.window_size)
testX_tensor = torch.FloatTensor(testX).to(device)

# 텐서 형태로 데이터 정의
trainset = TensorDataset(trainX_tensor, trainY_tensor)

# 데이터로더는 기본적으로 2개의 인자를 입력받으며 배치크기는 통상적으로 2의 배수를 사용
trainloader = DataLoader(
    trainset, batch_size=config.batch_size, shuffle=True, drop_last=True
)

# 텐서 형태로 데이터 정의
validset = TensorDataset(validX_tensor, validY_tensor)

# 데이터로더는 기본적으로 2개의 인자를 입력받으며 배치크기는 통상적으로 2의 배수를 사용
validloader = DataLoader(
    validset, batch_size=config.batch_size, shuffle=False, drop_last=True
)


# 설정값
data_dim = processed_train.shape[1]
output_dim = 2

if start_train:
    print(f"{'Model Training Started':=^40}")

    # 모델 학습
    lstm_dense = LstmDense(
        data_dim, lstm_node, dense_node, config.window_size, output_dim, 2
    ).to(device)
    model, train_hist, valid_hist = train_model(
        lstm_dense,
        trainloader,
        validloader,
        device=device,
        lr=learning_rate,
        verbose=1,
        num_epochs=config.epochs,
        patience=config.es,
    )

if not start_train:
    # 불러오기
    model = LstmDense(
        data_dim, lstm_node, dense_node, config.window_size, output_dim, 2
    ).to(device)
    model.load_state_dict(torch.load("checkpoint.pt", map_location=device), strict=False)


model.eval()


print(f"{'Model Testing Started':=^40}")

# 예측 테스트

valid_pred_inverse, validY_inverse = model_valid(
    model, scaler, validX_tensor, validY_tensor
)
test_pred_inverse = model_test(model, scaler, testX_tensor)

# 성능 측정
mae = mae_score(valid_pred_inverse, validY_inverse)
print("MAE SCORE : ", mae)

nowDatetime = datetime.now().strftime("%Y%m%d%H%M%S")

file_name = f"{nowDatetime}_{mae:06f}"

# 모델 저장
submit_csv = result(test_pred_inverse, file_name)


print(f"{'Model Visualizing Started':=^40}")

# 시각화
# epoch_hist(train_hist, valid_hist, file_name)
# plot_two(valid_pred_inverse, validY_inverse, file_name)
# plot_diff(valid_pred_inverse, validY_inverse, file_name)
