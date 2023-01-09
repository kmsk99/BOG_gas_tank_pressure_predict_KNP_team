import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from utils.scheduler import *
from utils.earlystopping import *


def weighted_mae_loss(output, target, prev):
    mae = torch.unsqueeze(torch.abs(output - target), dim=1)
    weight = torch.abs(target - prev)

    weight = torch.unsqueeze(
        weight / torch.unsqueeze(torch.sum(torch.add(weight, 0.00000001), 1), dim=1),
        dim=2,
    )

    return torch.sum(torch.bmm(mae, weight))


def train_model(
    model,
    train_loader,
    valid_loader,
    device,
    num_epochs=None,
    lr=None,
    verbose=20,
    patience=10,
):

    # 모델이 학습되는 동안 trainning loss를 track
    train_losses = []
    # 모델이 학습되는 동안 validation loss를 track
    valid_losses = []
    # epoch당 average training loss를 track
    avg_train_losses = []
    # epoch당 average validation loss를 track
    avg_valid_losses = []

    criterion = weighted_mae_loss
    optimizer = optim.Adam(model.parameters(), lr=0.000001)
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer, T_0=50, T_mult=1, eta_max=lr, T_up=10, gamma=0.5
    )
    early_stopping = EarlyStopping(patience=patience, verbose=False)

    # epoch마다 loss 저장
    train_hist = np.zeros(num_epochs)
    iters = len(train_loader)
    for epoch in range(num_epochs):

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for batch, (data, target) in enumerate(train_loader, 1):
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: 입력된 값을 모델로 전달하여 예측 출력 계산
            output = model(data)
            # calculate the loss
            loss = criterion(output, target, data[:, 5, :2])
            # backward pass: 모델의 파라미터와 관련된 loss의 그래디언트 계산
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            scheduler.step(epoch + batch / iters)
            # record training loss
            train_losses.append(loss.item())

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for data, target in valid_loader:
            # forward pass: 입력된 값을 모델로 전달하여 예측 출력 계산
            output = model(data)
            # calculate the loss
            loss = criterion(output, target, data[:, 5, :2])
            # record validation loss
            valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        if epoch % verbose == 0:
            epoch_len = len(str(num_epochs))

            print_msg = (
                f"[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] "
                + f"train_loss: {train_loss:.8f} "
                + f"valid_loss: {valid_loss:.8f}"
            )

            print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping는 validation loss가 감소하였는지 확인이 필요하며,
        # 만약 감소하였을경우 현제 모델을 checkpoint로 만든다.
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # best model이 저장되어있는 last checkpoint를 로드한다.
    model.load_state_dict(torch.load("checkpoint.pt"))

    return model, avg_train_losses, avg_valid_losses
