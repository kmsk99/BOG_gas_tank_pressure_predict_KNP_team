import torch
import torch.nn as nn


class LstmDense(nn.Module):
    # # 기본변수, layer를 초기화해주는 생성자
    def __init__(self, input_dim, hidden_dim, fc1_dim, seq_len, output_dim, layers):
        super(LstmDense, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.layers = layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=layers,
            # dropout = 0.1,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_dim, fc1_dim, bias=True)
        self.fc2 = nn.Linear(fc1_dim, output_dim, bias=True)

    # 학습 초기화를 위한 함수
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.layers, self.seq_len, self.hidden_dim),
            torch.zeros(self.layers, self.seq_len, self.hidden_dim),
        )

    # 예측을 위한 함수
    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc1(x[:, -1])
        x = self.fc2(x)
        return x
