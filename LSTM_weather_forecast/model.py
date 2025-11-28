import torch
import torch.nn as nn

class LSTMWeatherForecast(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.2):
        super(LSTMWeatherForecast, self).__init__()  # 继承父类初始化方法
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0 # 随即丢弃防止过拟合
        )
        # 全连接层
        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=1 # 最后输出一个值作为温度预测值
        )
    # 前向传播
    def forward(self, x):
        lstm_out, _ = self.lstm(x) # 所有时间步的隐藏状态输出
        last_step_feature = lstm_out[:, -1, :] # 从 LSTM 的输出中提取最后一个时间步的隐藏状态
        temp_pred = self.fc(last_step_feature) # 将最后一个时间步的特征传入全连接层
        return temp_pred 


if __name__ == "__main__":
    INPUT_SIZE = 21  
    LOOK_BACK = 10
    BATCH_SIZE = 32

    dummy_input = torch.randn(BATCH_SIZE, LOOK_BACK, INPUT_SIZE)
    print(f"模拟输入形状：{dummy_input.shape}")

    model = LSTMWeatherForecast(input_size=INPUT_SIZE)
    print(f"模型参数设备：{next(model.parameters()).device}")

    dummy_temp_pred = model(dummy_input)
    print(f"模型输出形状：{dummy_temp_pred.shape}")
    print("模型测试通过")