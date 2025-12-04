import torch
import torch.nn as nn

class LSTMWeatherForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, dropout=0.2, predict_steps=10):
        super(LSTMWeatherForecast, self).__init__()
        # LSTM层：接收单特征时序输入 [batch, look_back, 1]
        self.lstm = nn.LSTM(
            input_size=input_size,  # 固定为1（单特征：温度）
            hidden_size=hidden_size,  # 隐藏层维度（可调整）
            num_layers=num_layers,    # LSTM层数
            batch_first=True,         # 输入格式：[batch, seq_len, input_size]
            dropout=dropout if num_layers > 1 else 0  # 单层不启用dropout
        )
        # 全连接层：将LSTM输出映射为10步温度预测
        self.fc = nn.Linear(hidden_size, predict_steps)
        self.predict_steps = predict_steps

    def forward(self, x):
        # x shape: [batch_size, look_back, 1]（单特征输入）
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, look_back, hidden_size]
        last_step_feat = lstm_out[:, -1, :]  # 取最后一步隐藏状态（包含历史时序信息）
        temp_pred = self.fc(last_step_feat)  # 输出：[batch_size, predict_steps]（10步温度）
        return temp_pred

# 测试模型形状（确保适配单特征输入）
if __name__ == "__main__":
    LOOK_BACK = 10
    PREDICT_STEPS = 10
    BATCH_SIZE = 32

    # 模拟单特征输入：[32, 10, 1]（batch_size=32，look_back=10，input_size=1）
    dummy_input = torch.randn(BATCH_SIZE, LOOK_BACK, 1)
    print(f"模拟输入形状：{dummy_input.shape}")  # 输出：torch.Size([32, 10, 1])

    # 初始化模型
    model = LSTMWeatherForecast(input_size=1, predict_steps=PREDICT_STEPS)
    dummy_pred = model(dummy_input)
    print(f"模型输出形状：{dummy_pred.shape}")  # 输出：torch.Size([32, 10])（符合10步预测）
    print("✅ 模型形状测试通过（单特征输入适配成功）")
