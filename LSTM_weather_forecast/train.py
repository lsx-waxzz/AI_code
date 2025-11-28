import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_load import WeatherDataset
from model import LSTMWeatherForecast

if __name__ == "__main__":
    CSV_PATH = "./data/long_term_forecast/weather/weather.csv"
    LOOK_BACK = 10
    BATCH_SIZE = 32
    EPOCHS = 50    # 训练轮数
    LEARNING_RATE = 0.001    # 学习率
    SAVE_PATH = "./best_lstm_temp_model.pth"

    # 创建训练集（is_train=True，取前80%样本）和验证集（is_train=False，取后20%样本）
    train_dataset = WeatherDataset(csv_path=CSV_PATH, look_back=LOOK_BACK, is_train=True)
    val_dataset = WeatherDataset(csv_path=CSV_PATH, look_back=LOOK_BACK, is_train=False)

    # 构建DataLoader，负责批量加载、打乱（训练集）数据
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    INPUT_SIZE = len(train_dataset.features)
    print(f"自动获取输入特征数：{INPUT_SIZE}")

    model = LSTMWeatherForecast(input_size=INPUT_SIZE)
    criterion = nn.MSELoss()   # 损失函数：均方误差
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)    # 优化器：Adam

    best_val_loss = float('inf')   # 初始化最优验证损失为无穷大

    print("\n开始训练...")
    for epoch in range(EPOCHS):
        # 训练阶段：更新模型参数，最小化训练集上的损失
        model.train()   # 将模型切换为训练模式
        train_loss = 0.0   # 累加本轮训练的总损失
        for batch_x, batch_y in train_loader:  # 迭代训练集的所有批量
            outputs = model(batch_x)   # 前向传播：输入批量数据，得到模型预测值
            loss = criterion(outputs, batch_y)  # 计算损失：预测值与真实温度的均方误差
            optimizer.zero_grad()  # 梯度清零：避免上一个批量的梯度累积
            loss.backward()    # 反向传播：计算模型参数的梯度
            optimizer.step()   # 参数更新：根据梯度和学习率，更新模型的权重、偏置等参数
            train_loss += loss.item() * batch_x.size(0)   # 累加总损失
        
        # 计算本轮训练的平均损失
        avg_train_loss = train_loss / len(train_dataset)

        # 验证阶段：评估模型泛化能力，不更新参数
        model.eval()  # 将模型切换为验证模式
        val_loss = 0.0   # 累加本轮验证的总损失
        with torch.no_grad():  # 关闭梯度计算
            for batch_x, batch_y in val_loader:   # 迭代验证集的所有批量
                outputs = model(batch_x)  # 前向传播：仅计算预测值，无梯度
                loss = criterion(outputs, batch_y)  # 计算损失
                val_loss += loss.item() * batch_x.size(0)  # 累加总损失
        
        # 计算本轮验证的平均损失
        avg_val_loss = val_loss / len(val_dataset)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | 训练损失：{avg_train_loss:.6f} | 验证损失：{avg_val_loss:.6f}")
        # 若当前验证损失是历史最小，保存模型参数
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  → 保存最优模型（验证损失：{best_val_loss:.6f}）")

    print("\n训练结束")
    print(f"最优模型已保存至：{SAVE_PATH}")
    print(f"最小验证损失：{best_val_loss:.6f}")