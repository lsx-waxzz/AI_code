import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from data_load import WeatherDataset
from model import LSTMWeatherForecast

if __name__ == "__main__":
    # -------------------------- 训练配置（与评估一致）--------------------------
    CSV_PATH = "./data/long_term_forecast/weather/weather.csv"
    LOOK_BACK = 10          # 回看步数（历史窗口）
    PREDICT_STEPS = 10      # 单次预测步数（一次性输出10步）
    BATCH_SIZE = 32         # 批次大小
    EPOCHS = 50             # 训练轮数
    LEARNING_RATE = 0.001   # 学习率
    SAVE_PATH = "./10step_lstm_single_feature_model.pth"  # 模型保存路径

    # -------------------------- 1. 数据划分（严格隔离测试集）--------------------------
    full_df = pd.read_csv(CSV_PATH)
    time_col = next(col for col in full_df.columns if "date" in col.lower())
    full_df[time_col] = pd.to_datetime(full_df[time_col])
    full_df = full_df.sort_values(by=time_col).reset_index(drop=True)
    total_len = len(full_df)

    # 训练集：前80%数据（仅用训练集训练，不碰测试集）
    train_ratio = 0.8
    train_size_raw = int(total_len * train_ratio)
    train_raw_df = full_df.iloc[:train_size_raw].copy()

    # 测试集起始位置后移LOOK_BACK步（确保测试集评估时的初始输入不依赖训练集）
    test_start = train_size_raw + LOOK_BACK
    if test_start >= total_len:
        test_start = train_size_raw
        print(f"⚠️  数据量不足，测试集起始位置回退至训练集结束位置（{test_start}）")
    
    print("="*60)
    print("📌 数据划分结果（单特征：仅温度）")
    print(f"训练集时间范围：{train_raw_df[time_col].min()} ~ {train_raw_df[time_col].max()}")
    print(f"训练集原始数据行数：{len(train_raw_df)}")
    print(f"测试集起始位置：{test_start}（后续评估用，训练时不接触）")
    print("="*60)

    # -------------------------- 2. 创建训练集（单特征：仅温度）--------------------------
    train_dataset = WeatherDataset(
        raw_df=train_raw_df,
        look_back=LOOK_BACK,
        predict_steps=PREDICT_STEPS,
        is_train=True
    )
    if len(train_dataset) == 0:
        print("❌ 训练集样本数为0，请增大数据集或调整LOOK_BACK/PREDICT_STEPS")
        exit(1)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # -------------------------- 3. 初始化模型（单特征输入）--------------------------
    INPUT_SIZE = len(train_dataset.features)  # 固定为1（仅温度）
    print(f"\n✅ 模型初始化（输入特征数：{INPUT_SIZE}，仅温度）")
    model = LSTMWeatherForecast(
        input_size=INPUT_SIZE,
        hidden_size=32,
        num_layers=2,
        dropout=0.2,
        predict_steps=PREDICT_STEPS
    )

    # 损失函数+优化器
    criterion = nn.MSELoss()  # 适合回归任务，支持多步预测误差计算
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # -------------------------- 4. 开始训练（单特征时序预测）--------------------------
    print(f"\n🚀 开始训练（共{EPOCHS}轮，单特征LSTM）...")
    print("="*60)
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            # batch_x: [32, 10, 1]（单特征输入：10步历史温度）
            # batch_y: [32, 10]（10步未来真实温度）
            outputs = model(batch_x)  # outputs: [32, 10]（10步温度预测）
            loss = criterion(outputs, batch_y)  # 计算10步整体预测误差

            # 反向传播+参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)  # 按批次加权累加损失

        # 计算平均损失
        avg_train_loss = train_loss / len(train_dataset)
        print(f"Epoch [{epoch+1:2d}/{EPOCHS}] | 训练平均损失：{avg_train_loss:.6f}")

        # 保存最优模型（基于训练损失）
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(model.state_dict(), SAVE_PATH)

    # -------------------------- 5. 训练完成 --------------------------
    print("="*60)
    print(f"🎉 训练结束！")
    print(f"最优模型已保存至：{SAVE_PATH}")
    print(f"最优训练损失：{best_loss:.6f}")
    print(f"下一步：运行 evaluate.py 进行单特征盲预测评估（无伪数据）")
