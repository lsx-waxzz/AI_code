import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

class WeatherDataset(Dataset):
    def __init__(self, csv_path, look_back=10, is_train=True):
        self.look_back = look_back
        self.is_train = is_train
        self.target_col = "T (degC)"  # 目标列名
        # 读取数据集
        df = pd.read_csv(csv_path)  # df 是一个 DataFrame，行数 = 1000，列数 = 22

        # 处理时间列
        time_cols = [col for col in df.columns if "date" in col.lower()]  # 找到包含 "date" 关键词的列
        self.time_col = time_cols[0]  # 只有一个时间列"date"
        df[self.time_col] = pd.to_datetime(df[self.time_col]) # 转换为 datetime 格式
        df = df.sort_values(by=self.time_col).reset_index(drop=True)  # 按时间排序，并重置索引
        self.full_time_index = df[self.time_col].iloc[self.look_back:] # 筛选有效时间序列

        # 筛选数值特征+定位温度列
        self.features = df.select_dtypes(include=np.number).columns.tolist() # 筛选数值列
        self.target_idx = self.features.index(self.target_col)  # 目标列索引
        self.raw_data = torch.tensor(df[self.features].values, dtype=torch.float32) # 转换为张量

        # 归一化
        self.min_val = torch.min(self.raw_data, dim=0).values # 按列求最小值
        self.max_val = torch.max(self.raw_data, dim=0).values # 按列求最大值
        self.scaled_data = (self.raw_data - self.min_val) / (self.max_val - self.min_val + 1e-8) # 归一化

        # 构建时序样本
        self.X, self.y = self.create_sequences() # 构建时序样本

        # 划分训练/测试集（前80%/后20%）
        train_size = int(len(self.X) * 0.8)
        # 训练集：取前80%的输入和标签
        if self.is_train:
            self.X = self.X[:train_size]
            self.y = self.y[:train_size]
            self.time_index = self.full_time_index[:train_size]
        # 测试集：取后20%的输入和标签    
        else:
            self.X = self.X[train_size:]
            self.y = self.y[train_size:]
            self.time_index = self.full_time_index[train_size:]

    def create_sequences(self):
        X_list = [] # 存储输入样本：每个元素是[look_back, 特征数]的张量
        y_list = [] # 存储输出样本：每个元素是单个温度值
        for i in range(self.look_back, len(self.scaled_data)):
            X_list.append(self.scaled_data[i - self.look_back:i, :]) # 取look_back个时间步的数据作为输入
            y_list.append(self.scaled_data[i, self.target_idx]) # 取当前时间步的目标值作为输出
        return torch.stack(X_list), torch.tensor(y_list, dtype=torch.float32).unsqueeze(1) # 将列表转为张量

    def inverse_transform_temp(self, scaled_temp):
        # 反归一化温度
        temp_min = self.min_val[self.target_idx]
        temp_max = self.max_val[self.target_idx]
        return scaled_temp * (temp_max - temp_min) + temp_min

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 测试代码
if __name__ == "__main__":
    CSV_PATH = "./data/long_term_forecast/weather/weather.csv"
    
    train_dataset = WeatherDataset(csv_path=CSV_PATH, look_back=10, is_train=True)
    test_dataset = WeatherDataset(csv_path=CSV_PATH, look_back=10, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2) # 训练数据随机抽取，防止模型记忆顺序
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2) # 测试数据顺序不变

    # 打印核心信息
    print(f"训练集样本数：{len(train_dataset)}，测试集样本数：{len(test_dataset)}")
    for batch_x, batch_y in train_loader:
        print(f"输入形状：{batch_x.shape}，温度目标形状：{batch_y.shape}")
        print(f"数据设备：{batch_x.device}")
        break