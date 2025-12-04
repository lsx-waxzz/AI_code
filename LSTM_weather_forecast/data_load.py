import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class WeatherDataset(Dataset):
    def __init__(self, raw_df, look_back=10, predict_steps=10, train_min=None, train_max=None, is_train=True):
        self.look_back = look_back          # 回看步数（历史窗口）
        self.predict_steps = predict_steps  # 单次预测步数
        self.is_train = is_train
        self.target_col = "T (degC)"       
        self.time_col = None

        # 1. 处理时间列（按时间排序，避免时序混乱）
        time_cols = [col for col in raw_df.columns if "date" in col.lower()]
        if not time_cols:
            raise ValueError("数据中未找到含'date'的时间列，请检查CSV文件格式")
        self.time_col = time_cols[0]
        self.df = raw_df.sort_values(by=self.time_col).reset_index(drop=True)
        
        # 2. 仅保留温度列作为输入特征
        if self.target_col not in self.df.columns:
            raise ValueError(f"CSV文件中未找到温度列 '{self.target_col}'，请检查列名")
        self.features = [self.target_col]  # 输入特征 = 目标特征（仅温度）
        self.target_idx = self.features.index(self.target_col)  # 固定为0（单特征）
        self.raw_data = torch.tensor(self.df[self.features].values, dtype=torch.float32)  # shape: [N, 1]

        # 3. 归一化（无数据泄露：训练集自算统计量，测试集复用）
        if self.is_train:
            self.min_val = torch.min(self.raw_data, dim=0).values  # shape: [1]
            self.max_val = torch.max(self.raw_data, dim=0).values  # shape: [1]
            self.scaled_data = (self.raw_data - self.min_val) / (self.max_val - self.min_val + 1e-8)  # 避免除零
        else:
            if train_min is None or train_max is None:
                raise ValueError("测试集必须传入训练集的min_val和max_val，避免数据泄露")
            self.min_val = train_min
            self.max_val = train_max
            self.scaled_data = (self.raw_data - self.min_val) / (self.max_val - self.min_val + 1e-8)

        # 4. 构建时序样本：[look_back步历史温度] → [predict_steps步未来温度]
        self.X, self.y, self.time_index = self.create_sequences()

        # 打印数据集信息（仅训练集）
        if self.is_train:
            print(f"📊 训练集样本构建完成：")
            print(f"  - 原始数据长度：{len(self.df)}")
            print(f"  - 样本数：{len(self.X)}")
            print(f"  - 输入形状：{self.X.shape}（[样本数, 回看步数, 特征数]）")
            print(f"  - 输出形状：{self.y.shape}（[样本数, 预测步数]）")
            print(f"  - 温度归一化范围：[{self.min_val.item():.2f}℃, {self.max_val.item():.2f}℃]")

    def create_sequences(self):
        X_list = []  # 输入序列：[样本数, look_back, 1]
        y_list = []  # 输出序列：[样本数, predict_steps]
        time_index_list = []  # 每个样本的预测起始时间

        # 循环范围：确保预测不越界（i + predict_steps ≤ 数据长度）
        for i in range(self.look_back, len(self.scaled_data) - self.predict_steps + 1):
            # 输入：i-look_back ~ i-1 步的历史温度（归一化后）
            X = self.scaled_data[i - self.look_back:i, :]  # shape: [look_back, 1]
            # 输出：i ~ i+predict_steps-1 步的未来温度（归一化后）
            y = self.scaled_data[i:i + self.predict_steps, self.target_idx]  # shape: [predict_steps]
            # 记录预测起始时间（对应输出第1步的真实时间）
            time_idx = self.df[self.time_col].iloc[i]

            X_list.append(X)
            y_list.append(y)
            time_index_list.append(time_idx)

        # 堆叠为张量（空样本保护）
        return (
            torch.stack(X_list) if X_list else torch.tensor([]),  # X: [N, look_back, 1]
            torch.stack(y_list) if y_list else torch.tensor([]),  # y: [N, predict_steps]
            pd.Series(time_index_list)  # 预测起始时间序列
        )

    def inverse_transform_temp(self, scaled_temp):
        """反归一化：将归一化后的温度恢复为原始尺度"""
        temp_min = self.min_val[self.target_idx]
        temp_max = self.max_val[self.target_idx]
        return scaled_temp * (temp_max - temp_min) + temp_min

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
