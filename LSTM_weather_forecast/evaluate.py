import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_load import WeatherDataset
from model import LSTMWeatherForecast
import matplotlib.dates as mdates
from math import ceil

if __name__ == "__main__":
    # -------------------------- 评估配置（与训练一致）--------------------------
    CSV_PATH = "./data/long_term_forecast/weather/weather.csv"
    LOOK_BACK = 10          # 回看步数（历史窗口）
    PREDICT_STEPS = 10      # 单次预测步数
    BATCH_SIZE = 1          # 评估时使用批量1，便于逐样本处理
    MODEL_PATH = "./10step_lstm_single_feature_model.pth"  # 模型路径

    # -------------------------- 拆分绘图配置 --------------------------
    SPLIT_STRATEGY = "time"  # 固定按时间跨度拆分
    TIME_SPAN_DAYS = 10      # 每个子图展示10天数据
    SUBPLOTS_PER_ROW = 1     # 每页1行（单张大图更清晰）
    SUBPLOTS_PER_COL = 1     # 每页1列 → 每页仅1张10天的图

    # -------------------------- 1. 数据准备（严格隔离测试集）--------------------------
    full_df = pd.read_csv(CSV_PATH)
    time_col = next(col for col in full_df.columns if "date" in col.lower())
    full_df[time_col] = pd.to_datetime(full_df[time_col])
    full_df = full_df.sort_values(by=time_col).reset_index(drop=True)
    total_len = len(full_df)

    # 与训练集保持一致的划分方式
    train_ratio = 0.8
    train_size_raw = int(total_len * train_ratio)
    test_start = train_size_raw + LOOK_BACK
    if test_start >= total_len:
        test_start = train_size_raw
    
    # 训练集用于获取归一化参数
    train_raw_df = full_df.iloc[:train_size_raw].copy()
    train_dataset = WeatherDataset(
        raw_df=train_raw_df,
        look_back=LOOK_BACK,
        predict_steps=PREDICT_STEPS,
        is_train=True
    )

    # 测试集（不包含训练数据）
    test_raw_df = full_df.iloc[test_start - LOOK_BACK:].copy()  # 预留LOOK_BACK长度的初始输入
    test_dataset = WeatherDataset(
        raw_df=test_raw_df,
        look_back=LOOK_BACK,
        predict_steps=PREDICT_STEPS,
        train_min=train_dataset.min_val,
        train_max=train_dataset.max_val,
        is_train=False
    )
    
    if len(test_dataset) == 0:
        print("❌ 测试集样本数为0，请增大数据集或调整LOOK_BACK/PREDICT_STEPS")
        exit(1)
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("="*60)
    print("📌 测试集信息（单特征：仅温度）")
    print(f"测试集时间范围：{test_raw_df[time_col].min()} ~ {test_raw_df[time_col].max()}")
    print(f"测试集样本数：{len(test_dataset)}")
    print(f"使用滚动预测：用前面的预测值继续预测后面的值")
    print(f"绘图策略：1) 全量测试集总览图 2) 每{TIME_SPAN_DAYS}天生成一张拆分对比图")
    print("="*60)

    # -------------------------- 2. 加载模型 --------------------------
    model = LSTMWeatherForecast(
        input_size=1,
        hidden_size=32,
        num_layers=2,
        dropout=0.2,
        predict_steps=PREDICT_STEPS
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  # 切换到评估模式
    print(f"✅ 模型加载完成：{MODEL_PATH}")

    # -------------------------- 3. 滚动预测（不使用测试集真实数据）--------------------------
    all_predictions = []
    all_actuals = []
    all_times = []

    with torch.no_grad():  # 关闭梯度计算
        for i, (batch_x, batch_y) in enumerate(test_loader):
            # 初始输入：真实历史数据
            current_input = batch_x  # shape: [1, LOOK_BACK, 1]
            
            # 一次性预测未来PREDICT_STEPS步（全程不使用测试集真实数据）
            pred = model(current_input)  # shape: [1, PREDICT_STEPS]
            
            # 存储预测结果和真实值（反归一化）
            pred_denorm = test_dataset.inverse_transform_temp(pred).numpy()[0]
            actual_denorm = test_dataset.inverse_transform_temp(batch_y).numpy()[0]
            
            all_predictions.extend(pred_denorm)
            all_actuals.extend(actual_denorm)
            
            # 记录时间点（匹配数据的实际采样频率，小时级）
            start_time = test_dataset.time_index.iloc[i]
            time_steps = pd.date_range(start=start_time, periods=PREDICT_STEPS, freq='H')  # 小时级采样
            all_times.extend(time_steps)
            
            # 打印进度
            if (i + 1) % 10 == 0 or i + 1 == len(test_loader):
                print(f"已完成 {i + 1}/{len(test_loader)} 个测试样本预测")

    # 去重时间（处理滚动预测的重叠时间点）
    results_df = pd.DataFrame({
        'time': all_times,
        'predicted': all_predictions,
        'actual': all_actuals
    }).drop_duplicates(subset='time').sort_values('time').reset_index(drop=True)

    # -------------------------- 4. 计算评估指标 --------------------------
    mse = np.mean((results_df['predicted'] - results_df['actual']) **2)
    mae = np.mean(np.abs(results_df['predicted'] - results_df['actual']))
    rmse = np.sqrt(mse)

    print("\n" + "="*60)
    print("📊 预测评估指标（全量测试集）")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"测试集数据总量：{len(results_df)} 个时间点")
    print(f"测试集总时间跨度：{(results_df['time'].max() - results_df['time'].min()).days} 天")
    print("="*60)

    # -------------------------- 5. 第一步：绘制全量测试集总览图（所有数据在一张图）--------------------------
    print("\n🎨 开始绘制全量测试集总览图...")
    # 创建超大画布适配全量数据
    fig, ax = plt.subplots(figsize=(40, 12))

    # 绘制全量真实值和预测值
    ax.plot(results_df['time'], results_df['actual'], label='真实温度', color='blue', linewidth=1.5)
    ax.plot(results_df['time'], results_df['predicted'], label='预测温度', color='red', linestyle='--', linewidth=1.5)

    # 总览图标题
    total_start_str = results_df['time'].min().strftime("%Y-%m-%d")
    total_end_str = results_df['time'].max().strftime("%Y-%m-%d")
    ax.set_title(
        f'温度预测 vs 真实温度（滚动预测）- 全量测试集总览（{total_start_str} ~ {total_end_str}）',
        fontsize=22, pad=25
    )

    # 坐标轴配置
    ax.set_xlabel('时间', fontsize=18)
    ax.set_ylabel('温度 (°C)', fontsize=18)
    ax.legend(fontsize=16, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)

    # 格式化总览图时间轴（根据总跨度自适应）
    ax.tick_params(axis='x', rotation=45, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    total_days = (results_df['time'].max() - results_df['time'].min()).days
    
    # 总跨度>60天 → 按周显示；30-60天 → 每5天；<30天 → 每2天
    if total_days > 60:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))  # 每周
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif total_days > 30:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))      # 每5天
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))      # 每2天
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # 调整布局并保存总览图
    plt.tight_layout()
    total_plot_filename = f'temperature_forecast_full_testset_{total_start_str}_{total_end_str}.png'
    plt.savefig(total_plot_filename, dpi=300, bbox_inches='tight')
    print(f"✅ 全量测试集总览图已保存：{total_plot_filename}")
    plt.close()  # 释放内存

    # -------------------------- 6. 第二步：按10天拆分绘制独立图片 --------------------------
    def split_data_by_10days(df):
        """
        按每10天拆分数据，返回每个10天段的数据集
        """
        split_dfs = []
        start_time = df['time'].min()
        end_time = df['time'].max()
        current_start = start_time
        
        while current_start < end_time:
            # 每次取10天的时间范围
            current_end = current_start + pd.Timedelta(days=TIME_SPAN_DAYS)
            # 筛选当前10天的数据
            segment_df = df[(df['time'] >= current_start) & (df['time'] < current_end)].copy()
            if not segment_df.empty:
                split_dfs.append((current_start, current_end, segment_df))
            # 滑动到下一个10天
            current_start = current_end
        return split_dfs

    # 按10天拆分数据
    split_segments = split_data_by_10days(results_df)
    print(f"\n📎 测试集已拆分为 {len(split_segments)} 个10天时间段，开始绘制拆分图...")

    # 逐段绘制10天独立图片
    for seg_idx, (seg_start, seg_end, seg_df) in enumerate(split_segments):
        # 创建画布（适配10天数据的宽高）
        fig, ax = plt.subplots(figsize=(25, 10))
        
        # 绘制当前10天的预测值和真实值
        ax.plot(seg_df['time'], seg_df['actual'], label='真实温度', color='blue', linewidth=1.8)
        ax.plot(seg_df['time'], seg_df['predicted'], label='预测温度', color='red', linestyle='--', linewidth=1.8)
        
        # 子图标题（标注10天时间段）
        seg_start_str = seg_start.strftime("%Y-%m-%d")
        seg_end_str = seg_end.strftime("%Y-%m-%d")
        ax.set_title(
            f'温度预测 vs 真实温度（滚动预测）- 时间段：{seg_start_str} ~ {seg_end_str}（共{TIME_SPAN_DAYS}天）', 
            fontsize=18, pad=20
        )
        
        # 坐标轴标签
        ax.set_xlabel('时间', fontsize=16)
        ax.set_ylabel('温度 (°C)', fontsize=16)
        
        # 图例
        ax.legend(fontsize=14, loc='upper right')
        
        # 网格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 格式化时间轴（适配10天跨度）
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        
        # 10天跨度：按天显示刻度，每1天/2天一个刻度
        seg_total_days = (seg_end - seg_start).days
        if seg_total_days >= 10:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))  # 每2天显示一个刻度
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 不足10天则按天显示
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 仅显示年月日
        
        # 调整布局，避免标签被裁剪
        plt.tight_layout()
        
        # 保存当前10天的图片（命名包含时间段）
        seg_filename = f'temperature_forecast_10days_{seg_idx+1}_{seg_start_str}_{seg_end_str}.png'
        plt.savefig(seg_filename, dpi=300, bbox_inches='tight')
        print(f"✅ 第{seg_idx+1}个10天时间段图片已保存：{seg_filename}")
        
        # 保存当前10天的CSV数据
        seg_csv_filename = f'testset_10days_{seg_idx+1}_{seg_start_str}_{seg_end_str}.csv'
        seg_df.to_csv(seg_csv_filename, index=False, encoding='utf-8')
        
        # 关闭画布释放内存
        plt.close()

    # -------------------------- 7. 保存全量数据CSV --------------------------
    results_df.to_csv('testset_predictions_vs_actual_full.csv', index=False, encoding='utf-8')
    print(f"\n📝 全量预测结果已保存为：testset_predictions_vs_actual_full.csv")
    print(f"\n🎉 绘图完成！生成文件清单：")
    print(f"  1. 全量总览图：{total_plot_filename}")
    print(f"  2. {len(split_segments)} 张10天拆分对比图（文件名含10days标识）")
    print(f"  3. 全量数据CSV + {len(split_segments)} 个10天分段CSV")
