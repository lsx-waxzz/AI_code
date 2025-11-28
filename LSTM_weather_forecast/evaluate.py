import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
from data_load import WeatherDataset
from model import LSTMWeatherForecast

if __name__ == "__main__":
    CSV_PATH = "./data/long_term_forecast/weather/weather.csv"
    LOOK_BACK = 10
    BATCH_SIZE = 32
    MODEL_PATH = "./best_lstm_temp_model.pth"
    
    # 解决matplotlib中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans'] 
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 100  

    # 加载测试数据集
    test_dataset = WeatherDataset(csv_path=CSV_PATH, look_back=LOOK_BACK, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0 
    )
    INPUT_SIZE = len(test_dataset.features)
    print(f"输入特征数：{INPUT_SIZE}")

    # 加载训练好的模型
    try:
        model = LSTMWeatherForecast(input_size=INPUT_SIZE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))) 
        model.eval()
        print(f"已成功加载模型：{MODEL_PATH}")
    except FileNotFoundError:
        print(f"错误：未找到模型文件 {MODEL_PATH}，请先运行train.py训练模型")
        exit(1) 
    except Exception as e:
        print(f"模型加载失败：{str(e)}")
        exit(1)

    # 模型预测（测试集）
    print("\n开始预测测试集温度...")
    all_preds = []
    all_trues = []

    with torch.no_grad(): 
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            # 转换为numpy数组
            all_preds.extend(outputs.cpu().numpy())
            all_trues.extend(batch_y.cpu().numpy())

    # 转换为numpy数组并调整形状
    all_preds = np.array(all_preds).reshape(-1, 1)
    all_trues = np.array(all_trues).reshape(-1, 1)

    # 反归一化，恢复原始温度尺度
    all_preds_original = test_dataset.inverse_transform_temp(torch.tensor(all_preds)).numpy()
    all_trues_original = test_dataset.inverse_transform_temp(torch.tensor(all_trues)).numpy()

    # 计算预测误差指标
    mae = mean_absolute_error(all_trues_original, all_preds_original)
    rmse = np.sqrt(mean_squared_error(all_trues_original, all_preds_original))
    
    print("\n=== 测试集预测误差指标 ===")
    print(f"平均绝对误差（MAE）：{mae:.2f} ℃")
    print(f"均方根误差（RMSE）：{rmse:.2f} ℃")
    
    # 可视化预测结果
    print("\n绘制预测结果图...")
    test_time = test_dataset.time_index.values  # 时间轴

    plt.figure(figsize=(15, 7))  # 放大画布
    # 绘制真实温度和预测温度
    plt.plot(test_time, all_trues_original.flatten(), label='真实温度', color="#1fb435", linewidth=1.8)
    plt.plot(test_time, all_preds_original.flatten(), label='预测温度', color='#ff7f0e', linewidth=1.3, alpha=0.9)
    # 图表样式配置
    plt.xlabel('时间', fontsize=13)
    plt.ylabel('温度（℃）', fontsize=13)
    plt.title(f'LSTM天气温度预测结果（测试集）\nMAE: {mae:.2f}℃ | RMSE: {rmse:.2f}℃', fontsize=15, pad=20)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.gcf().autofmt_xdate()  # 自动旋转x轴时间标签

    # 保存图片
    plt.savefig(
        "./temp_prediction_result.png",
        dpi=300,
        bbox_inches='tight', 
        facecolor='white'     
    )
    plt.show()

    print(f"\n预测完成，结果图已保存至：./temp_prediction_result.png")