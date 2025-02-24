import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak  # 用于获取股票数据
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

# 下载指定股票的数据（以 sh603228 为例）
ticker = 'sh603228'  # 股票代码
data = ak.stock_zh_a_daily(symbol=ticker, start_date='20180101', end_date='20250217')  # 下载指定时间段内的数据

# 将日期列转换为日期格式并设置为索引
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# 删除第7列
data.drop(data.columns[6], axis=1, inplace=True)

# 将 'close' 列移动到最后一列
close_column = data.pop('close')
data['close'] = close_column  # 将 'close' 列添加到最后

# 可视化并保存图片
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['close'], label='Close Price')
plt.title(f'Stock Price: {ticker}')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.savefig('datareview.png', dpi=300)  # 保存为300 PPI的高质量图片

# 提取特征和目标
X = data.iloc[:, :].values  # 特征是所有列
y = data.iloc[:, -1:].values   # 目标是最后一列（'close' 列）

# 归一化
scaler_X = MinMaxScaler(feature_range=(-1, 1))  # 特征的归一化
scaler_Y = MinMaxScaler(feature_range=(-1, 1))  # 目标的归一化
X_scaled = scaler_X.fit_transform(X)  # 对特征进行归一化
y_scaled = scaler_Y.fit_transform(y)  # 对目标进行归一化

# 创建训练数据和测试数据
X_train_scl, X_test_scl, y_train_scl, y_test_scl = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

# 创建时间序列数据集（预测1天的股价）
def create_dataset(X, y, time_steps=60, prediction_days=5):
    X_data, y_data = [], []
    for i in range(time_steps, len(X) - prediction_days):  # 保证索引不越界
        X_data.append(X[i - time_steps:i])  # 使用前100行数据作为特征
        y_data.append(y[i:i + prediction_days])  # 预测未来3天的目标值
    return np.array(X_data), np.array(y_data)

# 创建训练数据和测试数据（预测3天）
X_train, y_train = create_dataset(X_train_scl, y_train_scl, time_steps=60, prediction_days=5)
X_test, y_test = create_dataset(X_test_scl, y_test_scl, time_steps=60, prediction_days=5)

# 构建LSTM模型
model = Sequential()
# 第一层 LSTM，返回序列
model.add(LSTM(units=60, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.1))
# 第二层 LSTM，不返回序列
model.add(LSTM(units=60))
# 输出层，预测3个目标值（未来3天的价格）
model.add(Dense(5))  
# 编译模型，使用均方误差作为损失函数，增加平均绝对误差（MAE）作为评估指标
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

# 打印模型摘要查看网络结构
model.summary()

# 训练模型
history = model.fit(X_train, y_train, epochs=20, batch_size=60, validation_data=(X_test, y_test))

# 预测
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 逆归一化预测值
# 注意逆归一化时要考虑维度变化。预测值是3维的，逆归一化时需要reshape为2维
train_pre_actual = scaler_Y.inverse_transform(train_predict)  # 恢复训练集预测值的原始股价
test_pre_actual = scaler_Y.inverse_transform(test_predict)  # 恢复测试集预测值的原始股价
y_test_reshaped = y_test.reshape(y_test.shape[0], y_test.shape[1])
test_actual = scaler_Y.inverse_transform(y_test_reshaped)  # 恢复真实股价

# 初始化为 Python 列表
mae_values = []
mse_values = []
r2_values = []

# 假设 prediction_days 已定义，表示预测的天数
prediction_days = 5  # 举例，假设要预测未来5天

# 循环遍历每一天的预测
for i in range(prediction_days):  # 遍历每一天的预测
    test_pre_actual_single_column = test_pre_actual[:, i]  # 获取第i天所有样本的预测值
    test_actual_single_column = test_actual[:, i]  # 获取第i天所有样本的真实值
    # 计算每一天的 MAE, MSE 和 R²
    mae = mean_absolute_error(test_pre_actual_single_column, test_actual_single_column)
    mse = mean_squared_error(test_pre_actual_single_column, test_actual_single_column)
    r2 = r2_score(test_pre_actual_single_column, test_actual_single_column)
    # 将每一天的评价指标添加到对应的列表中
    mae_values.append(mae)
    mse_values.append(mse)
    r2_values.append(r2)
    # 可视化第i天的预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(test_actual_single_column, color='blue', label='Actual Price')  # 蓝色线为真实股价
    plt.plot(test_pre_actual_single_column, color='red', label='Predicted Price')  # 红色线为预测股价
    plt.title(f'Stock Price Prediction for Day {i+1}')  # 标题为预测的天数
    plt.xlabel('Index')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(f'test_{i+1}.png', dpi=300)  # 保存每一天的图片

# 将 mae_values, mse_values 和 r2_values 转换为 numpy 数组
mae_values = np.array(mae_values)
mse_values = np.array(mse_values)
r2_values = np.array(r2_values)

# 输出最终的 MAE, MSE 和 R²
print("Final MAE values for each day:", mae_values)
print("Final MSE values for each day:", mse_values)
print("Final R² values for each day:", r2_values)


