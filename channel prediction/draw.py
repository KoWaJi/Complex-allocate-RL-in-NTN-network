import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# 假设的CSI数据
raw_data = scipy.io.loadmat('Channel.mat')
sequences, targets = [], []
CSI = raw_data['pow'].reshape(1,len(raw_data['pow'])).flatten()

# 创建横坐标
x_axis = np.arange(0, 10001)  # 从1到10000

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(x_axis, CSI, label='CSI Data')
plt.xlabel('Time Steps')
plt.ylabel('CSI Value')
plt.title('CSI Data Over Time')
plt.legend()
plt.show()
