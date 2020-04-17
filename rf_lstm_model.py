import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
from sklearn.metrics import r2_score # R square
from sklearn.model_selection import train_test_split

feature_data = pd.read_csv('B0007_feature.csv')
feature_data = feature_data.to_numpy()

feature_B0005 = pd.read_csv('B0005_feature.csv')
feature_B0005 = feature_B0005.to_numpy()
# print(len(feature_data))

x = []
y1 = []
y2 = []

xb5 = []
yb5 = []

# 用随机森林的方法选取特征值 0，2，6，8，10，12，14，16，18，20 （10个特征值被选取）
for i in range(len(feature_data)):
	# for j in range(21):
	# 	feature.append(feature_data[i][j])
	feature = []
	p = 0
	while p < 21:
		if p == 4:
			pass
		else:
			feature.append(feature_data[i][p])
		p += 2
	x.append(feature)
	y1.append(feature_data[i][21])
	y2.append(feature_data[i][22])


for i in range(len(feature_B0005)):
	# for j in range(21):
	# 	feature.append(feature_data[i][j])
	feature = []
	p = 0
	while p < 21:
		if p == 4:
			pass
		else:
			feature.append(feature_B0005[i][p])
		p += 2
	xb5.append(feature)
	yb5.append(feature_B0005[i][21])

x = np.array(x)
y1 = np.array(y1) # RUL
y2 = np.array(y2) # SOH

xb5 = np.array(xb5)
yb5 = np.array(yb5)

# 划分训练集和测试集，70%作为训练集

X_train, X_test, y_train, y_test = train_test_split(x, y1, test_size=0.1, random_state=0)
train_x = X_train.reshape(-1, 1, 10) # 一维的，10个参数的数组
train_y = y_train.reshape(-1, 1, 1)
test_x = X_test.reshape(-1, 1, 10)

train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).float()
test_x = torch.from_numpy(test_x).float()

xb5 = xb5.reshape(-1, 1, 10)
xb5 = torch.from_numpy(xb5).float()

# 建立模型

class lstm(nn.Module):
	def __init__(self, input_size=10, hidden_size=20, output_size=1, num_layer=2):
		super(lstm, self).__init__()
		self.lstm_layer = nn.LSTM(input_size, hidden_size, num_layer)
		self.output = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		x, _ = self.lstm_layer(x)
		s, b, h = x.size()
		x = x.view(s*b, h)
		x = self.output(x)
		x = x.view(s, b, -1)
		return x

model = lstm(10, 20, 1, 2)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# 训练
for t in range(300):
	var_x = Variable(train_x)
	var_y = Variable(train_y)
	# forward
	out = model(var_x)
	loss = criterion(out, var_y)
	# backward
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	if (t + 1) % 10 == 0: # 每10次输出结果
		print('time: {}, Loss: {:.5f}'.format(t + 1, loss.item()))

# 模型预测
model = model.eval()

# x = x.reshape(-1, 1, 10)
# x = torch.from_numpy(x).float()
# var_data_x = Variable(x)
# pred_test = model(var_data_x)
# pred_test = pred_test.view(-1).data.numpy()

# mse_test = mean_squared_error(y2, pred_test)
# r2 = r2_score(y2, pred_test)
# print("MSE = ", mse_test)
# print("r2 score", r2)

# label = np.arange(166)

# plt.plot(label, y2, 'o',label='raw_soh')
# plt.plot(label, pred_test, 'o',label='pred_soh')
# plt.xlabel('cycle')
# plt.ylabel('SOH')
# plt.legend()
# plt.show()

var_test_x = Variable(xb5)
pred_test = model(var_test_x)
pred_test = pred_test.view(-1).data.numpy()

r2 = r2_score(yb5, pred_test)
mse_test = mean_squared_error(yb5, pred_test)
print("r2 score", r2)
print("MSE = ", mse_test)

label = np.arange(len(yb5))
plt.plot(label, yb5, 'o',label='raw_rul')
plt.plot(label, pred_test, 'x',label='pred_rul')
plt.xlabel('cycle')
plt.ylabel('RUL')
plt.legend()
plt.show()