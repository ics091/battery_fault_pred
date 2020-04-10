import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable

feature_data = pd.read_csv('B0006_feature.csv')
feature_data = feature_data.to_numpy()

# print(len(feature_data))

x = []
y1 = []
y2 = []

for i in range(len(feature_data)):
	feature = []
	for j in range(21):
		feature.append(feature_data[i][j])
	x.append(feature)
	y1.append(feature_data[i][21])
	y2.append(feature_data[i][22])

x = np.array(x)
y1 = np.array(y1) # RUL
y2 = np.array(y2) # SOH

# 划分训练集和测试集，70%作为训练集
train_size = int(len(x) * 0.7)
test_size = len(x) - train_size
train_X = x[:train_size]
train_Y1 = y1[:train_size]
train_Y2 = y2[:train_size]
test_X = x[train_size:]
test_Y1 = y1[train_size:]
test_Y2 = y2[train_size:]

train_x = train_X.reshape(-1, 1, 21) # 一维的，21个参数的数组
train_y1 = train_Y1.reshape(-1, 1, 1)
train_y2 = train_Y2.reshape(-1, 1, 1)
test_x = test_X.reshape(-1, 1, 21)

train_x = torch.from_numpy(train_x).float()
train_y1 = torch.from_numpy(train_y1).float()
train_y2 = torch.from_numpy(train_y2).float()
test_x = torch.from_numpy(test_x).float()

# 建立模型

class lstm(nn.Module):
	def __init__(self, input_size=21, hidden_size=42, output_size=1, num_layer=2):
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

model = lstm(21, 42, 1, 2)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# 训练
for t in range(300):
	var_x = Variable(train_x)
	var_y = Variable(train_y2)
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

x = x.reshape(-1, 1, 21)
x = torch.from_numpy(x).float()
var_data_x = Variable(x)
pred_test = model(var_data_x)
pred_test = pred_test.view(-1).data.numpy()

label = np.arange(166)

plt.plot(label, y2, 'o',label='raw_soh')
plt.plot(label, pred_test, 'o',label='pred_soh')
plt.xlabel('cycle')
plt.ylabel('SOH')
plt.legend()
plt.show()