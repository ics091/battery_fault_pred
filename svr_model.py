import pandas as pd
import numpy as np
import sklearn.model_selection as sm
import sklearn.svm as svm
import sklearn.metrics as mm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
from sklearn.metrics import r2_score # R square

feature_data = pd.read_csv('B0006_feature.csv')
feature_data = feature_data.to_numpy()

# print(len(feature_data))

x = []
y1 = []
y2 = []

# for i in range(len(feature_data)):
# 	feature = []
# 	for j in range(21):
# 		feature.append(feature_data[i][j])
# 	x.append(feature)
# 	y1.append(feature_data[i][21])
# 	y2.append(feature_data[i][22])

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

x = np.array(x)
y1 = np.array(y1)
y2 = np.array(y2)

# print(len(x), len(y1), len(y2))

# print(x)
# x = x.reshape(-1, 1)
# print(x)

train_x, test_x, train_y, test_y = sm.train_test_split(x, y2, test_size=0.3, random_state=10)
print('训练集大小： ', len(train_x))
print('测试集大小： ', len(test_x))

# 基于支持向量机的回归模型
model = svm.SVR(kernel='rbf', C=1e3, gamma=0.001)
model.fit(train_x, train_y)

pred_y2 = model.predict(x)

# 模型得分 r2:拟合优度 越--->1.0越好
score = mm.r2_score(y2, pred_y2)
r2 = r2_score(y2, pred_y2)
mse_test = mean_squared_error(y2, pred_y2)
print('r2得分： ', score)
print("MSE = ", mse_test)
print("r2 score", r2)

label = np.arange(166)

pred_soh = model.predict(x)
# print(pred_soh)

plt.plot(label, y2, 'o',label='raw_soh')
plt.plot(label, pred_soh, 'o',label='pred_soh')
plt.xlabel('cycle')
plt.ylabel('SOH')
plt.legend()
plt.show()