import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

def to_one_zero(X):
	for i in range(len(X)):
		if X[i][0] >= 0.5:
			X[i][0] = 1
		else:
			X[i][0] = 0
	X = X.reshape(-1,)
	return X

y_true = [1,0,1,0,1]
y_pred = [0.1,0.1,0.2,0.1,0.21]
y_pred_onezero = [0,0,0,0,0]
y_true = torch.from_numpy(np.array(y_true)).float()
y_pred = torch.from_numpy(np.array(y_pred)).float()
y_pred_onezero = torch.from_numpy(np.array(y_pred_onezero)).float()

print('|AUC: %.5f ' % roc_auc_score(y_true,y_pred))
print('|logloss: %.5f ' % log_loss(y_true,y_pred))
print('|accu: %.5f ' % accuracy_score(y_true,y_pred_onezero))

# AUC——>越趋于1越好
# logloss——>越小越好
# 数值类型（long/float）不影响结果