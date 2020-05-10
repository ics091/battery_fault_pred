import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import random

SEED = 1024
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

BATCH_SIZE = 32
LR = 0.0125
EPOCH = 120

def load_data():
	path = './train.csv'
	data = pd.read_csv(path, header=None)
	X = data.iloc[:,:19]
	X = np.array(X)
	y = data.iloc[:,19]
	y = np.array(y)
	return X, y

def load_data_t():
	path = './test.csv'
	data = pd.read_csv(path, header=None)
	X = data.iloc[:,:19]
	X = np.array(X)
	y = data.iloc[:,19]
	y = np.array(y)
	return X, y

def to_one_zero(X):
	for i in range(len(X)):
		if X[i][0] >= 0.5:
			X[i][0] = 1
		else:
			X[i][0] = 0
	X = X.reshape(-1,)
	return X

class classifi(nn.Module):
	"""docstring for classifi"""
	def __init__(self):
		super(classifi, self).__init__()
		self.linear1 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(19, 32),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(32, 16),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(16,1),
			)

	def forward(self,x):
		out = self.linear1(x)
		return out

if __name__ == '__main__':
	X, y = load_data()
	X1, y1 = load_data_t() # test

	# normalize
	min_max_scaler = preprocessing.MinMaxScaler()
	X = min_max_scaler.fit_transform(X)
	print(X.shape) # (654,19)
	print(y.shape) # (654,1)

	X1 = min_max_scaler.fit_transform(X1)

	# train_test_split & to torch_tensor
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
	print('trian_len:',len(X_train),'test_len:',len(X_test))
	X_train = torch.from_numpy(X_train).float()
	X_test = torch.from_numpy(X_test).float()
	y_train = torch.from_numpy(y_train).float()
	y_test = torch.from_numpy(y_test).float()

	# test
	# X1 = torch.from_numpy(X1).float()
	# print(X1.shape)
	# print(y1.shape) # (73)
	# print(y1)

	# batch data_loader

	torch_dataset = Data.TensorDataset(X_train, y_train)
	loader = Data.DataLoader(
		dataset=torch_dataset,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=2,
		)

	model = classifi()
	optimizer = torch.optim.Adam(model.parameters(), lr=LR)
	# optimizer = torch.optim.SGD(model.parameters(), lr=LR)
	# loss_func = nn.CrossEntropyLoss()
	loss_func = nn.BCEWithLogitsLoss()

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=6, verbose=True)

	for epoch in range(EPOCH):
		for step, (b_x, b_y) in enumerate(loader):
			# print('step:',step,'batch_size',len(b_x))
			b_x = b_x.view(-1,19) # (8,19)
			output = model(b_x)
			b_y = b_y.numpy().reshape(-1,1)
			b_y = torch.from_numpy(b_y)
			loss = loss_func(output, b_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# print('epoch',epoch,'-','step',step,'-','train_loss:',loss.data.numpy())
			# print('output:',F.softmax(output,dim=-1),'\n')
			# print('pred:',torch.max(F.softmax(output,dim=-1), 1)[1].data.numpy())
			# print('train_loss:',loss.data.numpy())
			if step % 20 == 0:
				model.eval()
				with torch.no_grad():
					X_test = X_test.view(-1,19)
					test_output = model(X_test)
					y_test = y_test.numpy().reshape(-1,1)
					y_test = torch.from_numpy(y_test)
					test_loss = loss_func(test_output, y_test)
					true_y = y_test.numpy().reshape(-1,)
					pred_y = to_one_zero(torch.sigmoid(test_output).numpy())
					print('epoch',epoch + 1,'|step',step,'|test_loss: %.5f ' % test_loss.data.numpy(),
						'|AUC: %.5f ' % roc_auc_score(y_test,torch.sigmoid(test_output)), '|accu: %.5f ' % accuracy_score(pred_y, true_y))