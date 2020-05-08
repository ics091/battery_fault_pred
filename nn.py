import torch
from torch.nn import Linear
from torch import nn
from torch.autograd import Variable
from nltk import ngrams
import numpy as np
import torch.utils.data as Data

# L1 = Linear(in_features=10,out_features=5,bias=True)
# L2 = Linear(in_features=5,out_features=2,bias=True)
# R = nn.ReLU()

# X = Variable(torch.randn(3,10), requires_grad=True)

# # print(X)

# y = R(L2(L1(X)))

# target = Variable(torch.randn(3,2))

# loss = nn.MSELoss()
# ls = loss(y, target)
# ls.backward()
# print(y)
# print(target)
# print(ls)

# conv = nn.Conv1d(1,1,3,bias = False)
# sample = torch.randn(1,1,7)
# print(sample)
# print(conv.weight)
# print(conv(Variable(sample)))

# input_s = "hello, world!!!"
# # print(list(input_s))
# # print(input_s.split())
# print(list(ngrams(input_s.split(),2)))

X = [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]]

y = [0,1,2,3,4,5]

X = np.array(X)
y = np.array(y)

#print(X)
#print(y)

X = torch.from_numpy(X)
y = torch.from_numpy(y)

torch_dataset = Data.TensorDataset(X, y)
loader = Data.DataLoader(
	dataset=torch_dataset,
	batch_size=2,
	shuffle=True,
	num_workers=2,
	)

if __name__ == '__main__':
	for step, (b_x, b_y) in enumerate(loader):
		print(step,":")
		print('bx:',b_x,'by:',b_y)