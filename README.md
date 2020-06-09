### short-term fault early-warning by LSTM

#### 超参数

```python
BATCH_SIZE = 32
TIME_STEP = 30
INTPUT_SIZE = 14
LR = 0.001
```

#### 网络结构

##### CNN卷积层

```python
self.cnn = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(INTPUT_SIZE,8,2),
			nn.ReLU(),
			)
```

INTPUT_SIZE表示原始数据特征数量，14个特征，8表示卷积后特征数量，2表示卷积操作的窗口大小，表示对每两条连续数据做一次卷积；

##### LSTM层

```python
self.rnn = nn.LSTM(
			input_size=8,
			dropout=0.5,
			hidden_size=16,
			num_layers=2,
			batch_first=True,
			)
```

input_size=8表示经之前CNN卷积后的特征数量，hidden_size=16表示隐藏层大小，即隐藏层节点数，num_layers=2表示使用两层LSTM网络堆叠；

##### 输出层/全连接层

```python
self.dense= nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(16, 6),
			nn.ReLU(),
			nn.Linear(6,1)
			)
```

将LSTM的输出转化为模型需要的输出，经过两个线性层将LSTM 16维的输出转化成1维输出，表示模型预测出现短期故障的概率。

#### 关于原始数据输入的处理

将电动汽车电池管理系统每半小时记录的数据输入模型，即半小时内的时间序列数据，但是，半小时内记录的数据量不一样，即输入LSTM模型的序列数据长度不同，从0到180多条不等。

##### t4.py-论文中使用版本：

把原始数据截取部分，比如前30条，这样所有时间序列数据都等长，即

sequence length = 30

```python
if len(data) > 30:
				data = data.iloc[:30, 1:15].values.tolist()
```

##### 存在的问题：

1. 半小时内少于30条记录的数据直接被丢弃了；
2. 半小时内多于30条记录的数据，后面记录的数据被丢弃了，比如半小时内最多记录了180多条，则损失了150多条。目前30条只是为保证整体的数据量取得一个值；
3. 网络层次不够。

##### t3.py-可改进的版本：

考虑实际应用，变长时间序列作为输入更具灵活性，所以修改模型结构和训练数据加载方式，在每批数据进入模型之前通过

```python
X = rnn_utils.pad_sequence(X, batch_first=True, padding_value=0)
```

将所有序列填充“0”至这批数据中最长序列的长度，在进入LSTM后再通过

```python
# pack_padded x
x = rnn_utils.pack_padded_sequence(x, length, batch_first=True)
```

压缩到原序列长度，这样就实现模型变长序列输入。

##### 存在的问题：

可能由于修改之前的模型结构的原因，或者数据的序列长度差距太大（约0到180不等），训练效果不是太好，（test loss不明显收敛）而且训练速度很慢。
