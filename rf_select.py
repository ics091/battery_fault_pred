import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

feature_data = pd.read_csv('B0006_feature.csv')
feature_data = feature_data.to_numpy()

# print(len(feature_data))

X = []
y1 = []
y2 = []

for i in range(len(feature_data)):
	feature = []
	for j in range(21):
		feature.append(feature_data[i][j])
	X.append(feature)
	y1.append(feature_data[i][21])
	y2.append(feature_data[i][22])

X = np.array(X)
y = np.array(y2) # SOH

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

forest = RandomForestClassifier(n_estimators=1000,n_jobs=-1,random_state=0)
# for t in range(5):
forest.fit(X_train, y_train.astype('int'))
importances = forest.feature_importances_
# indices=np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
	#if importances[f] > 0.05:
	print("%2d) %f" % (f, importances[f]))

lable = range(0,21,1)
plt.bar(lable, importances)
plt.xlabel('feature_number')
plt.ylabel('relativity')
plt.xticks(lable)
plt.legend()
plt.show()