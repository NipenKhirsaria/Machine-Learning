
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


Y_pred_new = np.zeros((len(Y_test),15))
error = np.zeros((len(Y_test),15))
f1 = np.zeros(15)
for i in range(1,16):
    cls = KNeighborsClassifier(n_neighbors = i, metric = 'minkowski', p = 2)
    cls.fit(X_train, Y_train)

    Y_pred = cls.predict(X_test)
    Y_pred_new[:,i-1] = Y_pred
    error[:,i-1] = (Y_pred_new[:,i-1] - Y_test)**2        # Squared error

    f1[i-1] = f1_score(Y_test, Y_pred)


K = np.linspace(1,15,15)       # Array containing No. of neighbours
sum_error = sum(error)         # Sum of squared error

plt.figure(1)
plt.subplot(1,2,1)
plt.plot(K,sum_error,'ro-')
plt.title('Error vs no. of neighbours (k)')
plt.xlabel('No. of neighbours (k)')
plt.ylabel('Error')

plt.subplot(1,2,2)
plt.plot(K,f1,'ro-')
plt.title('F1 Score vs no. of neighbours (k)')
plt.xlabel('No. of neighbours (k)')
plt.ylabel('F1 Score')

plt.show()

