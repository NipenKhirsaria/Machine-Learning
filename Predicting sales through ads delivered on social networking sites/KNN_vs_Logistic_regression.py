
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Importing the dataset
import sklearn.linear_model

df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:, [2, 3]].values   # 400 x 2
Y = df.iloc[:, 4].values        # 400 x 1

sns.pairplot(df,hue='Purchased')
plt.show()


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
#X_train : 300 x 2   X_test : 100 x 2
#Y_train : 300 x 1   Y_test : 100 x 1


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
cls_LR = LogisticRegression(random_state = 0)
cls_LR.fit(X_train, Y_train)


# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
cls_KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean')
cls_KNN.fit(X_train, Y_train)


# Predicting the Test set results
Y_pred_LR = cls_LR.predict(X_test)
Y_pred_KNN = cls_KNN.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_LR = confusion_matrix(Y_test, Y_pred_LR)
cm_KNN = confusion_matrix(Y_test, Y_pred_KNN)


# Precision, Recall & F1 score
from sklearn.metrics import precision_score, recall_score, f1_score
precision_LR = precision_score(Y_test, Y_pred_LR)
recall_LR = recall_score(Y_test, Y_pred_LR)
f1_score_LR = f1_score(Y_test, Y_pred_LR)

precision_KNN = precision_score(Y_test, Y_pred_KNN)
recall_KNN = recall_score(Y_test, Y_pred_KNN)
f1_score_KNN = f1_score(Y_test, Y_pred_KNN)


print('\nLogistic Regression')
print('Confusion Matrix:\n',cm_LR)
print('Precision',precision_LR)
print('Recall',recall_LR)
print('F1 score',f1_score_LR)

print('\nKNN')
print('Confusion Matrix:\n',cm_KNN)
print('Precision',precision_KNN)
print('Recall',recall_KNN)
print('F1 score',f1_score_KNN)


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min()-1, stop = X_set[:, 0].max()+1, step = 0.01),
                     np.arange(start = X_set[:, 1].min()-1, stop = X_set[:, 1].max()+1, step = 0.01))

plt.contourf(X1, X2, cls_LR.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.25, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)


plt.title('(Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# Visualising the Test set results of Logistic Regression and KNN
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.figure(3)
plt.subplot(1,2,1)
plt.contourf(X1, X2, cls_LR.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.3, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

plt.subplot(1,2,2)
plt.contourf(X1, X2, cls_KNN.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.3, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
