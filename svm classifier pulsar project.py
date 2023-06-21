# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 21:44:34 2023

@author: MRUTYUNJAY
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv(r"D:\Downloads\archive (2)\Pulsar.csv")
df.shape
df.head()
col_names=df.columns
col_names
df.columns=df.columns.str.strip()
df.columns
df.columns=['IP Mean','IP Sd','IP Kurtosis','IP Skewness','DM-SNR Mean','DM-SNR Sd','DM-SNR Kurtosis','DM-SNR Skewness','target_class']
df.columns
df['target_class'].value_counts()
df['target_class'].value_counts()/np.float(len(df))
df.info()
df.isnull().sum()
round(df.describe(),2)
plt.figure(figsize=(24,20))
plt.subplot(4,2,1)
fig=df.boxplot(column='IP Mean')
fig.set_title('')
fig.set_ylabel('IP Mean')
plt.subplot(4, 2, 2)
fig = df.boxplot(column='IP Sd')
fig.set_title('')
fig.set_ylabel('IP Sd')

plt.subplot(4, 2, 3)
fig = df.boxplot(column='IP Kurtosis')
fig.set_title('')
fig.set_ylabel('IP Kurtosis')

plt.subplot(4, 2, 4)
fig = df.boxplot(column='IP Skewness')
fig.set_title('')
fig.set_ylabel('IP Skewness')

plt.subplot(4, 2, 5)
fig = df.boxplot(column='DM-SNR Mean')
fig.set_title('')
fig.set_ylabel('DM-SNR Mean')

plt.subplot(4, 2, 6)
fig = df.boxplot(column='DM-SNR Sd')
fig.set_title('')
fig.set_ylabel('DM-SNR Sd')

plt.subplot(4, 2, 7)
fig = df.boxplot(column='DM-SNR Kurtosis')
fig.set_title('')
fig.set_ylabel('DM-SNR Kurtosis')

plt.subplot(4, 2, 8)
fig = df.boxplot(column='DM-SNR Skewness')
fig.set_title('')
fig.set_ylabel('DM-SNR Skewness')

plt.figure(figsize=(24,20))
plt.subplot(4,2,1)
fig=df['IP Mean'].hist(bins=20)
fig.set_xlabel('IP Mean')
fig.set_ylabel('Number of pulsar stars')
plt.subplot(4,2,2)
fig=df['IP Sd'].hist(bins=20)
fig.set_xlabel('IP Sd')
fig.set_ylabel('Number of pulsar stars')
plt.subplot(4,2,3)
fig=df['IP Kurtosis'].hist(bins=20)
fig.set_xlabel('IP Kurtosis')
fig.set_ylabel('Number of pulsar stars')
plt.subplot(4,2,4)
fig=df['IP Skewness'].hist(bins=20)
fig.set_xlabel('IP Skewness')
fig.set_ylabel('Number of pulsar stars')
plt.subplot(4,2,5)
fig=df['DM-SNR Mean'].hist(bins=20)
fig.set_xlabel('DM-SNR Mean')
fig.set_ylabel('Number of pulsar stars')
plt.subplot(4,2,6)
fig=df['DM-SNR Sd'].hist(bins=20)
fig.set_xlabel('DM-SNR Sd')
fig.set_ylabel('Number of pulsar stars')
plt.subplot(4,2,7)
fig=df['DM-SNR Kurtosis'].hist(bins=20)
fig.set_xlabel('DM-SNR Kurtosis')
fig.set_ylabel('Number of pulsar stars')
plt.subplot(4,2,8)
fig=df['DM-SNR Skewness'].hist(bins=20)
fig.set_xlabel('DM-SNR Skewness')
fig.set_ylabel('Number of pulsar stars')
X=df.drop(['target_class'],axis=1)
y=df['target_class']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_train.shape,X_test.shape
cols=X_train.columns
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
X_train=pd.DataFrame(X_train,columns=[cols])
X_test=pd.DataFrame(X_test,columns=[cols])
X_train.describe()
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svc=SVC()
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Model accuracy score with default hyperparameters: {0:0.04f}'. format(accuracy_score(y_test,y_pred)))
svc=SVC(C=100.0)
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Model accuracy score with rbf kernel and c=100.0 : {0:0.04f}'. format(accuracy_score(y_test,y_pred)))
svc=SVC(C=1000.0)
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Model accuracy score with rbf kernel and c=1000.0 : {0:0.04f}'. format(accuracy_score(y_test,y_pred)))
linear_svc=SVC(kernel='linear',C=1.0)
linear_svc.fit(X_train,y_train)
y_pred_test=linear_svc.predict(X_test)
print('Model accuracy score with linear kernel and c=1.0 : {0:0.04f}'. format(accuracy_score(y_test,y_pred_test)))
linear_svc=SVC(kernel='linear',C=100.0)
linear_svc.fit(X_train,y_train)
y_pred=linear_svc.predict(X_test)
print('Model accuracy score with linear kernel and c=100.0 : {0:0.04f}'. format(accuracy_score(y_test,y_pred)))
linear_svc=SVC(kernel='linear',C=1000.0)
linear_svc.fit(X_train,y_train)
y_pred=linear_svc.predict(X_test)
print('Model accuracy score with linear kernel and c=1000.0 : {0:0.04f}'. format(accuracy_score(y_test,y_pred)))
y_pred_train=linear_svc.predict(X_train)
y_pred_train
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_test,y_pred)))
print('Training set score: {:.4f}'. format(linear_svc.score(X_train,y_train)))
print('Test set score: {:.4f}'. format(linear_svc.score(X_test,y_test)))
y_test.value_counts()
null_accuracy=(3306/(3306+274))
print('Null accuracy score: {0:0.4f}'. format(null_accuracy))
poly_svc=SVC(kernel='poly',C=1.0)
poly_svc.fit(X_train,y_train)
y_pred=poly_svc.predict(X_test)
print('Model accuracy score with polynomial kernel and c=1.0 : {0:0.04f}'. format(accuracy_score(y_test,y_pred)))
poly_svc100=SVC(kernel='poly',C=100.0)
poly_svc100.fit(X_train,y_train)
y_pred=poly_svc100.predict(X_test)
print('Model accuracy score with polynomial kernel and c=100.0 : {0:0.04f}'. format(accuracy_score(y_test,y_pred)))
sigmoid_svc=SVC(kernel='sigmoid',C=1.0)
sigmoid_svc.fit(X_train,y_train)
y_pred=sigmoid_svc.predict(X_test)
print('Model accuracy score with sigmoid kernel and c=1.0 : {0:0.04f}'. format(accuracy_score(y_test,y_pred)))
sigmoid_svc100=SVC(kernel='sigmoid',C=100.0)
sigmoid_svc100.fit(X_train,y_train)
y_pred=sigmoid_svc100.predict(X_test)
print('Model accuracy score with sigmoid kernel and c=100.0 : {0:0.04f}'. format(accuracy_score(y_test,y_pred)))
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_test)
print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])
cm_matrix=pd.DataFrame(data=cm,columns=['Actual Positive:1','Actual Negative:0'],index=['Predict Positive:1','Pridict Negative:0'])
sns.heatmap(cm_matrix,annot=True,fmt='d',cmap='YlGnBu')
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_test))
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('Classification error : {0:0.4f}'.format(classification_error))
precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))
recall = TP / float(TP + FN)
print('Recall or Sensitivity : {0:0.4f}'.format(recall))
true_positive_rate = TP / float(TP + FN)
print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
false_positive_rate = FP / float(FP + TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Predicting a Pulsar Star classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()
from sklearn.metrics import roc_auc_score
ROC_AUC = roc_auc_score(y_test, y_pred_test)
print('ROC AUC : {:.4f}'.format(ROC_AUC))
from sklearn.model_selection import cross_val_score
Cross_validated_ROC_AUC = cross_val_score(linear_svc, X_train, y_train, cv=10, scoring='roc_auc').mean()
print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))
from sklearn.model_selection import KFold
kfold=KFold(n_splits=5, shuffle=True, random_state=0)
linear_svc=SVC(kernel='linear')
linear_scores = cross_val_score(linear_svc, X, y, cv=kfold)
print('Stratified cross-validation scores with linear kernel:\n\n{}'.format(linear_scores))
print('Average stratified cross-validation score with linear kernel:{:.4f}'.format(linear_scores.mean()))
rbf_svc=SVC(kernel='rbf')
rbf_scores = cross_val_score(rbf_svc, X, y, cv=kfold)
print('Stratified Cross-validation scores with rbf kernel:\n\n{}'.format(rbf_scores))
print('Average stratified cross-validation score with rbf kernel:{:.4f}'.format(rbf_scores.mean()))
