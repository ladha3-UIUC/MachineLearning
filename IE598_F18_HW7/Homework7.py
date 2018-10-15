import numpy as np
from sklearn.model_selection import train_test_split
import sys
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from tabulate import tabulate

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

in_sample=[]
out_sample=[]
in_sample_cross_val=[]
for k in range(1,10,1):
    temp_in_sample=[]
    temp_out_sample=[]
    temp_in_sample_cross_val=[]
    temp_in_sample.append(k)
    temp_out_sample.append(k)
    temp_in_sample_cross_val.append(k)
    forest=RandomForestClassifier(criterion='gini', n_estimators=k, random_state=1, n_jobs=-1)
    forest.fit(X_train,y_train)
    y_train_pred=forest.predict(X_train)
    y_test_pred=forest.predict(X_test)
    temp_in_sample.append(metrics.accuracy_score(y_train,y_train_pred))
    temp_out_sample.append(metrics.accuracy_score(y_test,y_test_pred))
    temp_in_sample_cross_val.append(cross_val_score(forest,X_train,y_train,cv=10,n_jobs=-1))
    in_sample.append(temp_in_sample)
    out_sample.append(temp_out_sample)
    in_sample_cross_val.append(temp_in_sample_cross_val)
    
print tabulate(in_sample, headers=['Value of n_estimator(from 1 to 10)','In-sample Accuracy Score-RandomForest'])
print tabulate(out_sample, headers=['Value of n_estimator(from 1 to 10)','Out-sample Accuracy Score-RandomForest'])
print tabulate(in_sample_cross_val, headers=['Value of n_estimator(from 1 to 10)','In-sample Cross Val Accuracy Score-RandomForest'])
print("\n")
in_sample=[]
out_sample=[]
in_sample_cross_val=[]   
for k in range(20,110,10):
    temp_in_sample=[]
    temp_out_sample=[]
    temp_in_sample_cross_val=[]
    temp_in_sample.append(k)
    temp_out_sample.append(k)
    temp_in_sample_cross_val.append(k)
    forest=RandomForestClassifier(criterion='gini', n_estimators=k, random_state=1, n_jobs=-1)
    forest.fit(X_train,y_train)
    y_train_pred=forest.predict(X_train)
    y_test_pred=forest.predict(X_test)
    temp_in_sample.append(metrics.accuracy_score(y_train,y_train_pred))
    temp_out_sample.append(metrics.accuracy_score(y_test,y_test_pred))
    temp_in_sample_cross_val.append(cross_val_score(forest,X_train,y_train,cv=10,n_jobs=-1))
    in_sample.append(temp_in_sample)
    out_sample.append(temp_out_sample)
    in_sample_cross_val.append(temp_in_sample_cross_val)
    
print tabulate(in_sample, headers=['Value of n_estimator(from 20 to 100)','In-sample Accuracy Score-RandomForest'])
print tabulate(out_sample, headers=['Value of n_estimator(from 20 to 100)','Out-sample Accuracy Score-RandomForest'])
print tabulate(in_sample_cross_val, headers=['Value of n_estimator(from 20 to 100)','In-sample Cross Val Accuracy Score-RandomForest'])
print("\n")
#Assessing feature importance

feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000,
                              random_state=0,
                              n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
   print("%2d) %-*s %f" % (f + 1, 30, 
                           feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
         color='lightblue', 
        align='center')
plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

 
print("My name is Rishabh Ladha")
print("My NetID is: ladha3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

