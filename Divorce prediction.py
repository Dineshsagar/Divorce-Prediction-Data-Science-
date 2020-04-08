#Any Issues please get back to me.

import numpy as np
import pandas as pd 
import xgboost as xgb

df=pd.read_csv("divorce.csv",sep=';')
df.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=10,test_size=0.3)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

from sklearn.metrics import confusion_matrix,accuracy_score
cmm=confusion_matrix(y_test,svc.predict(x_test))
cmm
ac=accuracy_score(y_test,svc.predict(x_test))

pg= { 'C':[0.1, 1, 10, 100, 1000], 'gamma' : [1, 0.1, 0.01, 0.001], 'kernel' : ['rbf','linear']}
from sklearn.model_selection import GridSearchCV
gd=GridSearchCV(SVC(),pg,refit=True,verbose=3)
gd.fit(x_train,y_train)

gd.score(x_test,y_test)
