import pandas as pd
import numpy as np
import seaborn as sb

df=pd.read_csv('C:\\Users\\DELL\\Desktop\\KITS\\Book2.csv')


import matplotlib.pyplot as plt  

df["GENDER"].value_counts().plot(kind="bar");
plt.xlabel("Gender")
plt.ylabel("number")
plt.title("gender")




from sklearn import preprocessing
string=["REG.NO.", "NAME", "GENDER",	"CATEGORY",	"FOOD HABITS",	"LOCATION",	"WHERE DO YOU STAY?",	"FAMILY STATUS", "SCHOOL TYPE",	"FATHER'S QUALIFICATION",	"STUDENT'S (10TH%)",	"12TH(%)",	"active in social media",	"no. of hrs spending on studies",	"no. of family members"]
number = preprocessing.LabelEncoder()
for i in string:
    df[i]=number.fit_transform(df[i])
df.shape

from sklearn.preprocessing import MinMaxScaler
ns=["REG.NO.", "NAME", "GENDER",	"CATEGORY",	"FOOD HABITS",	"LOCATION",	"WHERE DO YOU STAY?",	"FAMILY STATUS", "SCHOOL TYPE",	"FATHER'S QUALIFICATION",	"STUDENT'S (10TH%)",	"12TH(%)",	"active in social media",	"no. of hrs spending on studies",	"no. of family members"]
for k in ns:
    mm=MinMaxScaler()
    df[ns]=mm.fit_transform(df[ns])
    
from sklearn.model_selection import train_test_split
x=df.iloc[:,:15]
y=df.iloc[:,15]
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=0)# Splitting data into training and testing data set


from sklearn.tree import  DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy')
pred = model.fit(x_train,y_train).predict(x_test)

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
ignb = GaussianNB()
imnb = MultinomialNB()

pred_gnb = ignb.fit(x_train,y_train).predict(x_test)
pred_mnb = imnb.fit(x_train,y_train).predict(x_test)

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics 

model_linear = SVC(kernel = "linear")
pred1 = model_linear.fit(x_train,y_train).predict(x_test)
model_poly = SVC(kernel = "poly")
pred2 = model_poly.fit(x_train,y_train).predict(x_test)
model_rbf = SVC(kernel = "rbf")
pred3 = model_rbf.fit(x_train,y_train).predict(x_test)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=100,criterion="entropy")
pred4 = rf.fit(x_train,y_train).predict(x_test)


from sklearn.metrics import confusion_matrix
from sklearn import metrics 

confusion_matrix1 = confusion_matrix(y_test,pred)
pd.crosstab(y_test.values.flatten(),pred)
print("Decision Tree model accuracy(in %):", metrics.accuracy_score(y_test, pred)*100)
np.mean(pred==y_test.values.flatten())

confusion_matrix(y_test,pred_gnb) 
pd.crosstab(y_test.values.flatten(),pred_gnb)
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, pred_gnb)*100)
np.mean(pred_gnb==y_test.values.flatten())

confusion_matrix(y_test,pred_mnb) 
pd.crosstab(y_test.values.flatten(),pred_mnb)  
print("Multinomial Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, pred_mnb)*100)
np.mean(pred_mnb==y_test.values.flatten())

confusion_matrix(y_test,pred1)
pd.crosstab(y_test.values.flatten(),pred1)
print("SVM Linear model accuracy(in %):", metrics.accuracy_score(y_test, pred1)*100)
np.mean(pred1==y_test.values.flatten())

confusion_matrix(y_test,pred2)
pd.crosstab(y_test.values.flatten(),pred2)
print("SVM Polynomial model accuracy(in %):", metrics.accuracy_score(y_test, pred2)*100)
np.mean(pred2==y_test.values.flatten())

confusion_matrix(y_test,pred3)
pd.crosstab(y_test.values.flatten(),pred3)
print("SVM RBF model accuracy(in %):", metrics.accuracy_score(y_test, pred3)*100)
np.mean(pred3==y_test.values.flatten())

confusion_matrix(y_test,pred4)
pd.crosstab(y_test.values.flatten(),pred4)
print("Random Forest model accuracy(in %):", metrics.accuracy_score(y_test, pred4)*100)
np.mean(pred4==y_test.values.flatten())


df["Result"].value_counts().plot(kind="bar");
plt.xlabel("Result")
plt.ylabel("number")
plt.title("Result")


df.Result.value_counts().plot(kind="pie")
