import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('loan.csv')
df.isnull().sum()
df1=df.dropna(axis='columns')     #drop only if ALL columns are NaN



use_colms= df1[['issue_d','earliest_cr_line','loan_amnt','funded_amnt','funded_amnt_inv','verification_status','grade',
           'int_rate','total_acc','purpose','total_pymnt_inv','last_pymnt_amnt','term',
           'annual_inc','pymnt_plan','installment','loan_status']]

use_colms = use_colms.iloc[0:10000,:] # selecting only 10000 rows out of 39k

use_colms.installment.unique()

#****************************************************************************************************************
'''
# lets plot the histogram to get detail 
fig = use_colms.loan_amnt.hist(bins=50)
fig.set_title('loan_amount_requwsted')
fig.set_xlabel('loan_amount')
fig.set_ylabel('number of loans')

# lets plot the histogram to get detail 
fig = use_colms.total_acc.hist(bins=50)
fig.set_xlim(0,20)  # limiting the values on x-axis
fig.set_title('loan_amount_requwsted')
fig.set_xlabel('number of users')
fig.set_ylabel('number of loans')

# lets plot the histogram to get detail 
fig = use_colms.total_pymnt_inv.hist(bins=50)
fig.set_title('loan_amount_requwsted')
fig.set_xlabel('payment_done')
fig.set_ylabel('number of loans')

# lets plot the histogram to get detail 
fig = use_colms.last_pymnt_amnt.hist(bins=50)
fig.set_title('loan_amount_requwsted')
fig.set_xlabel('payment_done')
fig.set_ylabel('number of loans')

# lets plot the histogram to get detail 
fig = use_colms.annual_inc.hist(bins=50)
fig.set_title('loan_amount_requwsted')
fig.set_xlabel('anual income')
fig.set_ylabel('number of users')

# lets plot the histogram to get detail 
fig = use_colms.installment.hist(bins=50)
fig.set_xlim(0,600)  # limiting the values on x-axis
fig.set_title('loan_amount_requwsted')
fig.set_xlabel('intalment amont ')
fig.set_ylabel('users ')
'''
#*******************************************************************************************************************

# lets check data of date time from obejet to date time format
use_colms.dtypes

from datetime import datetime
use_colms['issue_d'] = use_colms['issue_d'].apply(lambda x: datetime.strptime(x, '%b-%y'))

use_colms['month']=use_colms['issue_d'].apply(lambda x: x.month)
# number of loans across months
use_colms.groupby('month').month.count()

use_colms['year']=use_colms['issue_d'].apply(lambda x: x.year)
# number of loans across months

use_colms.groupby('year').year.count()

'''
# visulizing  using month
sns.barplot(x='loan_status', y = 'month' ,data = use_colms)
plt.show()
# visulizing  using year
sns.barplot(x='loan_status', y = 'year' ,data = use_colms)
plt.show()
'''

# need to follow same earliest_cr_line

use_colms['earliest_cr_line'] = use_colms['earliest_cr_line'].apply(lambda x: datetime.strptime(x, '%b-%y'))

# can be extracted as same as above for visulization purpoase

#*********************************************************************************************************************************
use_colms.dtypes
use_colms = use_colms.drop('pymnt_plan', axis=1) # dropping the column pymnt_plan whole column consists of no



##################################################################################################################################
use_colms.head()
# filtering only fully paid or charged-off
use_colms = use_colms[use_colms['loan_status'] != 'Current']
use_colms['loan_status'] = use_colms['loan_status'].apply(lambda x: 0 if x=='Fully Paid' else 1)

# converting loan_status to integer type
use_colms['loan_status'] = use_colms['loan_status'].apply(lambda x: pd.to_numeric(x))

# summarising the values
use_colms['loan_status'].value_counts()

##################################################################################################################


# taking only categorical column
cat_col = use_colms[['verification_status','grade','int_rate','purpose','loan_status']]
# The column int_rate is character type, let's convert it to float
############################IMPPRTANT#########################################################################
cat_col['int_rate'] = cat_col['int_rate'].apply(lambda x: pd.to_numeric(x.split("%")[0]))
#cat_col['term'] = cat_col['term'].apply(lambda x: pd.to_numeric(x.split("months")[0]))
##########################################IMPORTANT########################################################
'''
this is used to remove special characterstic like % etc by passing in the argument  of split('%0'[0]),0 helps to turn % to 0 in whole column
by using this function able to remove the special functions like in columns 
ex 36 months , 24 months = we need to remove 'months' and and retain only 36 and 24 
'''


for col in cat_col:
    print(col, ':' ,len(cat_col[col].unique()))
    
'''
if colummn is having more than 40,50,60 or more than 100,200,300 etc unique variables(categories) in column 
then its very difficult to apply the one hot  encoding and again it leads in to multiple variable in the columns 
which leads to 

curse of dimentinality:"The curse of dimensionality refers to various phenomena that arise when 
analyzing and organizing data in high-dimensional spaces (often with hundreds or thousands of dimensions) 
that do not occur in low-dimensional settings such as the three-dimensional physical space of everyday experience"

The curse of dimensionality refers to various phenomena that arise when analyzing and organizing data in high-dimensional 
spaces (often with hundreds or thousands of dimensions)that do not occur in low-dimensional settings such as the three-dimensional
 physical space of everyday experience
'''
# in this kind of situations follow the  there is technique we need to take top 10 (depends on the data set)most frequent
#(sorted in acending order ) categories ""
# for this cases 'Feature Engineering-How to Perform One Hot Encoding for Multi Categorical Variables' , krish naik videos : shown below
'''
top_10 = [x for x in cat_col.purpose.value_counts().sort_values(ascending=False).head(10).index]


for label in top_10:
    cat_col[label] = np.where(cat_col['purpose']==label,1,0)
    
cat_col[['purpose']+top_10].head(5)
    
'''

# removing  the cat columns
use_colms.drop(labels=cat_col, axis="columns", inplace=True)


# creating the dummy variables for categorical columns
cat_col = pd.get_dummies(cat_col)


# replacing the categorical column after creating dummy variables
use_colms=pd.concat([use_colms,cat_col], axis =1)   
use_colms['term'] = pd.get_dummies(use_colms['term'])

use_colms.isnull().sum()

use_colms = use_colms.drop(['month','year','issue_d','earliest_cr_line'], axis = 1)
#####################################################################################################################################

# moving the dependent(loan_status) variable column to last to seperate independent and dependent variable
m= use_colms.drop(['loan_status'], axis =1)
n= use_colms['loan_status']
m=pd.concat([m,n],axis=1)
use_colms = m
##################################################################################################################################
'''
X = use_colms.iloc[:,0:33]
y = use_colms.iloc[:,36:]
'''
# or

y = use_colms.loan_status
X = use_colms.drop('loan_status',axis = 1)

# checking the imbalance data sets in the problem 
predictions = use_colms['loan_status'].value_counts() # here we have 0:7662 and 1 : 1549 . its an imbalanced data set



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Resampling Techniques — Oversample minority class
# concatenate our training data back together
X = pd.concat([X_train, y_train], axis=1)


# separate minority and majority classes
false = X[X.loan_status==0]
true = X[X.loan_status==1]

# upsample minority
from sklearn.utils import resample
false_upsampled = resample(true,
                          replace=True, # sample with replacement
                          n_samples=len(false), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([false, false_upsampled])

# check new class counts
upsampled.loan_status.value_counts()

# seperating in to x train and  y train

# splitting the train and test and comapring the models
y_train = upsampled.loan_status.values
X_train = upsampled.drop('loan_status', axis=1).values
X_test = X_test.values
y_test = y_test.values



# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# pca should be applied after feature scaling
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

'''
# Fittng the Logistic model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state= 0 ).fit(X_train,y_train)


# Fittng the KNN model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier (n_neighbors= 5 ,metric='minkowski', p=2) 
# less neighbors leads to over fitting , p=2 euclidian distance
classifier.fit(X_train,y_train)
'''

# here we are getiing more accuracy in SVM model without cosidering PCA
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

'''
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
'''

# by using  PCA we are getting more accuracy on Random forest model compared to other models
# Fittng the Random forest model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators= 300, criterion ='entropy',random_state= 0 ).fit(X_train,y_train)

'''
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
'''



#######################################################XG Boost#######################################################
# need to install XG boost library

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)


#######################################################XG Boost#######################################################




# predicitng the results
y_pred = classifier.predict(X_test)

# confusion matrix
# confusion matrix is used only for classification model (because it can predict only true or false values)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix (y_test , y_pred) 


# Checking accuracy
from sklearn.metrics import accuracy_score , f1_score , recall_score , precision_score ,classification_report

# accuracy
acc_scr = accuracy_score(y_test, y_pred)

# f1 score
f1_scr = f1_score(y_test, y_pred)
 
#recal score
recal_scr = recall_score(y_test, y_pred) # this in built function of sklearn takes TN (true -ve) for recall


#precsion score
prec_scr = precision_score(y_test , y_pred) # this in built function of sklearn takes TN (true -ve) for precision

# report of precision,recall,f1 
class_report = classification_report(y_test, y_pred) 


#######################################################cross validation#######################################################


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 15) # here will fit the classifier and cross validation is performed
accuracies.mean()
accuracies.std()



#######################################################cross validation#######################################################










# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


