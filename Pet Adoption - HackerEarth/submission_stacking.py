import numpy as np
import pandas as pd
import datetime
df = pd.read_csv("train.csv")
df_sub = pd.read_csv("test.csv")


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')


#To include Turn Around time to the list of independent variables
format = '%Y-%m-%d %H:%M:%S'
df['issue_date'] = [datetime.datetime.strptime(x, format) for x in df['issue_date']]
df['listing_date'] = [datetime.datetime.strptime(x, format) for x in df['listing_date']]

t = pd.DataFrame()
t['TA Time'] = df['listing_date'] - df['issue_date']
t['TA Time'] = t['TA Time']/np.timedelta64(1, 'D')

c = df['color_type']

X = df.iloc[:,[3, 4, 5, 6, 7, 8]].values
t = t.iloc[:].values
X = np.append(X, t, axis = 1)
imputer = imputer.fit(X[:, 0:1])
X[:, 0:1] = imputer.transform(X[:, 0:1])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
ct = ColumnTransformer([('color_type', OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)
Train = X.astype(float)

format = '%Y-%m-%d %H:%M:%S'
df_sub['issue_date'] = [datetime.datetime.strptime(x, format) for x in df_sub['issue_date']]
df_sub['listing_date'] = [datetime.datetime.strptime(x, format) for x in df_sub['listing_date']]

t = pd.DataFrame()
t['TA Time'] = df_sub['listing_date'] - df_sub['issue_date']
t['TA Time'] = t['TA Time']/np.timedelta64(1, 'D')

Test = df_sub.iloc[:,[3,4,5,6,7,8]].values
t = t.iloc[:].values
Test = np.append(Test, t, axis = 1)
Test[:, 0:1] = imputer.transform(Test[:, 0:1])
Test[:, 1] = le.transform(Test[:, 1])
Test = ct.transform(Test)
Test = Test.astype(float)

from sklearn.ensemble import RandomForestClassifier
base_classifier1 = RandomForestClassifier(max_features='sqrt',bootstrap=True,min_samples_leaf=2,min_samples_split=5,criterion='entropy',n_estimators=450)

from sklearn.linear_model import LogisticRegression
base_classifier2 = LogisticRegression(solver = 'lbfgs', max_iter = 150)

from sklearn.ensemble import GradientBoostingClassifier
base_classifier3 = GradientBoostingClassifier(n_estimators = 400)

y_submission = np.empty((0,1),float)
i=3

for t in range(2):
    
    y_Train = df.iloc[:,i+6].values

    y_pred_1 = np.empty((0,i),float)
    y_pred_2 = np.empty((0,i),float)
    y_pred_3 = np.empty((0,i),float)
    y_test_fold = np.empty((0,1))
    
    
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits = 5)
    for train_index, test_index in skf.split(Train, y_Train):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_Train[train_index], y_Train[test_index]
        
        base_classifier1.fit(X_train, y_train)
        y_pred_1 = np.append(y_pred_1,base_classifier1.predict_proba(X_test)).reshape(-1,i)
        
        base_classifier2.fit(X_train, y_train)
        y_pred_2 = np.append(y_pred_2,base_classifier2.predict_proba(X_test)).reshape(-1,i)
        
        base_classifier3.fit(X_train, y_train)
        y_pred_3 = np.append(y_pred_3,base_classifier3.predict_proba(X_test)).reshape(-1,i)
        
        y_test_fold = np.append(y_test_fold, y_test.reshape(-1,1))
        
    meta_train_X = np.concatenate((y_pred_1, y_pred_2, y_pred_3),axis=1)
    meta_train_y = y_test_fold
    meta_test = np.concatenate((base_classifier1.predict_proba(Test), base_classifier2.predict_proba(Test), base_classifier3.predict_proba(Test)),axis=1)
    
    
    from xgboost import XGBClassifier
    meta_classifier = XGBClassifier()
    meta_classifier.fit(meta_train_X, meta_train_y)
    yhat = meta_classifier.predict(meta_test)
    if i == 3:
        y_submission = np.vstack((y_submission, yhat.reshape(-1,1)))
    else:
        y_submission = np.concatenate((y_submission, yhat.reshape(-1,1)), axis=1)
    
    i+=1

sub = np.column_stack((df_sub['pet_id'].values, y_submission))
sub = pd.DataFrame(sub)
sub.columns = ['pet_id', 'breed_category', 'pet_category']
sub.to_csv('submission.csv', index = False, index_label = None)

#s = pd.read_excel("submission_stacking.xlsx")
#s.to_csv('submission_stacking1.csv', index = False, index_label = None)