import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns 
import matplotlib.pyplot as plt 
%matplotlib inline

th_data = pd.read_csv('thyroidDF.csv')

th_data.describe()
tf_variables = [
    'sex',
    'on_thyroxine',
    'query_on_thyroxine',
    'on_antithyroid_meds',
    'sick',
    'pregnant',
    'thyroid_surgery',
    'I131_treatment',
    'query_hypothyroid',
    'query_hyperthyroid',
    'lithium',
    'goitre',
    'tumor',
    'hypopituitary',
    'psych',
    'TSH_measured',
    'T3_measured',
    'TT4_measured',
    'T4U_measured',
    'FTI_measured',
    'TBG_measured'
]

num_variables = ['age','TSH','T3','T4U','FTI','TBG']

#handling missing data
th_data.isnull().sum()
isnull_colnames = ['TSH','T3','T4U','FTI','TBG']
for col in isnull_colnames:
    assert th_data.loc[
            (th_data[f'{col}_measured']=='t') 
            & (th_data[col].isnull())
        ].shape[0] == 0 \
            , f'Other missing values in column {col}'


sns.displot(x=th_data[th_data.TSH>0].TSH, log_scale=True)
sns.displot(x=th_data[th_data.T3>0].T3, log_scale=True)
sns.displot(x=th_data[th_data.T4U>0].T4U, log_scale=True)
sns.displot(x=th_data[th_data.FTI>0].FTI, log_scale=True)
sns.displot(x=th_data[th_data.TBG>0].TBG, log_scale=True)


#try imputing missing sex as women if pregnant = 't'
th_data.loc[(th_data['sex'].isnull())&(th_data['pregnant'] == 't'), 'sex'] = 'F'
th_data = th_data[th_data.age<=100]

target_dict = {
    'hyper':['A','B','C','D'],
    'hypo':['E','F','G','H'],
    'protein':['I','J'],
    'non_thyroid':['K'],
    'therapy':['L','M','N'],
    'anty_thyroid':['O','P','Q'],
    'none':['-'],
    'other':['R','S','T']
}

y_target= th_data[['target']]
for key, value in target_dict.items():
    y_target.loc[y_target.target.isin(value)] = key

target_index = y_target.target.isin(list(target_dict.keys()))
y_full = y_target.loc[target_index]

features = th_data.columns.tolist()
features.remove('target')
x_full = th_data.loc[target_index,features].copy()

#separate test data
x, x_test, y, y_test = train_test_split(
        x_full, y_full, train_size=0.8, test_size=0.2, random_state=0, stratify=y_full)

#separate validation data
x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, train_size=0.8, test_size=0.2, random_state=0, stratify=y)


numeric_features = ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
numeric_transformer = Pipeline(
    steps=[
    ('imputer', KNNImputer()),
    ('log_scaler', PowerTransformer(method='box-cox', standardize=True))]
)

categorical_features = tf_variables + ['referral_source']

categorical_transformer = Pipeline(
    steps=[
        ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore')),
    ]
)

no_transformer = Pipeline(
    steps=[
        ('identity', FunctionTransformer(lambda x: x, feature_names_out='one-to-one'))
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('idnt', no_transformer, ['age'])
    ]
)

clf = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier())
    ]
)


clf.fit(x_train, y_train)
print("model score: %.3f" % clf.score(x_train, y_train))
print("model score: %.3f" % clf.score(x_valid, y_valid))



clf = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ]
)

clf.fit(x_train, y_train.values.ravel())
print("model score: %.3f" % clf.score(x_train, y_train.values.ravel()))
print("model score: %.3f" % clf.score(x_valid, y_valid.values.ravel()))


print("model score: %.3f" % clf.score(x_test, y_test.values.ravel()))