import pandas as pd

df=pd.read_csv('car data.csv')
print(df.shape)
print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())

#checking missing or null values (we have inbuilt function .isnull)

null=df.isnull().sum()
print(df.columns)

final_dataset=df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
print(final_dataset.head())
final_dataset['Current_year']=2021
final_dataset['numberofyears']=final_dataset['Current_year']-final_dataset['Year']
final_dataset.drop(['Year'],axis=1,inplace=True)
final_dataset.drop(['Current_year'],axis=1,inplace=True)

final_dataset=pd.get_dummies(final_dataset,drop_first=True)

correlation=final_dataset.corr()

import seaborn as sns
corr=sns.pairplot(final_dataset)

import matplotlib.pyplot as plt
corrmat=final_dataset.corr()
top_corr_feature=corrmat.index
plt.figure(figsize=(20,20))

g = sns.heatmap(final_dataset[top_corr_feature].corr(),annot=True, cmap="RdYlGn")

#dep, independent
X=final_dataset.iloc[:,1:]
Y=final_dataset.iloc[:,0]

#feature importance

from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,Y)
print(model.feature_importances_)

#plot graph of feature importances for better visualization 
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)

X_train.shape


from sklearn.ensemble import RandomForestRegressor
rf_regressorr=RandomForestRegressor()

###Hyperparameter
import numpy as np
n_estimators=[int(x) for x in np.linspace(start=100, stop=1200, num=12)]
print(n_estimators)

from sklearn.model_selection import RandomizedSearchCV


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)

rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)













