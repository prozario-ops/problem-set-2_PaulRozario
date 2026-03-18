'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Save dataframe(s) save as .csv('s) in `data/`
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC

#load data
print("Loading part 4 data...")
df_arrests_train = pd.read_csv("data/df_arrests_train.csv")
df_arrests_test= pd.read_csv("data/df_arrests_test.csv")

features= ['current_charge_felony', 'num_fel_arrests_last_year']
X_train= df_arrests_train[features].fillna(0)
y_train=df_arrests_train['y']
X_test= df_arrests_test[features].fillna(0)

param_grid_dt = {
    'max_depth': [1, 3, 5]
}

#model
dt_model = DTC(random_state=120)

#cross validation
cv = KFold_strat(n_splits=5, shuffle=True, random_state=120)

#grid search
gs_cv = GridSearchCV(
    estimator=dt_model,
param_grid=param_grid_dt,
    cv=cv,
scoring= 'accuracy'
)

#fit
gs_cv.fit(X_train, y_train)

#best depth
best_depth = gs_cv.best_params_['max_depth']
print("Best max_depth:", best_depth)

if best_depth == min(param_grid_dt['max_depth']):
    print("This tree has the most regularization, so it is the simplest.")
elif best_depth == max(param_grid_dt['max_depth']):
    print("This tree has the least regularization, so it is the most complex.")
else:
    print("This is in the middle, so it has moderate regularization.")

#predictions
df_arrests_test['pred_dt'] = gs_cv.predict(X_test)

print("Decision Tree prediction stats:")
print(df_arrests_test['pred_dt'].value_counts())

#save
df_arrests_test.to_csv("data/df_arrests_test.csv", index=False)
