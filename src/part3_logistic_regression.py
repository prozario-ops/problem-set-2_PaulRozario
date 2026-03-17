'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: pred_universe, num_fel_arrests_last_year
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression as lr


# Your code here
# Load data
print("Loading part 3 data...")
df_arrests = pd.read_csv("data/df_arrests.csv")

# Check target distribution
print(df_arrests['y'].value_counts())

# Train-test split, create two dataframes from `df_arrests`
df_arrests_train, df_arrests_test = train_test_split(
df_arrests,
    test_size=0.3,
    shuffle=True,
stratify=df_arrests['y'],
    random_state=120
)

print("Train shape:", df_arrests_train.shape)
print("Test shape:", df_arrests_test.shape)

# Features
features = ['current_charge_felony', 'num_fel_arrests_last_year']

X_train = df_arrests_train[features]
y_train = df_arrests_train['y']
X_test = df_arrests_test[features]

# Fill na values
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Parameter grid
param_grid = {'C': [0.1, 1, 10]}

#model
lr_model = lr(max_iter=1000)

# 5 fold cross validation setup
cv = KFold_strat(n_splits=5, shuffle=True, random_state=120)

# Grid search
gs_cv = GridSearchCV(
estimator=lr_model,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy'
)

# Fit model
gs_cv.fit(X_train, y_train)

# Best C value
best_C_value = gs_cv.best_params_['C']
print("Best C value from grid search:", best_C_value)

# Interpretation
if best_C_value == min(param_grid['C']):
    print("This is the strongest regularization option.")
elif best_C_value == max(param_grid['C']):
    print("This is the weakest regularization option.")
else:
    print("This is the middle regularization option.")

# Predictions
df_arrests_test['pred_lr'] = gs_cv.predict(X_test)
print("Prediction Stats:")
print(df_arrests_test['pred_lr'].value_counts())

# Save outputs
df_arrests_train.to_csv("data/df_arrests_train.csv", index=False)
df_arrests_test.to_csv("data/df_arrests_test.csv", index=False)

