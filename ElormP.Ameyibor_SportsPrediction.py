#%% md
# # Preprocessing 
#%%
import pandas as pd
import numpy as np 
#%%
legacy = pd.read_csv('/Users/elorm/Downloads/male_players (legacy).csv',low_memory=False,na_values='-')
#%%
columns_to_drop = ['player_id', 'player_url', 'fifa_version', 'fifa_update', 'fifa_update_date', 'short_name', 'long_name', 'dob', 
                   'league_id', 'league_name', 'club_team_id', 'club_name', 'club_position', 'club_jersey_number', 'club_loaned_from', 
                   'club_joined_date', 'club_contract_valid_until_year', 'nationality_id', 'nationality_name', 'nation_team_id', 
                   'nation_position', 'nation_jersey_number', 'real_face', 'player_face_url', 'player_tags', 'player_traits', 'ls',
                   'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 
                   'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk']
legacy.drop(columns_to_drop,axis=1, inplace=True)
#%% md
# Drop Columns With More Than 30% missing values
#%%
L=[]
L_less=[]
for i in legacy.columns:
  if((legacy[i].isnull().sum())<(0.4*(legacy.shape[0]))):
    L.append(i)
  else:
    L_less.append(i)
#%%
legacy=legacy[L]
legacy
#%% md
# Separate Numeric and Non-numeric Data
#%%
numeric_data = legacy.select_dtypes(include = np.number)
non_numeric = legacy.select_dtypes(include = ['object'])
#%% md
# Multivariate Imputation
#%%
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy='mean')
numeric_data = pd.DataFrame(imp.fit_transform(numeric_data), columns=numeric_data.columns)

#%%
legacy.info()
#%%
#Fill all null values with the average values of the column
for i in legacy.columns:
    if legacy[i].dtype == 'float64':
        legacy[i].fillna(legacy[i].mean(),inplace=True)

#%%
legacy.info()
#%% md
# Dealing With Non-Numeric Data
#%%
non_num=[]

for i in legacy.columns:
    if legacy[i].dtype == 'object':
        non_num.append(i)
print(non_num)
#%%
y= non_numeric[non_num]
#%%
factorized_df = y.apply(lambda col: pd.factorize(col)[0])
#%%
for col in non_num:
    legacy[col] = factorized_df[col]
#%%
#Reassigning factorized data to y
y = factorized_df
y
#%%
non_numeric.drop(y,axis=1,inplace=True)
#%%
non_numeric=pd.get_dummies(y).astype(int)
#%%
non_numeric.head()
#%%
X=pd.concat([numeric_data,non_numeric],axis=1)
X
#%% md
# # Feature Engineering
#%%
# Correlation with the dependent variable
overall_corr = legacy.corr()['overall'].abs().sort_values(ascending=False)
overall_corr
#%%
threshold = 0.4
#%%
selected_features = overall_corr[overall_corr > threshold].index

print("The features with the maximum correlation with the dependent variable are: ")
for i in selected_features:
    print(i)
#%%
# New DataFrame with Featured/Selected Subsets
feature_subset = legacy[selected_features]
feature_subset
#%%
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)
#%%
print(y.value_counts())
#%% md
# # Cross Validation Training
#%% md
# Training RandomForestRegressor
#%%
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error
#%%
X = feature_subset.drop(columns=['overall'])
y = feature_subset['overall']

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
scorer = make_scorer(mean_absolute_error, greater_is_better=False)
#%%
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42,n_estimators=30)
rf_cv_scores = cross_val_score(model, X, y, cv=3, scoring=scorer) 

print("Cross-validation scores (MAE):", -rf_cv_scores)
print("Mean cross-validation score (MAE):", -rf_cv_scores.mean())
print("Standard deviation of cross-validation scores (MAE):", rf_cv_scores.std())

#%% md
# Training XGBoostRegressor
#%%
import xgboost as xgb

xgb_model = xgb.XGBRegressor(random_state=42)
xgb_cv_scores = cross_val_score(xgb_model, Xtrain, Ytrain, cv=5, scoring=scorer)

print("XGBoost CV scores (MAE):", -xgb_cv_scores)
print("XGBoost Mean CV score (MAE):", -xgb_cv_scores.mean())
print("XGBoost CV score std (MAE):", xgb_cv_scores.std())
#%% md
# Training GradientBoostingRegressor
#%%
from sklearn.ensemble import GradientBoostingRegressor 

gbr_model = GradientBoostingRegressor(random_state=42)
gbr_cv_scores = cross_val_score(gbr_model, Xtrain, Ytrain, cv=5, scoring=scorer)

print("GradientBoosting CV scores (MAE):", -gbr_cv_scores)
print("GradientBoosting Mean CV score (MAE):", -gbr_cv_scores.mean())
print("GradientBoosting CV score std (MAE):", gbr_cv_scores.std())
#%% md
# # Performance and Fine Tuning
#%%
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

# Defining the parameter grid
param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring=scorer, n_jobs=-1, verbose=2)
grid_search.fit(Xtrain, Ytrain)

#%%
# Retrieving the best model
best_model = grid_search.best_estimator_
print(f"Best parameter is: {grid_search.best_params_}")

# Save the model
import joblib
joblib.dump(best_model, 'best_xgboost_model.pkl')
#%%
# Training the best model on the entire training set
best_model.fit(Xtrain, Ytrain)

# Making predictions on the training set and the test set
Ytrain_pred = best_model.predict(Xtrain)
Ytest_pred = best_model.predict(Xtest)

# Calculating MAE for the training set and the test set
train_mae = mean_absolute_error(Ytrain, Ytrain_pred)
test_mae = mean_absolute_error(Ytest, Ytest_pred)

print(f"Best Model Training MAE: {train_mae}")
print(f"Best Model Test MAE: {test_mae}")
#%% md
# # Test Set
#%%
players_22 = pd.read_csv('/Users/elorm/Downloads/players_22-1.csv',low_memory=False)

columns_to_drop_test = ['sofifa_id','player_url', 'short_name', 'long_name', 'dob', 'league_name', 'club_team_id', 'club_name',
    'club_position', 'club_jersey_number', 'club_loaned_from', 'club_joined', 'club_contract_valid_until',
    'nationality_id', 'nationality_name','club_logo_url','club_flag_url','nation_logo_url','nation_flag_url','nation_team_id', 
    'nation_position', 'nation_jersey_number','goalkeeping_speed','real_face', 'player_face_url', 'player_tags', 'player_traits',
    'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf','rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 
    'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb',
    'cb', 'rcb', 'rb', 'gk']

players_22.drop(columns_to_drop_test,axis=1, inplace=True)
#%% md
# Drop Columns With More Than 30% missing values
#%%
LT=[]
LT_less=[]
for i in legacy.columns:
  if((players_22[i].isnull().sum())<(0.4*(players_22.shape[0]))):
    LT.append(i)
  else:
    LT_less.append(i)

players_22
#%%
numeric_data_test = players_22.select_dtypes(include = np.number)
non_numeric_test = players_22.select_dtypes(include = ['object'])
#%%
imp = SimpleImputer(strategy='mean')
numeric_data_test = pd.DataFrame(imp.fit_transform(numeric_data_test), columns=numeric_data_test.columns)
#%%
players_22.info()
#%%
# Fill missing values
for i in players_22.columns:
    if players_22[i].dtype == 'float64':
        players_22[i].fillna(players_22[i].mean(),inplace=True)

players_22.info()
#%% md
# Dealing with Non-Numeric Data
#%%
non_num_test=[]

for i in players_22.columns:
    if players_22[i].dtype == 'object':
        non_num_test.append(i)
print(non_num_test)
#%%
y= non_numeric_test[non_num_test]
#%%
factorized_dft = y.apply(lambda col: pd.factorize(col)[0])
#%%
for col in non_num_test:
    players_22[col] = factorized_dft[col]
#%%
y = factorized_dft
y
#%%
non_numeric_test.drop(y,axis=1,inplace=True)
#%%
non_numeric_test=pd.get_dummies(y).astype(int)
#%%
X_test = players_22[selected_features]
#%%
y_test = players_22['overall']
X_test = X_test.drop(columns=['overall'])
#%%
best_model = joblib.load('best_xgboost_model.pkl')

# Predictions on the players_22 test set
y_test_pred = best_model.predict(X_test)

# Calculate MAE for the players_22 test set
test_mae_22 = mean_absolute_error(y_test, y_test_pred)
print(f"Model Test MAE on players_22 data: {test_mae_22}")

#%%
