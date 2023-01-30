from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from proba import obelezja, labela

y = labela 
y = y.values.ravel()

from sklearn.preprocessing import StandardScaler
# select the columns to exclude from standard scaling
int_cols = obelezja.select_dtypes(include=['int'])
# create a list of columns to scale
cols_to_scale = [col for col in obelezja.columns if col not in int_cols]
# create a StandardScaler instance
scaler = StandardScaler()
# fit and transform the dataframe
obelezja[cols_to_scale] = scaler.fit_transform(obelezja[cols_to_scale])

X = obelezja 

# param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# param_grid = {'C': (1e-3, 1e3, 'log-uniform'),
                    # 'degree': [2,3,4],
                    # 'gamma': (1e-3, 1e3, 'log-uniform'),
                    # 'kernel': ['linear', 'rbf', 'poly']}

param_grid = {'C': (1e-3, 1e3, 'log-uniform'),
                    'gamma': (1e-3, 1e3, 'log-uniform'),
                    'kernel': ['linear']}



grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X, y)
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_
print("Best parameters found: ", best_params)
print("Best estimator found: ", best_estimator)