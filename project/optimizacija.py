from sklearn.svm import SVC
from skopt import BayesSearchCV    # pip install scikit-optimize
from sklearn.datasets import load_iris
from proba import obelezja, labela
from sklearn.preprocessing import StandardScaler

# In Python, you can use the scikit-learn library to perform Bayesian optimization
#  on the parameters of an SVM model. One way to do this is by using the BayesSearchCV 
#  class from the scikit-optimize library, which is a wrapper around the GridSearchCV 
#  class that uses Bayesian optimization to find the optimal hyperparameters. Here is
#   an example of how to use this class to perform Bayesian optimization on the C and 
#   gamma parameters of an SVM model:

merged = obelezja
y = labela 

# select the columns to exclude from standard scaling     # necemo skalirati INT obelezja,
# jer su to diskretne varijable. FLOAT obelezja hocemo jer ce obuka klasifikatora biti brza 
int_cols = merged.select_dtypes(include=['int'])
# create a list of columns to scale
cols_to_scale = [col for col in merged.columns if col not in int_cols]
# create a StandardScaler instance
scaler = StandardScaler()
# fit and transform the dataframe
merged[cols_to_scale] = scaler.fit_transform(merged[cols_to_scale])


y = y.values.ravel()
X = merged.values



opt = BayesSearchCV(SVC(),
                   {'C': (1e-3, 1e3, 'log-uniform'),
                    'gamma': (1e-3, 1e3, 'log-uniform')},
                   n_iter=32,
                   cv=5,
                   random_state=0)

opt.fit(X, y)


# This will perform Bayesian optimization on the C and gamma parameters of an SVM model using 
# 5-fold cross-validation, with 32 total iterations of optimization. The BayesSearchCV class 
# will use the log-uniform distribution for the C and gamma parameters, which makes sense since
#  these parameters are usually searched on a logarithmic scale.

# The fit() method will return an object that contains the best parameters found by the optimization
#  process, which you can access like this:


print("Best parameters found: ",opt.best_params_)
# Best parameters found:  OrderedDict([('C', 23.563182541616264), ('gamma', 0.001)])
# OVO SU REZULTATI ZA PRVU KLASIFIKACIJU 

# It's important to note that, Bayesian optimization is computationally expensive and can be time-consuming.
#  It's also important to have a good understanding of the underlying assumptions and limitations of the
#   optimization algorithm and the search space of the parameters to be optimized.