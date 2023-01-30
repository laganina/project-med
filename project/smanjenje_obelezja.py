# There are several ways to perform feature selection for an SVM model in Python. 
# Here are a few popular methods:

# SELEKCIJA OBELEZJA 


from proba import obelezja, labela
from sklearn import svm
from sklearn.feature_selection import RFE
from proba import obelezja, labela
from sklearn.preprocessing import StandardScaler

merged = obelezja
y = labela 
feature_names = list(merged.columns)

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



# PRIMERI:
# 1
# Wrapper Methods: Wrapper methods use the performance of the model to evaluate the 
# importance of each feature. They involve training the model with different subsets
#  of features and selecting the subset that results in the best performance. One popular 
#  wrapper method for SVM is Recursive Feature Elimination (RFE). RFE starts with all features
#   and recursively removes the least important feature until a desired number of features is reached. 
#   Here's an example of how to use RFE to perform feature selection for an SVM model:




# Create the SVM model
clf = svm.SVC(kernel='linear')

# Create the RFE model and select 3 attributes
rfe = RFE(clf, n_features_to_select=10)
rfe = rfe.fit(X, y)

# Summarize the selection of the attributes
print('rfe.support_')
mask = rfe.support_
print(mask)
print('rfe.ranking_')
print(rfe.ranking_)
selected_feature_names = [feature_names[i] for i, selected in enumerate(mask) if selected]
print(f'selected_names: {selected_feature_names}')

# 2
# Filter Methods: Filter methods use a ranking metric to evaluate the importance of each feature. 
# Some popular filter methods include Mutual Information, Variance Threshold, and SelectKBest. 
# Here's an example of how to use SelectKBest to select the top 3 features for an SVM model:

from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif

# Create the SVM model
clf = SVC(kernel='linear')

# Create the SelectKBest object and select 3 features
selector = SelectKBest(f_classif, k=3)
selector.fit(X, y)

# Summarize the selection of the attributes
print('selector.get_support():')
print(selector.get_support())


# 3 
# Embedded Methods: Embedded methods learn feature importance as a byproduct of the model training process.
#  Two popular embedded methods for SVM are LASSO and Ridge. Here's an example of how to use Lasso for
#   feature selection for an SVM model:

from sklearn.svm import SVC
from sklearn.linear_model import Lasso

# Create the SVM model
clf = SVC(kernel='linear')

# Create the Lasso model
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

# Summarize the selection of the attributes
print('lasso.coef_:')
print(lasso.coef_)



# # REDUKCIJA DIMENZIONALNOSTI SA PCA ANALIZOM:
# # 1
# # Principal Component Analysis (PCA): PCA is a linear dimensionality reduction technique that 
# # finds the linear combinations of the features that account for the most variance in the data. 
# # Here's an example of how to use PCA to reduce the dimensionality of a feature space:
# # Copy code
# from sklearn.decomposition import PCA

# # Create the PCA model
# pca = PCA(n_components=3)

# # Fit and transform the feature space
# X_reduced = pca.fit_transform(X)       # POSLE OVO KORISTIS U KROSVALIDACIJI 

# # Once you have fit a principal component analysis (PCA) model to your data, you can use the components_ 
# # attribute of the model to identify which features make the principal components. The components_ attribute 
# # is an array where each row represents a principal component and each column represents a feature. The value
# #  of each cell in the array represents the weight of the corresponding feature in the corresponding principal 
# #  component.

# pca.components_[0]


# #  This will return an array of 10 values representing the weights of the features for the first principal component. 
# #  The feature with the highest absolute weight is considered as the feature that makes the most contribution to the
# #   principal component.

# # You can use the same method to access the weights of the features for the second and third principal components and
# #  so on.

# # It's important to note that the principal components are linear combinations of the original features, so the features
# #  that make the principal components are not necessarily the most important features in the original dataset. And also,
# #   it's important to consider the explained variance of each principal component when interpreting the contribution of 
# #   each feature.
