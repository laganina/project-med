import numpy as np
from sklearn.datasets import load_iris
from podaci_prvi import obelezja, labela 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

# ZVANICNA SKLEARN DOKUMENTACIJA PREPORUCUJE OVAKO.
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html


print('ulazi: ')
print(obelezja)
print(obelezja.info())

y = labela
y = y.values.ravel()
klase = np.unique(y)
print(f'klase: {klase}')

for i in range(len(klase)):
    broj_uzoraka = sum(y == klase[i])
    print(f'Broj uzoraka u {i}-toj klasi je: {broj_uzoraka}')

# probaj da skaliras obelezja:
# select the columns to exclude from standard scaling
int_cols = obelezja.select_dtypes(include=['int'])
# create a list of columns to scale
cols_to_scale = [col for col in obelezja.columns if col not in int_cols]
# create a StandardScaler instance
scaler = StandardScaler()
# fit and transform the dataframe
obelezja[cols_to_scale] = scaler.fit_transform(obelezja[cols_to_scale])



X = obelezja.to_numpy()
# y = y.to_numpy()               # vec je konvertovano 

# iris = load_iris()
# target_names = iris.target_names

# X, y = iris.data, iris.target
# X, y = X[y != 2], y[y != 2]
n_samples, n_features = X.shape
# We also add noisy features to make the problem harder.

random_state = np.random.RandomState(0)
# X = np.concatenate([X, random_state.randn(n_samples, 200 * n_features)], axis=1)

# Here we run a SVC classifier with cross-validation and plot the ROC curves fold-wise. Notice that 
# the baseline to define the chance level (dashed ROC curve) is a classifier that would always predict 
# the most frequent class.

import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

cv = StratifiedKFold(n_splits=5)
classifier = LogisticRegression()

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(6, 6))    # ima samo veze sa velicinom 
for fold, (train, test) in enumerate(cv.split(X, y)):
    print(f'Fold: {fold}')
    classifier.fit(X[train], y[train])
    print(train)
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X[test],
        y[test],
        name=f"РКП парт. {fold}",
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
ax.plot([0, 1], [0, 1], "k--", label="тзв. ниво шансе (AUC = 0.5)")


mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Средња РКП (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 стд",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    xlabel="Стопа лажних позитива",
    ylabel="Стопа истинских позитива",
    title=f"Средња РКП (ROС-крива) логистичког регресора са варијабилностима\n(ознака позитива: '{1}')",
)
ax.axis("square")
ax.legend(loc="lower right")


plt.savefig('ROC curve Log Reg.png')   
plt.show()