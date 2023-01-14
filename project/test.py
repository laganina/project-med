import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from project import dejta_frejm
from sklearn import svm 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from project.dejta_frejm import merged, y, df1
from sklearn.metrics import precision_score, recall_score, f1_score, \
    confusion_matrix, classification_report, roc_auc_score, plot_roc_curve, RocCurveDisplay

# training and testing
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(merged, y, test_size = 0.2)


clf = svm.SVC(kernel='linear', C=2)
clf.fit(x_train, y_train)


y_pred = clf.predict(x_test)

# it doesn’t tell us anything about the errors our machine learning models make on new data we haven’t seen before
# the same accuracy metrics for two different models may indicate different model performance towards different classes
# in case of imbalanced dataset, accuracy metrics is not the most effective metrics to be used
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)

# in some cases, it may be more important to have a high precision (e.g. in medical diagnosis),
# while in others, a high recall may be more important (e.g. in fraud detection)
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))

# to balance precision and recall, practitioners often use the F1 score, which is a combination of the two metric
# it can be difficult to determine the optimal balance between precision and recall for a given application
# useful measure of the model in the scenarios where one tries to optimize either of precision or recall score
# and as a result, the model performance suffers
print('F1 Score: %.3f' % f1_score(y_test, y_pred))

# a graphical plot that illustrates the diagnostic ability of
# a binary classifier system as its discrimination threshold is varied

# posto puca program zbog Boolean array expected for the condition, not int64, df konvertovan u Boolean array
'''
merged_copy = merged.select_dtypes(include=[np.number]).columns
merged[merged_copy] = merged[merged_copy].astype(bool)



# another classifier
# Separate labels from training data
x = merged #Training data
y   #Prediction label
print(x.shape)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.5, random_state=0)

dfrst = RandomForestClassifier(n_estimators=3, max_depth=4, min_samples_split=6, class_weight='balanced')
ranfor = dfrst.fit(x_train, y_train)
y_pred = ranfor.predict(x_test)

# Create heatmap from the confusion matrix
def createConfMatrix(class_names, matrix):
    class_names=[0, 1]
    tick_marks = [0.5, 1.5]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="Blues", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label'); plt.xlabel('Predicted label')
    plt.yticks(tick_marks, class_names); plt.xticks(tick_marks, class_names)

# Create a confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
createConfMatrix(matrix=cnf_matrix, class_names=[0, 1])

# Classification Report (Alternative)
results_log = classification_report(y_test, y_pred, output_dict=True)
results_df_log = pd.DataFrame(results_log).transpose()
print(results_df_log)

# Compute ROC curve
fig, ax = plt.subplots(figsize=(10, 6))
RocCurveDisplay.from_estimator(ranfor, x_test, y_test, ax=ax)
plt.title('ROC Curve for the Car Price Classifier')
plt.show()

# Calculate probability scores
y_scores = cross_val_predict(ranfor, x_test, y_test, cv=3, method='predict_proba')
# Because of the structure of how the model returns the y_scores, we need to convert them into binary values
y_scores_binary = [1 if x[0] < 0.5 else 0 for x in y_scores]
# Now, we can calculate the area under the ROC curve
auc = roc_auc_score(y_test, y_scores_binary, average="macro")
print(auc) # Be aware that due to the random nature of cross validation, the results will change when you run the code '''
