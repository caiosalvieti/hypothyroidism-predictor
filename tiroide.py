# LIBRARIES
import pandas as pd # for data frame operations
import matplotlib.pyplot as plt # for plots
import numpy as np # for math
import seaborn as sns # another way to make plots
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression


#DATA
dtthyroid = pd.read_csv("/Users/caiosalvieti/Downloads/hypothyroid.data")
dtt = dtthyroid
dtt = dtt.rename(columns={'hypothyroid': 'diseases','72': 'age','M': 'sex','f': 'on_thyroxine','f.1': 'query_on_thyroxine','f.2': 'on_antithyroid_medication','f.3': 'thyroid_surgery','f.4': 'query_hypothyroid','f.5': 'query_hyperthyroid','f.6': 'pregnant','f.7': 'sick','f.8': 'tumor','f.9': 'lithium','f.10': 'goitre','y': 'TSH_measured','30': 'TSH','y.1': 'T3_measured','0.60': 'T3','y.2': 'TT4_measerued','15': 'TT4','y.3': 'T4U_measured','1.48': 'T4U','y.4': 'FTI_measured','10': 'FTI','n': 'TBG_measured','?': 'TBG'})
dtt = dtt.replace("?", np.nan)


# DICIONARY = THE BEST WAY FAST, QUICK AND SHORT WHY NOT?
# KEY:VALUE, KEY:VALUE
# .REPLACE WITH  NEW VARIABLE = TO DTT


# Convert categories
dttr = {'hypothyroid': 1, 'negative': 0, 'F': 1, 'M': 0, 'f': 0, 't': 1, 'y': 1, 'n': 0}
dtt = dtt.replace(dttr)

# Delete useless columns
dtt = dtt.drop(['TBG_measured','query_hypothyroid','query_hyperthyroid','pregnant','sick','tumor','sex','lithium','goitre','on_thyroxine','query_on_thyroxine','on_antithyroid_medication','thyroid_surgery','query_on_thyroxine','TSH_measured','TBG','T3_measured','TT4_measerued','T4U_measured','FTI_measured'], axis=1)
dtt = dtt.dropna(axis=0)

# Target
target = dtt.iloc[:,0]
print(target)

# Final dtt without disease
dtt = dtt.drop(['diseases'],axis=1)
print(dtt)

# Plots
dtt = dtt.apply(pd.to_numeric, errors='coerce')
sns.jointplot(data=dtt, x="TSH", y="TT4", hue="age", palette='viridis')

sns.jointplot(data=dtt, x="TSH", y="T4U", hue="age", palette='viridis')

sns.jointplot(data=dtt, x="T3", y="FTI", hue="age", palette='viridis')

#plt.show()

#pca

print(dtt)
# organization
x = StandardScaler().fit_transform(dtt)

# clean target
target_clean = target[dtt.index]

pca = PCA()
principal_components = pca.fit_transform(x)

# DATAFRAME
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
pca_df['target'] = target_clean.values
print(pca_df)

# Scree plot (explained variance by PCs)
print(pca.explained_variance_)

dfScree = pd.DataFrame({'var':pca.explained_variance_ratio_,'PC':['PC1','PC2','PC3','PC4','PC5','PC6']})
sns.barplot(x='PC',y="var",data=dfScree, color="c").set_title('Fig 2. Component Variance');
#plt.show()

# Loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadingsDF = pd.DataFrame(data=loadings, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'], index=['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI'])
print(loadingsDF)

# SHOW SHOW SHOW
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='target', palette='viridis')
plt.title('PCA - MAIN P.1')
#plt.show()
sns.scatterplot(data=pca_df, x='PC1', y='PC3', hue='target', palette='viridis')
plt.title('PCA - MAIN P.2')
#plt.show()
sns.scatterplot(data=pca_df, x='PC1', y='PC4', hue='target', palette='viridis')
plt.title('PCA - MAIN P.3')
#plt.show()
sns.scatterplot(data=pca_df, x='PC2', y='PC3', hue='target', palette='viridis')
plt.title('PCA - MAIN P.4')
#plt.show()
sns.scatterplot(data=pca_df, x='PC2', y='PC4', hue='target', palette='viridis')
plt.title('PCA - MAIN P.5')
#plt.show()
sns.scatterplot(data=pca_df, x='PC3', y='PC4', hue='target', palette='viridis')
plt.title('PCA - MAIN P.6')
#plt.show()

correlation_matrix = dtt.corr()
print(correlation_matrix)
plt.figure(figsize=(8, 6))  # Adjust figure size as needed
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")  # Use annot=True for values on heatmap
plt.title("Correlation Matrix")
#plt.show()


# Split the data into features (X) and target (y)
X = dtt.iloc[:,[2,3,4]]
print(X)
y = target
print(y)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy test set:", accuracy)

# Nest steps :)

print(confusion_matrix(y_test, y_pred))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

cm_display.plot()
plt.show()

precision = precision_score(y_test, y_pred, average='weighted')
print(precision)

recal = recall_score(y_test, y_pred, average='weighted')
print(recal)

f1 = f1_score(y_test, y_pred, average='weighted')
print(f1)

mcc = matthews_corrcoef(y_test, y_pred)
print(mcc)


# Oversampling
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# split for train and test etc.
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_pred = pd.DataFrame(y_pred, columns=['predicted'])

print("after smote")
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy test set:", accuracy)

print(confusion_matrix[y_test, y_pred])

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

cm_display.plot()
plt.show()

precision = precision_score(y_test, y_pred, average='weighted')
print(precision)

recal = recall_score(y_test, y_pred, average='weighted')
print(recal)

f1 = f1_score(y_test, y_pred, average='weighted')
print(f1)

mcc = matthews_corrcoef(y_test, y_pred)
print(mcc)

# Undersampling
rus = RandomUnderSampler(random_state=0)
X_res, y_res = sm.fit_resample(X, y)

# split for train and test etc.
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_pred = pd.DataFrame(y_pred, columns=['predicted'])

print("after under")
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy test set:", accuracy)


print(confusion_matrix[y_test, y_pred])

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

cm_display.plot()
plt.show()

precision = precision_score(y_test, y_pred, average='weighted')
print(precision)

recal = recall_score(y_test, y_pred, average='weighted')
print(recal)

f1 = f1_score(y_test, y_pred, average='weighted')
print(f1)

mcc = matthews_corrcoef(y_test, y_pred)
print(mcc)

#NEXT SVM, GridSearchcv ,RANDOM FOREST, LOGISTIC REGRESSION, XGBOOST

clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred_svm = clf.predict(X_test)

# Define features and target (use the same ones from your project)
X = dtt[['T3', 'TT4', 'T4U']]
y = target

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up pipeline with scaler + model
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Define the hyperparameter grid
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan']
}

# GridSearchCV setup
grid = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1,   n_jobs=-1)

# Fit the model
grid.fit(X_train, y_train)

# Best parameters and accuracy
print("Best parameters found: ", grid.best_params_)
print("Best cross-validation accuracy: {:.3f}".format(grid.best_score_))

# Evaluate on test set
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, matthews_corrcoef

y_pred_knn = grid.predict(X_test)
print("Test Accuracy: {:.3f}".format(accuracy_score(y_test, y_pred_knn)))
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))
print(matthews_corrcoef(y_test, y_pred_knn))

#RF
# Define your features and target again (or reuse)
X = dtt[['T3', 'TT4', 'T4U']]
y = target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional: RandomForest doesn’t need scaling, but for consistency:
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])

# Hyperparameter grid for RandomForest
param_grid = {
    'rf__n_estimators': [50, 100, 150],
    'rf__max_depth': [None, 5, 10],
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf': [1, 2],
    'rf__criterion': ['gini', 'entropy']
}

# Setup GridSearchCV
grid_rf = GridSearchCV(estimator=pipe, param_grid=param_grid,  cv=5,  scoring='accuracy', verbose=1, n_jobs=-1)

# Fit
grid_rf.fit(X_train, y_train)

# Evaluate
y_pred_rf = grid_rf.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, matthews_corrcoef

print("Best RandomForest params:", grid_rf.best_params_)
print("Cross-validated accuracy:", grid_rf.best_score_)
print("Test set accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("MCC:", matthews_corrcoef(y_test, y_pred_rf))

# Save metrics for both models
results = {
    'KNN': {
        'Accuracy': accuracy_score(y_test, y_pred_knn),
        'MCC': matthews_corrcoef(y_test, y_pred_knn)
    },
    'Random Forest': {
        'Accuracy': accuracy_score(y_test, y_pred_rf),
        'MCC': matthews_corrcoef(y_test, y_pred_rf)
    }
}

# Print a clean comparison table
print("\nModel Performance Comparison:")
print(f"{'Model':<15}{'Accuracy':<12}{'MCC':<12}")
for model, metrics in results.items():
    print(f"{model:<15}{metrics['Accuracy']:<12.3f}{metrics['MCC']:<12.3f}")

models = list(results.keys())
accuracies = [results[m]['Accuracy'] for m in models]
mccs = [results[m]['MCC'] for m in models]

x = range(len(models))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x, accuracies, width, label='Accuracy')
plt.bar([i + width for i in x], mccs, width, label='MCC')

plt.xlabel("Model")
plt.ylabel("Score")
plt.title("Model Comparison: Accuracy and MCC")
plt.xticks([i + width / 2 for i in x], models)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()

# Probabilities
y_proba_knn = grid.predict_proba(X_test)[:, 1]
y_proba_rf = grid_rf.predict_proba(X_test)[:, 1]

#Curve
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

prec_knn, rec_knn, _ = precision_recall_curve(y_test, y_proba_knn)
prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_proba_rf)

auc_knn = auc(fpr_knn, tpr_knn)
auc_rf = auc(fpr_rf, tpr_rf)

# Plot side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ROC
ax1.plot(fpr_knn, tpr_knn, label=f"KNN (AUC = {auc_knn:.2f})")
ax1.plot(fpr_rf, tpr_rf, label=f"RF (AUC = {auc_rf:.2f})")
ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_title("ROC Curve")
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.legend()

# PR
ax2.plot(rec_knn, prec_knn, label="KNN")
ax2.plot(rec_rf, prec_rf, label="Random Forest")
ax2.set_title("Precision-Recall Curve")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.legend()

plt.tight_layout()
plt.show()

#logic
# Pipeline
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000))
])

# Hyperparameter grid
param_grid_lr = {
    'lr__C': [0.01, 0.1, 1, 10],
    'lr__penalty': ['l2'],
    'lr__solver': ['lbfgs']  
}

# Grid search
grid_lr = GridSearchCV(pipe_lr, param_grid=param_grid_lr, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_lr.fit(X_train, y_train)

# Predictions
y_pred_lr = grid_lr.predict(X_test)
y_proba_lr = grid_lr.predict_proba(X_test)[:, 1]

# Evaluation
print("Best LR params:", grid_lr.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("MCC:", matthews_corrcoef(y_test, y_pred_lr))

