#all the imports needed to build the model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import plotly
import plotly.express as px
#Models from Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.metrics import plot_roc_curve
import pickle 

df = pd.read_csv("fetal_health.csv")


#split data into X and y

X = df.drop("fetal_health", axis = 1)
y = df["fetal_health"]



#split data into train and test sets
np.random.seed(50)

# Split into train and test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Put models in a dictionary
models = {"Logistic Regression": LogisticRegression(),
         "KNN": KNeighborsClassifier(),
         "Random Forest": RandomForestClassifier()}

#Create a function to fit and socre models
def fit_score (models, x_train, x_test, y_train, y_test):
    """ 
    Fits and evaluates machine learning models.
    models: dictionary of different machine learning models
    x_train: training data
    x_test: testing data
    y_train: training labels
    y_test: test labels
    """
    
    #set seed
    np.random.seed(50)
    
    #list to make a dictionary to keep model scores
    model_scores = {}
    #loop though models
    for name, model in models.items():
        #fit model to data
        model.fit(x_train, y_train)
        #evaluate model and append score to model_scores
        model_scores[name] = model.score(x_test, y_test)
    return model_scores

model_scores = fit_score(models = models, 
                        x_train = x_train,
                        x_test = x_test,
                        y_train = y_train,
                        y_test = y_test)
model_scores


#hyperparameter tuning for Random Forest Model. Parameters brought over from initial calculations in Jupyter Notebook.

#hyper parameters using best parameters
gs_rs_rf = RandomForestClassifier(max_depth = 10,
                            min_samples_leaf = 1,
                            min_samples_split = 14,
                            n_estimators = 960)
#set up grid for random forest

# fit grid hyperparameter search model
gs_rs_rf.fit(x_train, y_train)


#make prediction using this model, set variable to be used in matrix on the main page.
y_prediction = gs_rs_rf.predict(x_test)

#saving model to disk to use in web application
clf1 = gs_rs_rf.fit(x_train, y_train)


#scores to compare to cross/val
#accuracy 

accuracy_scr = balanced_accuracy_score(y_true=y_test, y_pred=y_prediction) 
accuracy_scr = np.mean(accuracy_scr)

#precision
preciss = precision_score(y_true=y_test, y_pred=y_prediction, average = 'weighted')
preciss = np.mean(preciss)

#recall
recc = recall_score(y_true=y_test, y_pred=y_prediction, average = 'weighted')
recc = np.mean(recc)

#f1 
f1sc = f1_score(y_true=y_test, y_pred=y_prediction, average = 'macro')
f1sc = np.mean(f1sc)

#some cross validation scores for use in data visualization
#Cross-validated accuracy
cv_acc = cross_val_score(gs_rs_rf, X, y,
                        cv = 5,
                        scoring = "balanced_accuracy")
cv_acc = np.mean(cv_acc)
cv_acc

#Cross-validated precision
cv_precision = cross_val_score(gs_rs_rf, X, y,
                        cv = 5,
                        scoring = "precision_weighted")
cv_precision = np.mean(cv_precision)

cv_precision

#Cross-validated recall
cv_recall = cross_val_score(gs_rs_rf, X, y,
                        cv = 5,
                        scoring = "recall_weighted")
cv_recall = np.mean(cv_recall)

cv_recall

#Cross-validated F1 score
cv_f1 = cross_val_score(gs_rs_rf, X, y,
                        cv = 5,
                        scoring = "f1_macro")
cv_f1 = np.mean(cv_f1)


#feature importances

feat_importances = pd.Series(gs_rs_rf.feature_importances_, index = X.columns)
    
features = gs_rs_rf.feature_importances_

