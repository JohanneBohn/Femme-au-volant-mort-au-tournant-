import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from datetime import time, datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

import xgboost as xgb

def reg_log(
    X,
    y
):
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit()
    odds_ratios = np.exp(model.params)
    print(f'Odds ratios : \n{odds_ratios}')
    return model.summary()

def random_forest(
    X_train,
    X_test,
    y_train,
    y_test
): 
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    return y_pred, y_prob

def plot_roc_auc(
    y_test,
    y_prob_rf,
    y_prob_xgb,
    y_prob_nn
):
    plt.figure(figsize=(6, 5))    
    if y_prob_rf is not None:
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
        roc_auc = auc(fpr_rf, tpr_rf)
        plt.plot(fpr_rf, tpr_rf, color='green', label=f"AUC = {roc_auc:.2f}")
    if y_prob_xgb is not None:
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
        roc_auc = auc(fpr_xgb, tpr_xgb)
        plt.plot(fpr_xgb, tpr_xgb, color='blue', label=f"AUC = {roc_auc:.2f}")
    if y_prob_nn is not None:
        fpr_nn, tpr_nn, _ = roc_curve(y_test, y_prob_nn)
        roc_auc = auc(fpr_nn, tpr_nn)
        plt.plot(fpr_nn, tpr_nn, color='purple', label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbe ROC")
    plt.legend()
    plt.show()
    return None
