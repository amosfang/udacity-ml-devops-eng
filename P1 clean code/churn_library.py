# library doc string
"""
This is the Python test for the churn_library.py module.
Artifact produced in images and models folders.
Logs generated in log folder.

Author name: A. Fang
Date: 2023/04/24
"""

# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, classification_report

import os

import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Global variables
TEST_SIZE = .3
RANDOM_STATE = 42

PARAM_GRID = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy']
}


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)

    if 'Attrition_Flag' in df.columns:
        df['Churn'] = df['Attrition_Flag'].replace(
            ['Existing Customer', 'Attrited Customer'], [0, 1])

    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    output_pth = './images/eda/'

    # 1. EDA columns
    hist_lst = ['Churn', 'Customer_Age']
    bar_lst = ['Marital_Status']
    histplot_lst = ['Total_Trans_Ct']

    plt.figure(figsize=(10, 8))
    for col in hist_lst:
        df[col].hist()
        plt.savefig(output_pth + col + '_hist.png')

    for col in bar_lst:
        df[col].value_counts('normalize').plot(kind='bar')
        plt.savefig(output_pth + '{}'.format(col) + '_hist.png')

    for col in histplot_lst:
        sns.histplot(df[col], stat='density', kde=True)
        plt.savefig(output_pth + '{}'.format(col) + '_histplot.png')

    plt.close()

    # 4. Heatmap of feature and target columns
    plt.figure(figsize=(25, 20))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(
        df.corr(),
        mask=mask,
        center=0,
        annot=True,
        fmt='.2f',
        square=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig(output_pth + 'eda_heatmap.png')
    plt.close()

    return None


def encoder_helper(df, category_lst, response=None):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables
            or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    for category_column in category_lst:
        cat_lst = list()
        category_column_grouped = df.groupby(category_column).mean()[response]

        for val in df[category_column]:
            cat_lst.append(category_column_grouped.loc[val])

        df[category_column + '_' + response] = cat_lst

    return df


def perform_feature_engineering(df, response=None):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used
              for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']

    # perform feature engineering
    df = encoder_helper(df, cat_columns, response)

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_' + response,
        'Education_Level_' + response,
        'Marital_Status_' + response,
        'Income_Category_' + response,
        'Card_Category_' + response]

    X, y = pd.DataFrame(), df[response]
    X[keep_cols] = df[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    output_pth = './images/results/'

    model_dict = {'lr': 'Logistic Regression', 'rf': 'Random Forest'}

    data_dict = {
        'lr': [
            y_train_preds_lr, y_test_preds_lr],
        'rf': [
            y_train_preds_rf, y_test_preds_rf]}

    for model_key, model_value in model_dict.items():
        plt.figure(figsize=(10, 10))

        plt.text(.01, 1.25, model_value + ' Train',
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(.01, .05, str(classification_report(y_train, data_dict[model_key][0])), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(.01, 6, model_value + ' Test',
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(.01, .7, str(classification_report(y_test, data_dict[model_key][1])), {
                 'fontsize': 10}, fontproperties='monospace')

        plt.axis('off')
        plt.savefig(output_pth + model_key + '_classification_rpt.png')
        plt.close()

    return None


def feature_importance_plot(cv_model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            cv_model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # 1. Shap feature importance plot
    explainer = shap.TreeExplainer(cv_model.best_estimator_)
    shap_values = explainer.shap_values(X_data)

    plt.figure(figsize=(10, 15))
    shap.summary_plot(shap_values, X_data, plot_type='bar', show=False)
    plt.title('Feature Importances')
    plt.savefig(output_pth + 'shap_summary_plot.png')
    plt.close()

    # 2. Feature importance plot

    importances = cv_model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    plt.figure(figsize=(10, 8))
    plt.title('Feature importance')
    plt.ylabel('Importance')

    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth + 'feature_importance_plot.png')
    plt.close()

    return None


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    rfc = RandomForestClassifier(random_state=RANDOM_STATE)

    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=PARAM_GRID, cv=5)

    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Logistic Regression and Random Forest ROC curves

    y_test_pred_proba_rf = cv_rfc.best_estimator_.predict_proba(X_test)[:, 1]
    y_test_pred_proba_lr = lrc.predict_proba(X_test)[:, 1]

    # Compute the FPR and TPR for each classifier
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_test_pred_proba_rf)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_test_pred_proba_lr)

    # Compute the AUC for each classifier
    auc_rf = roc_auc_score(y_test, y_test_pred_proba_rf)
    auc_lr = roc_auc_score(y_test, y_test_pred_proba_lr)

    plt.figure(figsize=(10, 10))
    plt.plot(
        fpr_rf,
        tpr_rf,
        label=f"Random Forest (AUC = {auc_rf:.2f})",
        alpha=0.8)
    plt.plot(
        fpr_lr,
        tpr_lr,
        label=f"Logistic Regression (AUC = {auc_lr:.2f})",
        alpha=0.8)

    plt.legend()
    plt.title('Receiving Operating Characteristic Curves')
    plt.savefig('./images/results/roc_curves.png')

    feature_importance_plot(cv_rfc, X_test, './images/results/')

    return None