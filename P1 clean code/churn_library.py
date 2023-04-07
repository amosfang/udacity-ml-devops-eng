# library doc string


# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

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
test_size = .3
random_state = 42


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
        df['Churn'] = df['Attrition_Flag'].replace(['Existing Customer', 'Attrited Customer'], [0, 1])
    
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

    # 1. EDA columns to .hist() or histogram
    hist_lst = ['Churn', 'Customer_Age'] 
 
    plt.figure(figsize=(20, 10))
    for col in hist_lst:
        plt.hist(df[col])
        plt.savefig(output_pth + col + '_hist.png')
    
    # 2. EDA categorical columns to barplot
    bar_lst = ['Marital_Status']
    
    for col in bar_lst:
        df[col].value_counts('normalize').plot(kind='bar')
        plt.savefig(output_pth + '{}'.format(col) + '_hist.png')
            
    # 3. EDA columns to .histplot()
    histplot_lst = ['Total_Trans_Ct']
    
    for col in histplot_lst:
        sns.histplot(df[col], stat='density', kde=True)
        plt.savefig(output_pth + '{}'.format(col) + '_histplot.png')
    
    # 4. Heatmap of feature and target columns
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(output_pth + 'eda_heatmap.png')
    
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

    if response:
        df['Churn'] = response
    
    for cat in category_lst:
        cat_lst = list()
        cat_groups = df.groupby(cat).mean()['Churn']
        
        for val in df[cat]:
            cat_lst.append(cat_groups.loc[val])
        
        df[cat + '_Churn'] = cat_lst

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
    df = encoder_helper(df, cat_columns)
    
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']
    
    X = pd.DataFrame()
    X[keep_cols] = df[keep_cols]
    
    if response:
        y = response
    else:
        y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
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
    data_dict = {'lr': [y_train_preds_lr, y_test_preds_lr], 'rf': [y_train_preds_rf, y_test_preds_rf]}
    
    for model_key, model_value in model_dict.items():
        plt.figure(figsize=(10, 10))
        plt.text(.01, 1.25, str(model_value + ' Train'), {'fontsize': 10}, fontproperties='monospace')
        plt.text(.01, .05, str(classification_report(y_train, data_dict[model_key][0])), {'fontsize': 10},
                 fontproperties='monospace')
        plt.text(.01, 6, str(model_value + ' Test'), {'fontsize': 10}, fontproperties='monospace')
        plt.text(.01, .7, str(classification_report(y_test, data_dict[model_key][1])), {'fontsize': 10},
                 fontproperties='monospace')
        plt.axis('off')
        plt.savefig(output_pth+model_key+'_classification_rpt.png')

    return None
        

def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(X_data)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_data, plot_type='bar', show=False)
    plt.title('Feature Importances')
    plt.savefig(output_pth + 'shap_summary_plot.png')
    
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]
    
    # Create plot
    plt.figure(figsize=(10, 8))
    # Create plot title
    plt.title('Feature importance')
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    
    # Save feature importance plot
    plt.savefig(output_pth + 'feature_importance_plot.png')
    
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

    rfc = RandomForestClassifier(random_state=random_state)
    
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
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
    rfc_model = joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    lr_model = joblib.dump(lrc, './models/logistic_model.pkl')

    # Logistic Regression and Random Forest ROC curves

    fig, ax = plt.subplots(figsize=(10, 8))
    rfc_plot = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=.8)
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    lrc_plot.plot(ax=ax, alpha=.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(['Random Forest', 'Logistic Regression'])
    plt.savefig('./images/results/roc_curves.png')
    
    # Feature importance plots
    feature_importance_plot(cv_rfc, X_train, './images/results/')
    
    return None
