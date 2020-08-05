
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from sklearn.dummy import DummyClassifier

def conveter_OnePre (df):
    #Média geral de satisfação da feature EMP_Sat_OnPrem
    cols_Sat_OnPrem = df.filter(regex='EMP_Sat_OnPrem').columns.tolist()
    df['EMP_Sat_OnPrem_mean'] = round(df[cols_Sat_OnPrem].mean(axis=1),1)
    # dropando as features 
    df = df.drop(cols_Sat_OnPrem, axis=1)
    return df


def conveter_EMP_Sat_Remote (df):
    #Média geral de satisfação da feature EMP_Sat_OnPrem
    cols_EMP_Sat_Remote = df.filter(regex='EMP_Sat_Remote').columns.tolist()
    df['EMP_Sat_Remote_mean'] = round(df[cols_EMP_Sat_Remote].mean(axis=1),1)
    # dropando as features 
    df = df.drop(cols_EMP_Sat_Remote, axis=1)
    return df

def conveter_EMP_Engagement(df):
    #Média geral de satisfação da feature EMP_Sat_OnPrem
    cols_EMP_Engagement = df.filter(regex='EMP_Engagement').columns.tolist()
    df['EMP_Engagement_mean'] = round(df[cols_EMP_Engagement].mean(axis=1),1)
    # dropando as features 
    df = df.drop(cols_EMP_Engagement, axis=1)
    return df


def conveter_Emp_Work_Status(df):
    #Média geral de satisfação da feature Emp_Work_Status
    cols_Emp_Work_Status = df.filter(regex='Emp_Work_Status').columns.tolist()
    df['Emp_Work_Status_mean'] = round(df[cols_Emp_Work_Status].mean(axis=1),1)
    # dropando as features 
    df= df.drop(cols_Emp_Work_Status, axis=1)
    return df


def conveter_Emp_Competitive(df):
    #Média geral de satisfação da feature Emp_Competitive.
    cols_Emp_Competitive = df.filter(regex='Emp_Competitive').columns.tolist()
    df['Emp_Competitive_mean'] = round(df[cols_Emp_Competitive].mean(axis=1),1)
    # dropando as features 
    df= df.drop(cols_Emp_Competitive, axis=1)
    return df

def conveter_Emp_Collaborative(df):
    #Média geral de satisfação da feature Emp_Collaborative.
    cols_Emp_Collaborative = df.filter(regex='Emp_Collaborative').columns.tolist()
    df['Emp_Collaborative_mean'] = round(df[cols_Emp_Collaborative].mean(axis=1),1)
    # dropando as features 
    df = df.drop(cols_Emp_Collaborative, axis=1)
    return df


#funcao preencher por zero. 
def preencher_missing_0(df):
    list_to_null = ['EMP_Sat_OnPrem_mean', 'EMP_Sat_Remote_mean','EMP_Engagement_mean','Emp_Work_Status_mean','Emp_Competitive_mean','Emp_Collaborative_mean']
    df.loc[:,list_to_null] = df.loc[:,list_to_null] .fillna(0)
    return df


def eval_thresh(y_real, y_proba):
    """
    Essa função gera um dataframe com vários thresholds e as respectivas métricas de acuráccia: precision, recall e f1

    """
    recall_score_thresh = []
    precision_score_thresh = []
    f1_score_thresh = []
    for thresh in np.arange(0,1,0.001):
        y_thresh = [1 if x >= thresh  else 0 for x in y_proba ]
        recall_score_thresh.append(recall_score(y_real, y_thresh))
        precision_score_thresh.append(precision_score(y_real, y_thresh))
        f1_score_thresh.append(f1_score(y_real, y_thresh))
    dict_metrics = {'threshold':np.arange(0,1,0.001),'recall_score':recall_score_thresh,\
                    'precision_score':precision_score_thresh,'f1_score':f1_score_thresh}
    df_metrics = pd.DataFrame(dict_metrics)
    return df_metrics 


def naive_classifiers(df_train, df_test, target = None):
  
    """
    Essa função nos mostra a performance de classificadores naive, baseados em regras simples como: 
    -prever a classe mais frequente,
    -prever de forma estratificada, 
    - prever de forma uniforme e 
    - prever a classe minoritária

    """
    strategies = ['most_frequent', 'stratified', 'uniform', 'constant'] 
    metrics  = {}
    for s in strategies: 
        if s =='constant': 
            dclf = DummyClassifier(strategy = s, random_state = 0, constant =1) 
            dclf.fit(df_train.drop(target, axis =1), df_train[target]) 
            y_pred_train = dclf.predict(df_train)
            y_pred_test = dclf.predict(df_test)
            recall_test = recall_score(df_test[target], y_pred_test)
            precision_test = precision_score(df_test[target], y_pred_test)
            f1_test = f1_score(df_test[target], y_pred_test)
            recall_train = recall_score(df_train[target], y_pred_train)
            precision_train = precision_score(df_train[target], y_pred_train)
            f1_train = f1_score(df_train[target], y_pred_train)
            
            metrics[s] = {'recall_test':recall_test,
                        'recall_train':recall_train,
                        'precision_train':precision_train,
                        'precision_test': precision_test,
                        'f1_train':f1_train,
                        'f1_test': f1_test}
        else: 
            dclf = DummyClassifier(strategy = s, random_state = 0) 
            dclf.fit(df_train.drop(target, axis = 1), df_train[target]) 
            y_pred_train = dclf.predict(df_train)
            y_pred_test = dclf.predict(df_test)
            recall_test = recall_score(df_test[target], y_pred_test)
            precision_test = precision_score(df_test[target], y_pred_test)
            f1_test = f1_score(df_test[target], y_pred_test)
            recall_train = recall_score(df_train[target], y_pred_train)
            precision_train = precision_score(df_train[target], y_pred_train)
            f1_train = f1_score(df_train[target], y_pred_train)
            
            metrics[s] = {'recall_test':recall_test,
                        'recall_train':recall_train,
                        'precision_train':precision_train,
                        'precision_test': precision_test,
                        'f1_train':f1_train,
                        'f1_test': f1_test}
    metrics_df = pd.DataFrame.from_records(metrics)
  
    return metrics_df


def plot_metrics(df):

    """
    Essa função plota as métricas que estão no dataframe df (métricas vs thresholds)
    """
    plt.plot(df['threshold'],df['recall_score'], '-.')
    plt.plot(df['threshold'],df['precision_score'], '-.')
    plt.plot(df['threshold'],df['f1_score'],'-.')
    plt.legend(['recall','precision','f1_score'])
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.show()