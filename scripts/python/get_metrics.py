
import numpy as np
import pandas as pd

# POST PROCESSING BIAS METRICS FOR THE TRAINED MODEL

def get_PPL(data_frame, column_to_group_by, column_to_count):
    """
    :param data_frame: 
    :param column_to_group_by: 
    :param column_to_count: 
    :return: 
    """
    
    ppl = []
    
    counts_total = data_frame.groupby([column_to_group_by]) \
    .size() \
    .rename('count_total') \
    .reset_index() \
    .sort_values(by=column_to_group_by, ascending=False) \
    .set_index(column_to_group_by)
    
    
    counts_grouped = data_frame.groupby([column_to_group_by, column_to_count]) \
    .size() \
    .rename('count') \
    .reset_index() \
    .sort_values(by=column_to_group_by, ascending=False) \
    .set_index(column_to_group_by)

    positive_counts = counts_grouped.loc[counts_grouped[column_to_count] == 1]
    df = pd.merge(positive_counts, counts_total, left_index=True, right_index=True)
    df['PPL'] = df['count'] / df['count_total']
    
    return df['PPL']


def get_DPPL(data_frame, column_to_group_by, column_to_count):
    """
    :param data_frame: 
    :param column_to_group_by: 
    :param column_to_count: 
    :return: 
    """
    
    ppl = []
    
    df = get_PPL(data_frame, column_to_group_by, column_to_count)

    for i in df.iteritems():
        ppl.append(i[1])
        
    return sorted(set([i - j for i in ppl for j in ppl if i != j]))


def get_CA(data_frame, column_to_group_by, column_to_count_acctuals, column_to_count_predictions):
    """
    :param data_frame: 
    :param column_to_group_by: 
    :param column_to_count_acctuals: 
    :param column_to_count_predictions: 
    :return: 
    """
    
    counts_total_labels = data_frame.groupby([column_to_group_by]) \
    .size() \
    .rename('count_total') \
    .reset_index() \
    .sort_values(by=column_to_group_by, ascending=False) \
    .set_index(column_to_group_by)
    
    
    counts_grouped_acctuals = data_frame.groupby([column_to_group_by, column_to_count_acctuals]) \
    .size() \
    .rename('count_grouped_acctuals') \
    .reset_index() \
    .sort_values(by=column_to_group_by, ascending=False) \
    .set_index(column_to_group_by)
    
    positive_counts_grouped_acctuals = counts_grouped_acctuals.loc[counts_grouped_acctuals[column_to_count_acctuals] == 1]
    
    
    counts_grouped_predictions = data_frame.groupby([column_to_group_by, column_to_count_predictions]) \
    .size() \
    .rename('count_gruped_predictions') \
    .reset_index() \
    .sort_values(by=column_to_group_by, ascending=False) \
    .set_index(column_to_group_by)

    positive_counts_grouped_predictions= counts_grouped_predictions.loc[counts_grouped_predictions[column_to_count_predictions] == 1]


    df = pd.merge(positive_counts_grouped_acctuals, positive_counts_grouped_predictions, left_index=True, right_index=True)
    df['CA'] = df['count_grouped_acctuals'] / df['count_gruped_predictions']
            
    return df['CA']


def get_DCA(data_frame, column_to_group_by, column_to_count_acctuals, column_to_count_predictions):
    """
    :param data_frame: 
    :param column_to_group_by: 
    :param column_to_count_acctuals: 
    :param column_to_count_predictions: 
    :return: 
    """
    
    ca = []
    
    df = get_CA(data_frame, column_to_group_by, column_to_count_acctuals, column_to_count_predictions)
    
    for i in df.iteritems():
        ca.append(i[1])
            
    return sorted(set([i - j for i in ca for j in ca if i != j]))


def get_CR(data_frame, column_to_group_by, column_to_count_acctuals, column_to_count_predictions):
    """
    :param data_frame: 
    :param column_to_group_by: 
    :param column_to_count_acctuals: 
    :param column_to_count_predictions: 
    :return: 
    """
    
    counts_total_labels = data_frame.groupby([column_to_group_by]) \
    .size() \
    .rename('count_total') \
    .reset_index() \
    .sort_values(by=column_to_group_by, ascending=False) \
    .set_index(column_to_group_by)
    
    
    counts_grouped_acctuals = data_frame.groupby([column_to_group_by, column_to_count_acctuals]) \
    .size() \
    .rename('count_grouped_acctuals') \
    .reset_index() \
    .sort_values(by=column_to_group_by, ascending=False) \
    .set_index(column_to_group_by)
    
    negative_counts_grouped_acctuals = counts_grouped_acctuals.loc[counts_grouped_acctuals[column_to_count_acctuals] == 0]
    
    
    counts_grouped_predictions = data_frame.groupby([column_to_group_by, column_to_count_predictions]) \
    .size() \
    .rename('count_gruped_predictions') \
    .reset_index() \
    .sort_values(by=column_to_group_by, ascending=False) \
    .set_index(column_to_group_by)

    negative_counts_grouped_predictions= counts_grouped_predictions.loc[counts_grouped_predictions[column_to_count_predictions] == 0]


    df = pd.merge(negative_counts_grouped_acctuals, negative_counts_grouped_predictions, left_index=True, right_index=True)
    df['CR'] = df['count_grouped_acctuals'] / df['count_gruped_predictions']
            
    return df['CR']


def get_DCR(data_frame, column_to_group_by, column_to_count_acctuals, column_to_count_predictions):
    """
    :param data_frame: 
    :param column_to_group_by: 
    :param column_to_count_acctuals: 
    :param column_to_count_predictions: 
    :return: 
    """
    
    cr = []
    
    df = get_CR(data_frame, column_to_group_by, column_to_count_acctuals, column_to_count_predictions)
    
    for i in df.iteritems():
        cr.append(i[1])
    
    return sorted(set([i - j for i in cr for j in cr if i != j]))


def get_class_cm(data_frame, acctuals, predictions, column_to_group_by):
    
    data = pd.DataFrame(columns=['Class', 'TN', 'FN', 'FP', 'TP'])
    
    for group in set(data_frame[column_to_group_by]):
        sample = data_frame.loc[data_frame[column_to_group_by] == group]
    
        cm = confusion_matrix(sample[acctuals], sample[predictions].notnull())
        TN, FN, FP, TP = cm[0][0], cm[1][0], cm[0][1], cm[1][1]
        
        data = data.append({'Class': group, 'TN': TN, 'FN': FN, 'FP': FP, 'TP': TP}, ignore_index=True)
        
    return data
