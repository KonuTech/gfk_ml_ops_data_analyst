import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

def get_stacked_bar_chart(data_frame, date_column, column_to_group_by, column_to_count):
    """
    :param data_frame: 
    :param date_column: 
    :param column_to_group_by: 
    :param column_to_count: 
    :return: 
    """

    data_frame['month_year'] = data_frame[date_column].dt.to_period('M')
    total = data_frame.groupby([data_frame['month_year'], column_to_group_by])[column_to_count].count().reset_index()
    #     print(total)

    rate = data_frame[data_frame[column_to_count] == 1].groupby([data_frame['month_year'], column_to_group_by])[column_to_count].sum().reset_index()
    #     print(rate)

    try:
        rate['rate'] = [i / j * 100 for i, j in zip(rate[column_to_count].fillna(0), total['predict_automatch'].fillna(0))]
    except ZeroDivisionError:
        rate['rate'] = 0

    rate['group'] = rate['month_year'].astype(str) + " " + rate[column_to_group_by].astype(str)
    # print(rate)

    try:
        total['total_rate'] = [i / j * 100 for i, j in zip(total[column_to_count].fillna(0), total[column_to_count].fillna(0))]
    except ZeroDivisionError:
        total['total_rate'] = 0

    total['group'] = total['month_year'].astype(str) + " " + total[column_to_group_by].astype(str)
    # print(total)

    for group in set(total[column_to_group_by]):

        plt.figure(figsize=(10, 2))

        # legend
        top_bar = mpatches.Patch(color='darkblue', label='No')
        bottom_bar = mpatches.Patch(color='lightblue', label='Yes')
        plt.legend(handles=[top_bar, bottom_bar])

        print('\n', group)
        bar1 = sns.barplot(x='month_year',  y="total_rate", data=total[total[column_to_group_by] == group], color='darkblue')
        bar2 = sns.barplot(x='month_year',  y="rate", data=rate[rate[column_to_group_by] == group], color='lightblue')

        plt.show()