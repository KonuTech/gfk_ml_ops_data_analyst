import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import itertools


def get_monthly_stability_chart(col, data_frame, date_column, column_to_group_by, column_to_count):
    """
    :param col:
    :param data_frame:
    :param date_column:
    :param column_to_group_by:
    :param column_to_count:
    :return:
    """
    from contextlib import redirect_stdout

    data_frame['month_year'] = data_frame[date_column].dt.to_period('M')

    total = data_frame.groupby([data_frame['month_year'], column_to_group_by])[column_to_count].count().reset_index()
    total['group'] = total['month_year'].astype(str) + " " + total[column_to_group_by].astype(str)

    rate = data_frame[data_frame[column_to_count] == 1].groupby([data_frame['month_year'], column_to_group_by])[column_to_count].sum().reset_index()
    rate['group'] = rate['month_year'].astype(str) + " " + rate[column_to_group_by].astype(str)

    output = pd.merge(rate[['group', column_to_count]], total, how='right', left_on='group', right_on='group')
    output.rename(columns={column_to_count + '_x': "predict_1", column_to_count + '_y': "predict_total"}, inplace=True)
    output['rate'] = output['predict_1'] / output['predict_total']
    output['rate_total'] = 1.0
    # print(output, '\n', "DF SHAPE: ", output.shape)

    with open('output/rates_output.txt', 'w') as f:
        with redirect_stdout(f):
            print(output.to_string())

    for group in set(output[column_to_group_by]):

        plt.figure(figsize=(10, 2))

        # legend
        top_bar = mpatches.Patch(color='darkblue', label='0')
        bottom_bar = mpatches.Patch(color='lightblue', label='1')
        plt.legend(handles=[top_bar, bottom_bar])

        print("PRODUCED A CHART OF " + str(column_to_count) + " MONTHLY STABILITY FOR: ", str(col), " VARIABLE CLASS: ", group)
        bar1 = sns.barplot(
            x='month_year',
            y="rate_total",
            data=output[output[column_to_group_by] == group],
            color='darkblue',
            # alpha=0.5
        )

        bar2 = sns.barplot(
            x='month_year',
            y="rate",
            data=output[output[column_to_group_by] == group],
            color='lightblue',
            alpha=0.5
        )

        plt.savefig("output/charts/monthly_stability/" + str(column_to_count) + "/" + str(col) + "/" + "CLASS_" + str(group) + "_monthly_stability_grouped" + '.jpg')
