import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import itertools


def get_monthly_stability_chart(col, data_frame, date_column, column_to_group_by, column_to_count):
    """
    :param data_frame:
    :param date_column:
    :param column_to_group_by:
    :param column_to_count:
    :return:
    """
    from contextlib import redirect_stdout

    rate_longest = pd.DataFrame()

    data_frame['month_year'] = data_frame[date_column].dt.to_period('M')
    total = data_frame.groupby([data_frame['month_year'], column_to_group_by])[column_to_count].count().reset_index()

    rate = data_frame[data_frame[column_to_count] == 1].groupby([data_frame['month_year'], column_to_group_by])[column_to_count].sum().reset_index()
    rate_longest['rate'] = [(i / j * 100) if (i not in (None, 0) and j not in (None, 0)) else 0 for i, j in itertools.zip_longest(rate[column_to_count], total[column_to_count])]
    rate_longest['group'] = total['month_year'].astype(str) + " " + total[column_to_group_by].astype(str)
    rate_longest[column_to_count] = rate[column_to_count]
    #     print(rate, '\n', "DF SHAPE: ", rate.shape)

    total['total_rate'] = [(i / j * 100) if (i not in (None, 0) and j not in (None, 0)) else 100 for i, j in itertools.zip_longest(total[column_to_count], total[column_to_count])]
    total['group'] = total['month_year'].astype(str) + " " + total[column_to_group_by].astype(str)
    #     print(total, '\n', "DF SHAPE: ", total.shape)

    # output = pd.merge(total, rate_longest, how='left', left_on='group', right_on='group')
    output = pd.merge(total, rate_longest, left_on='group', right_on='group')
    print(output, '\n', "DF SHAPE: ", output.shape)

    with open('out.txt', 'a') as f:
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
            y="total_rate",
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

        plt.savefig("docs/images/monthly_stability/" + str(column_to_count) + "/" + str(col) + "_CLASS_" + str(group) + "_monthly_stability_grouped" + '.jpg')
        # plt.show()
