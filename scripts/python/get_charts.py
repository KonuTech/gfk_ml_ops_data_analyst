from datetime import datetime
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import itertools

matplotlib.use("Agg")


def get_monthly_stability_chart(data_frame, date_column, column_to_count, columns_to_exclude):
    """
    :param data_frame:
    :param date_column:
    :param column_to_count:
    :param columns_to_exclude:
    :return:
    """

    from contextlib import redirect_stdout

    data_frame['year_month'] = data_frame[date_column].dt.to_period('M')

    for column in data_frame.columns:
        if column not in sorted(set(columns_to_exclude)):

            print(column)

            total = data_frame.groupby([data_frame['year_month'], column])[column_to_count].count().reset_index()
            total['group'] = total['year_month'].astype(str) + " " + total[column].astype(str)

            rate = data_frame[data_frame[column_to_count] == 1].groupby([data_frame['year_month'], column])[
                column_to_count].sum().reset_index()
            rate['group'] = rate['year_month'].astype(str) + " " + rate[column].astype(str)

            output = pd.merge(rate[['group', column_to_count]], total, how='right', left_on='group', right_on='group')
            output.rename(columns={f"{column_to_count}_x": "predict_1", f"{column_to_count}_y": "predict_total"},
                          inplace=True)
            output['rate'] = output['predict_1'] / output['predict_total']
            output['rate_total'] = 1.0
            # print(output, '\n', "DF SHAPE: ", output.shape)

            with open('logs/rates_output.txt', 'w') as f:
                with redirect_stdout(f):
                    print(output.to_string())

            for group in sorted(set(output[column])):

                plt.figure(figsize=(10, 2))

                # legend
                top_bar = mpatches.Patch(color='darkblue', label='0')
                bottom_bar = mpatches.Patch(color='lightblue', label='1')
                plt.legend(handles=[top_bar, bottom_bar])

                try:
                    sns.barplot(
                        x='year_month',
                        y="rate_total",
                        data=output[output[column] == group],
                        color='darkblue',
                        # alpha=0.5
                    )
                    # plt.clf()

                    sns.barplot(
                        x='year_month',
                        y="rate",
                        data=output[output[column] == group],
                        color='lightblue',
                        alpha=0.5
                    )
                    # plt.clf()

                    plt.savefig("output/charts/monthly_stability/" + str(column_to_count) + "/" + str(
                        column) + "/" + "CLASS_" + str(group) + "_monthly_stability_grouped" + '.jpg')
                    plt.clf()
                    print("PRODUCED A CHART OF " + str(column_to_count) + " MONTHLY STABILITY FOR: ", str(column),
                          " VARIABLE CLASS: ", group)

                except Exception:
                    pass


def get_weekly_stability_chart(data_frame, date_column, column_to_count, columns_to_exclude):
    """
    :param data_frame:
    :param date_column:
    :param column_to_count:
    :param columns_to_exclude:
    :return:
    """

    from contextlib import redirect_stdout

    for column in data_frame.columns:
        if column not in sorted(set(columns_to_exclude)):

            data_frame['week'] = data_frame[date_column].dt.strftime('%Y-%V')

            total = data_frame.groupby([data_frame['week'], column])[column_to_count].count().reset_index()
            total['group'] = total['week'] + " " + total[column]

            rate = data_frame[data_frame[column_to_count] == 1].groupby([data_frame['week'], column])[
                column_to_count].sum().reset_index()
            rate['group'] = rate['week'] + " " + rate[column]

            output = pd.merge(rate[['group', column_to_count]], total, how='right', left_on='group', right_on='group')
            output.rename(columns={f"{column_to_count}_x": "predict_1", f"{column_to_count}_y": "predict_total"},
                          inplace=True)
            output['rate'] = output['predict_1'] / output['predict_total']
            output['rate_total'] = 1.0
            # print(output, '\n', "DF SHAPE: ", output.shape)

            with open('logs/rates_output.txt', 'w') as f:
                with redirect_stdout(f):
                    print(output.to_string())

            for group in sorted(set(output[column])):
                print(group)

                plt.figure(figsize=(10, 2))

                # legend
                top_bar = mpatches.Patch(color='darkblue', label='0')
                bottom_bar = mpatches.Patch(color='lightblue', label='1')
                plt.legend(handles=[top_bar, bottom_bar])

                sns.barplot(
                    x='week',
                    y="rate_total",
                    data=output[output[column] == group],
                    color='darkblue',
                    # alpha=0.5
                )

                sns.barplot(
                    x='week',
                    y="rate",
                    data=output[output[column] == group],
                    color='lightblue',
                    alpha=0.5
                )

                plt.savefig("output/charts/weekly_stability/" + str(column_to_count) + "/" + str(
                    column) + "/" + "CLASS_" + str(group) + "_weekly_stability_grouped" + '.jpg')
                plt.clf()
                print("PRODUCED A CHART OF " + str(column_to_count) + " WEEKLY STABILITY FOR: ", str(column),
                      " VARIABLE CLASS: ", group)
