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

    # loop over columns
    for column in data_frame.columns:
        if column not in sorted(set(columns_to_exclude)):

            # get yyyy-mm column
            data_frame['year_month'] = data_frame[date_column].dt.strftime('%Y-%m')

            # agg total
            total = data_frame.groupby([data_frame['year_month'], column])[column_to_count].count().reset_index()
            total['group'] = f"{total['year_month'].astype(str)} {total[column].astype(str)}"

            # agg when prediction == 1
            rate = data_frame[data_frame[column_to_count] == 1].groupby([data_frame['year_month'], column])[
                column_to_count].sum().reset_index()
            rate['group'] = f"{rate['year_month'].astype(str)} {rate[column].astype(str)}"

            # left join
            output = pd.merge(rate[['group', column_to_count]], total, how='right', left_on='group', right_on='group')
            output.rename(
                columns={f"{column_to_count}_x": "predict_1", f"{column_to_count}_y": "predict_total"}, inplace=True
            )
            output['rate'] = output['predict_1'] / output['predict_total']
            output['rate_total'] = 1.0

            # log rates to txt
            with open('logs/rates_output.txt', 'a') as f:
                with redirect_stdout(f):
                    print(output.to_string())

            # loop over Classes
            for group in sorted(set(output[column])):

                plt.figure(figsize=(10, 2))

                # set up chart legend
                top_bar = mpatches.Patch(color='darkblue', label='0')
                bottom_bar = mpatches.Patch(color='lightblue', label='1')
                plt.legend(handles=[top_bar, bottom_bar])

                try:
                    sns.barplot(
                        x='year_month',
                        y="rate_total",
                        data=output[output[column] == group],
                        color='darkblue'
                    )

                    sns.barplot(
                        x='year_month',
                        y="rate",
                        data=output[output[column] == group],
                        color='lightblue',
                        alpha=0.5
                    )

                    plt.savefig(
                        f"output/charts/monthly_stability/{column_to_count}/{column}/CLASS_{group}_monthly_stability_grouped.jpg"
                    )
                    plt.clf()
                    print(f"PRODUCED A CHART OF {column_to_count} MONTHLY STABILITY FOR VAR:{column}, CLASS:{group}")

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

    # loop over columns
    for column in data_frame.columns:
        if column not in sorted(set(columns_to_exclude)):

            # get week column
            data_frame['week'] = data_frame[date_column].dt.strftime('%Y-%V')

            # agg total
            total = data_frame.groupby([data_frame['week'], column], as_index=False)[column_to_count].count()
            total['group'] = total['week'] + " " + total[column]

            # agg when prediction == 1
            rate = data_frame[data_frame[column_to_count] == 1].groupby([data_frame['week'], column], as_index=True)[
                column_to_count].sum().reset_index()
            rate['group'] = rate['week'] + " " + rate[column]

            # left join
            output = pd.merge(rate[['group', column_to_count]], total, how='right', left_on='group', right_on='group')
            output.rename(columns={f"{column_to_count}_x": "predict_1", f"{column_to_count}_y": "predict_total"},
                          inplace=True)
            output['rate'] = output['predict_1'] / output['predict_total']
            output['rate_total'] = 1.0

            # log rates to txt
            with open('logs/rates_output.txt', 'a') as f:
                with redirect_stdout(f):
                    print(output.to_string())

            # loop over Classes
            for group in sorted(set(output[column])):

                plt.figure(figsize=(10, 2))

                # set up chart legend
                top_bar = mpatches.Patch(color='darkblue', label='0')
                bottom_bar = mpatches.Patch(color='lightblue', label='1')
                plt.legend(handles=[top_bar, bottom_bar])

                try:
                    sns.barplot(
                        x='week',
                        y="rate_total",
                        data=output[output[column] == group],
                        color='darkblue'
                    )

                    sns.barplot(
                        x='week',
                        y="rate",
                        data=output[output[column] == group],
                        color='lightblue',
                        alpha=0.5
                    )

                    plt.savefig(
                        f"output/charts/weekly_stability/{column_to_count}/{column}/CLASS_{group}_weekly_stability_grouped.jpg"
                    )
                    plt.clf()
                    print(f"PRODUCED A CHART OF {column_to_count} WEEKLY STABILITY FOR VAR:{column}, CLASS: {group}")

                except Exception:
                    pass
