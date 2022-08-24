def get_DPPL(data_frame, column_to_count, predictions_column):
    """
    :param data_frame:
    :param column_to_count:
    :param predictions_column:
    :return:
    """

    pp = []

    counts_total = data_frame.groupby(['prod_gr_id']) \
        .size() \
        .rename('count_total') \
        .reset_index() \
        .sort_values(by='prod_gr_id', ascending=False) \
        .set_index('prod_gr_id')

    positive_counts_total = counts_total.loc[counts_total['count_total'] == 1]

    print('Total counts: \n', positive_counts_total, '\n')


    counts_grouped = data_frame.groupby([column_to_count, predictions_column]) \
        .size() \
        .rename('count') \
        .reset_index() \
        .sort_values(by=column_to_count, ascending=False) \
        .set_index(column_to_count)

    positive_counts_grouped = counts_grouped.loc[counts_grouped['count'] == 1]

    #     print('Predicted Labels counts: \n', counts_grouped, '\n')

    df = pd.merge(positive_counts_grouped, positive_counts_total, left_index=True, right_index=True)
    #     positive_counts = df.loc[df[predictions_column] == '1']
    #     print(positive_counts)

    df['PPL'] = df['count'] / df['count_total']
    print(df)


    #     print(positive_counts)

    #     total_actual_positive = df[predictions_column].value_counts()[0]
    #     print(total_actual_positive)

    for key, value in df.items():
        print(key, value)
        if key == 'count':
            print("Positive Proportion in Predicted Labels (PPL) for each Class of Variable [prod_gr_id]: \n")
            ppipl = value / total_actual_positive
            for i in ppipl.iteritems():
                print('Positive Proportion in Predicted Labels (PPL) for Variable [prod_gr_id] for Class ==', i[0],":", '\n', f'{i[1]:.0%}', '\n')
                pp.append(i[1])

            unique_abs_diff = sorted(set([abs(i - j) for i in pp for j in pp if i != j]))

            print("Differences in Positive Proportion in Predicted Labels (PPL) for Variable [prod_gr_id]")
            for j in unique_abs_diff:
                print(f'{j:.0%}')

            return unique_abs_diff