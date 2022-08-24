def get_PPL(data_frame, column_to_count, predictions_column):
    """
    :param data_frame:
    :param column_to_count:
    :param predictions_column:
    :return:
    """

    counts = df.groupby([column_to_count, predictions_column]) \
        .size() \
        .rename('count') \
        .reset_index() \
        .sort_values(by=column_to_count, ascending=False) \
        .set_index(column_to_count)


    print('Predicted Labels counts: \n', counts, '\n')

    positive_counts = counts.loc[counts[predictions_column] == '1']

    #     print(positive_counts)

    total_actual_positive = df[predictions_column].value_counts()[0]

    #     print(total_actual_positive)

    for key, value in positive_counts.items():
        if key == 'count':
            print("Positive Proportion in Predicted Labels for each Class of Variable [prod_gr_id]: \n")
            ppipl = value / total_actual_positive
            for i in ppipl.iteritems():
                print('Positive Proportion in Predicted Labels (PPL) for Variable [prod_gr_id] for Class ==', i[0],":", '\n', f'{i[1]:.0%}', '\n')