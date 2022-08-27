import matplotlib as plt
from evidently.dashboard import Dashboard
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab, ClassificationPerformanceTab

plt.rcParams.update({'figure.max_open_warning': 0})


def get_target_drift_report(data_frame, target, prediction, datetime,
                            categorical_fatures, columns_to_exclude, breaking_point_dt):
    """
    :param data_frame:
    :param target:
    :param prediction:
    :param datetime:
    :param categorical_fatures:
    :param columns_to_exclude:
    :param breaking_point_dt:
    :return:
    """

    target_drift_raport = Dashboard(tabs=[CatTargetDriftTab(verbose_level=1)])

    cf = categorical_fatures

    df_column_mapping = ColumnMapping()
    df_column_mapping.categorical_features = cf
    df_column_mapping.target = target
    df_column_mapping.prediction = prediction
    df_column_mapping.datetime = datetime

    data_frame = data_frame.drop(columns=columns_to_exclude)
    data_frame.sort_values(by=[datetime], inplace=True)
    data_frame.reset_index(drop=True, inplace=True)
    data_frame['week_number'] = data_frame[datetime].dt.strftime('%Y%V').astype(int)

    reference = data_frame[data_frame[datetime] < breaking_point_dt]
    reference['week_number'] = reference['week_number'].apply(str)
    # reference.drop('week_number', axis=1, inplace=True)
    current = data_frame[data_frame[datetime] >= breaking_point_dt]
    current['week_number'] = current['week_number'].apply(str)
    # current.drop('week_number', axis=1, inplace=True)

    target_drift_raport.calculate(reference, current, column_mapping=df_column_mapping)
    target_drift_raport.save("output/reports/target_drift/000000_target_drift_report.html")
    print("PRODUCED A CHART OF TARGET DRIFT GLOBAL")


def get_target_drift_report_weekly(data_frame, target, prediction, datetime, categorical_fatures,
                                   columns_to_exclude):
    """
    :param data_frame:
    :param target:
    :param prediction:
    :param datetime:
    :param categorical_fatures:
    :param columns_to_exclude:
    :return:
    """

    target_drift_raport = Dashboard(tabs=[CatTargetDriftTab(verbose_level=1)])
    cf = categorical_fatures

    df_column_mapping = ColumnMapping()
    df_column_mapping.categorical_features = cf
    df_column_mapping.target = target
    df_column_mapping.prediction = prediction
    df_column_mapping.datetime = datetime

    data_frame = data_frame.drop(columns=columns_to_exclude)
    data_frame.sort_values(by=[datetime], inplace=True)
    data_frame.reset_index(drop=True, inplace=True)

    data_frame['week_number'] = data_frame[datetime].dt.strftime('%Y%V').astype(int)

    for week in sorted(set(data_frame['week_number']))[2:]:
        reference = data_frame[data_frame['week_number'] == week]
        reference['week_number'] = reference['week_number'].apply(str)
        # reference.drop('week_number', axis=1, inplace=True)
        print("\n REFERENCE SHAPE: ", reference.shape)
        current = data_frame[data_frame['week_number'] == (week + 1)]
        current['week_number'] = current['week_number'].apply(str)
        # current.drop('week_number', axis=1, inplace=True)
        print("\n CURRENT SHAPE: ", current.shape)

        if current.shape[0] > 0:
            target_drift_raport.calculate(reference, current, column_mapping=df_column_mapping)
            target_drift_raport.save("output/reports/target_drift/" + str(week) + "_target_drift_report.html")
            print("PRODUCED A CHART OF TARGET DRIFT WEEKLY FOR WEEK: " + str(week))


def get_data_drift_report(data_frame, target, prediction, datetime,
                          categorical_fatures, columns_to_exclude, breaking_point_dt):
    """
    :param data_frame:
    :param target:
    :param prediction:
    :param datetime:
    :param categorical_fatures:
    :param columns_to_exclude:
    :param breaking_point_dt:
    :return:
    """

    data_drift_report = Dashboard(tabs=[DataDriftTab()])

    cf = categorical_fatures

    df_column_mapping = ColumnMapping()
    df_column_mapping.categorical_features = cf
    df_column_mapping.target = target
    df_column_mapping.prediction = prediction
    df_column_mapping.datetime = datetime

    data_frame = data_frame.drop(columns=columns_to_exclude)
    data_frame.sort_values(by=[datetime], inplace=True)
    data_frame.reset_index(drop=True, inplace=True)
    data_frame['week_number'] = data_frame[datetime].dt.strftime('%Y%V').astype(int)

    reference = data_frame[data_frame[datetime] < breaking_point_dt]
    reference['week_number'] = reference['week_number'].apply(str)
    # reference.drop('week_number', axis=1, inplace=True)
    current = data_frame[data_frame[datetime] >= breaking_point_dt]
    current['week_number'] = current['week_number'].apply(str)
    # current.drop('week_number', axis=1, inplace=True)

    data_drift_report.calculate(reference, current, column_mapping=df_column_mapping)
    data_drift_report.save("output/reports/data_drift/000000_data_drift_report.html")
    print("PRODUCED A CHART OF DATA DRIFT GLOBAL")


def get_data_drift_report_weekly(data_frame, target, prediction, datetime, categorical_fatures,
                                 columns_to_exclude):
    """
    :param data_frame:
    :param target:
    :param prediction:
    :param datetime:
    :param categorical_fatures:
    :param columns_to_exclude:
    :return:
    """

    data_drift_report_weekly = Dashboard(tabs=[DataDriftTab()])
    cf = categorical_fatures

    df_column_mapping = ColumnMapping()
    df_column_mapping.categorical_features = cf
    df_column_mapping.target = target
    df_column_mapping.prediction = prediction
    df_column_mapping.datetime = datetime

    data_frame = data_frame.drop(columns=columns_to_exclude)
    data_frame.sort_values(by=[datetime], inplace=True)
    data_frame.reset_index(drop=True, inplace=True)

    data_frame['week_number'] = data_frame[datetime].dt.strftime('%Y%V').astype(int)

    for week in sorted(set(data_frame['week_number']))[2:]:
        reference = data_frame[data_frame['week_number'] == week]
        reference['week_number'] = reference['week_number'].apply(str)
        # reference.drop('week_number', axis=1, inplace=True)
        print("\n REFERENCE SHAPE: ", reference.shape)
        current = data_frame[data_frame['week_number'] == (week + 1)]
        current['week_number'] = current['week_number'].apply(str)
        # current.drop('week_number', axis=1, inplace=True)
        print("\n CURRENT SHAPE: ", current.shape)

        if current.shape[0] > 0:
            data_drift_report_weekly.calculate(reference, current, column_mapping=df_column_mapping)
            data_drift_report_weekly.save("output/reports/data_drift/" + str(week) + "_data_drift_report.html")
            print("PRODUCED A CHART OF DATA DRIFT WEEKLY FOR WEEK: " + str(week))


def get_classification_performance_report(data_frame, target, prediction, datetime,
                                          categorical_fatures, columns_to_exclude, breaking_point_dt):
    """
    :param data_frame:
    :param target:
    :param prediction:
    :param datetime:
    :param categorical_fatures:
    :param columns_to_exclude:
    :param breaking_point_dt:
    :return:
    """

    model_performance_report = Dashboard(tabs=[ClassificationPerformanceTab(verbose_level=1)])

    cf = categorical_fatures

    df_column_mapping = ColumnMapping()
    df_column_mapping.categorical_features = cf
    df_column_mapping.target = target
    df_column_mapping.prediction = prediction
    df_column_mapping.datetime = datetime

    data_frame = data_frame.drop(columns=columns_to_exclude)
    data_frame.sort_values(by=[datetime], inplace=True)
    data_frame.reset_index(drop=True, inplace=True)
    data_frame['week_number'] = data_frame[datetime].dt.strftime('%Y%V').astype(int)

    reference = data_frame[data_frame[datetime] < breaking_point_dt]
    reference['week_number'] = reference['week_number'].apply(str)
    # reference.drop('week_number', axis=1, inplace=True)
    current = data_frame[data_frame[datetime] >= breaking_point_dt]
    current['week_number'] = current['week_number'].apply(str)
    # current.drop('week_number', axis=1, inplace=True)

    model_performance_report.calculate(reference, current, column_mapping=df_column_mapping)
    model_performance_report.save(
        "output/reports/classification_performance/000000_classification_performance_report.html")
    print("PRODUCED A CHART OF CLASSIFICATION PERFORMANCE GLOBAL")


def get_classification_performance_report_weekly(data_frame, target, prediction, datetime, categorical_fatures,
                                                 columns_to_exclude):
    """
    :param data_frame:
    :param target:
    :param prediction:
    :param datetime:
    :param categorical_fatures:
    :param columns_to_exclude:
    :return:
    """

    model_performance_report_weekly = Dashboard(tabs=[ClassificationPerformanceTab(verbose_level=1)])
    cf = categorical_fatures

    df_column_mapping = ColumnMapping()
    df_column_mapping.categorical_features = cf
    df_column_mapping.target = target
    df_column_mapping.prediction = prediction
    df_column_mapping.datetime = datetime

    data_frame = data_frame.drop(columns=columns_to_exclude)
    data_frame.sort_values(by=[datetime], inplace=True)
    data_frame.reset_index(drop=True, inplace=True)

    data_frame['week_number'] = data_frame[datetime].dt.strftime('%Y%V').astype(int)

    for week in sorted(set(data_frame['week_number']))[2:]:
        reference = data_frame[data_frame['week_number'] == week]
        reference['week_number'] = reference['week_number'].apply(str)
        # reference.drop('week_number', axis=1, inplace=True)
        print("\n REFERENCE SHAPE: ", reference.shape)
        current = data_frame[data_frame['week_number'] == (week + 1)]
        current['week_number'] = current['week_number'].apply(str)
        # current.drop('week_number', axis=1, inplace=True)
        print("\n CURRENT SHAPE: ", current.shape)

        if current.shape[0] > 0:
            try:
                model_performance_report_weekly.calculate(reference, current, column_mapping=df_column_mapping)
                model_performance_report_weekly.save("output/reports/classification_performance/" + str(
                    week) + "_classification_performance_report.html")
                print("PRODUCED A CHART OF CLASSIFICATION PERFORMANCE WEEKLY FOR WEEK: " + str(week))
            except Exception:
                pass
