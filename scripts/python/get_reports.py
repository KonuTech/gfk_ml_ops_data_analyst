import numpy as np
import pandas as pd
import datetime
import matplotlib as plt
from scipy.spatial import distance
from evidently.dashboard import Dashboard
from evidently.options import DataDriftOptions
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab, ClassificationPerformanceTab

from evidently.test_suite import TestSuite
from evidently.tests import *
from evidently.test_preset import NoTargetPerformance, DataQuality, DataStability, DataDrift

plt.rcParams.update({'figure.max_open_warning': 0})


def jensenshannon_stat_test(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    return distance.cdist(np.array([reference_data, current_data]))


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
    print(f"\n REFERENCE SHAPE: {reference.shape}")

    current = data_frame[data_frame[datetime] >= breaking_point_dt]
    current['week_number'] = current['week_number'].apply(str)
    print(f"\n CURRENT SHAPE: {current.shape}")

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
        print(f"\n REFERENCE SHAPE: {reference.shape}")

        current = data_frame[data_frame['week_number'] == (week + 1)]
        current['week_number'] = current['week_number'].apply(str)
        print(f"\n CURRENT SHAPE: {current.shape}")

        if current.shape[0] > 0:
            target_drift_raport.calculate(reference, current, column_mapping=df_column_mapping)
            target_drift_raport.save(f"output/reports/target_drift/{week}_target_drift_report.html")
            print("PRODUCED A CHART OF TARGET DRIFT WEEKLY FOR WEEK: " + str(week))


def get_target_drift_report_custom(data_frame, target, prediction, datetime,
                            categorical_fatures, columns_to_exclude, date_from, date_to,
                                   breaking_point_dt, prod_gr_id):
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

    # filtering data_frame by date
    data_frame = data_frame[
        (data_frame[prediction].notnull())
        & (data_frame['prod_gr_id'] == str(prod_gr_id))
        & (data_frame[datetime].dt.date >= date_from)
        & (data_frame[datetime].dt.date <= date_to)
        ]
    print("data_frame.shape:\n", data_frame.shape)

    data_frame = data_frame.drop(columns=columns_to_exclude)
    data_frame.sort_values(by=[datetime], inplace=True)
    data_frame.reset_index(drop=True, inplace=True)
    data_frame['week_number'] = data_frame[datetime].dt.strftime('%Y%V').astype(int)

    reference = data_frame[data_frame[datetime].dt.date < breaking_point_dt]
    reference['week_number'] = reference['week_number'].apply(str)
    print(f"\n REFERENCE SHAPE: {reference.shape}")

    current = data_frame[data_frame[datetime].dt.date >= breaking_point_dt]
    current['week_number'] = current['week_number'].apply(str)
    print(f"\n CURRENT SHAPE: {current.shape}")

    target_drift_raport.calculate(reference, current, column_mapping=df_column_mapping)
    target_drift_raport.save(f"output/reports/target_drift/000{prod_gr_id}_target_drift_report_custom.html")
    print("PRODUCED A CHART OF TARGET DRIFT CUSTOM")


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
    print(f"\n REFERENCE SHAPE: {reference.shape}")

    current = data_frame[data_frame[datetime] >= breaking_point_dt]
    current['week_number'] = current['week_number'].apply(str)
    print(f"\n CURRENT SHAPE: {current.shape}")

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
        print(f"\n REFERENCE SHAPE: {reference.shape}")

        current = data_frame[data_frame['week_number'] == (week + 1)]
        current['week_number'] = current['week_number'].apply(str)
        print(f"\n CURRENT SHAPE: {current.shape}")

        if current.shape[0] > 0:
            data_drift_report_weekly.calculate(reference, current, column_mapping=df_column_mapping)
            data_drift_report_weekly.save(f"output/reports/data_drift/{week}_data_drift_report.html")
            print("PRODUCED A CHART OF DATA DRIFT WEEKLY FOR WEEK: " + str(week))


def get_data_drift_report_custom(data_frame, target, prediction, datetime,
                          categorical_fatures, columns_to_exclude, date_from, date_to,
                                 breaking_point_dt, prod_gr_id):
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

    # filtering data_frame by date
    data_frame = data_frame[
        (data_frame[prediction].notnull())
        & (data_frame['prod_gr_id'] == str(prod_gr_id))
        & (data_frame[datetime].dt.date >= date_from)
        & (data_frame[datetime].dt.date <= date_to)
        ]
    print("data_frame.shape:\n", data_frame.shape)

    data_frame = data_frame.drop(columns=columns_to_exclude)
    data_frame.sort_values(by=[datetime], inplace=True)
    data_frame.reset_index(drop=True, inplace=True)
    data_frame['week_number'] = data_frame[datetime].dt.strftime('%Y%V').astype(int)

    reference = data_frame[data_frame[datetime].dt.date < breaking_point_dt]
    reference['week_number'] = reference['week_number'].apply(str)
    print(f"\n REFERENCE SHAPE: {reference.shape}")

    current = data_frame[data_frame[datetime].dt.date >= breaking_point_dt]
    current['week_number'] = current['week_number'].apply(str)
    print(f"\n CURRENT SHAPE: {current.shape}")

    data_drift_report.calculate(reference, current, column_mapping=df_column_mapping)
    data_drift_report.save(f"output/reports/data_drift/000{prod_gr_id}_data_drift_report_custom.html")
    print("PRODUCED A CHART OF DATA DRIFT GLOBAL")


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
    print(f"\n REFERENCE SHAPE: {reference.shape}")

    current = data_frame[data_frame[datetime] >= breaking_point_dt]
    current['week_number'] = current['week_number'].apply(str)
    print(f"\n CURRENT SHAPE: {current.shape}")

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
        print(f"\n REFERENCE SHAPE: {reference.shape}")

        current = data_frame[data_frame['week_number'] == (week + 1)]
        current['week_number'] = current['week_number'].apply(str)
        print(f"\n CURRENT SHAPE: {current.shape}")

        if current.shape[0] > 0:
            try:
                model_performance_report_weekly.calculate(reference, current, column_mapping=df_column_mapping)
                model_performance_report_weekly.save(
                    f"output/reports/classification_performance/{week}_classification_performance_report.html")
                print("PRODUCED A CHART OF CLASSIFICATION PERFORMANCE WEEKLY FOR WEEK: " + str(week))
            except Exception:
                pass


def get_classification_performance_report_custom(data_frame, target, prediction, datetime,
                                                 categorical_fatures, columns_to_exclude, date_from, date_to,
                                                 breaking_point_dt, prod_gr_id):
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
    print("data_frame.shape:\n", data_frame.shape)

    # applying data constraints
    data_frame = data_frame[
          (data_frame[prediction].notnull())
        & (data_frame['prod_gr_id'] == str(prod_gr_id))
        & (data_frame[datetime].dt.date >= date_from)
        & (data_frame[datetime].dt.date <= date_to)
        ]
    print("data_frame.shape:\n", data_frame.shape)

    data_frame['week_number'] = data_frame[datetime].dt.strftime('%Y%V').astype(int)

    reference = data_frame[data_frame[datetime].dt.date < breaking_point_dt]
    reference['week_number'] = reference['week_number'].apply(str)
    print(f"\n REFERENCE SHAPE: {reference.shape}")

    current = data_frame[data_frame[datetime].dt.date >= breaking_point_dt]
    current['week_number'] = current['week_number'].apply(str)
    print(f"\n CURRENT SHAPE: {current.shape}")

    model_performance_report.calculate(reference, current, column_mapping=df_column_mapping)
    model_performance_report.save(
        f"output/reports/classification_performance/000{prod_gr_id}_classification_performance_report_custom.html")
    print("PRODUCED A CHART OF CLASSIFICATION PERFORMANCE CUSTOM")