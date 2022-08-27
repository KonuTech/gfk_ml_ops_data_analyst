#!/usr/bin/python

import os
import logging
from scripts.python.read_csv import read_input_csv
from scripts.python.get_reports import *
from scripts.python.get_charts import get_monthly_stability_chart, get_weekly_stability_chart


logging.basicConfig(filename='logs/logs.log', filemode='a', format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

# SET GLOBAL VARIABLES
INPUT_PATH = "input"
OUTPUT_PATH = "output"
CONFIG_PATH = "config"

INPUT_DATA = "test_data2"
INPUT_DATA_CONFIG = "input_config"
OUTPUT_DATA = "output"

INPUT_EXTENSION = "csv"
INPUT_CONFIG_EXTENSION = "json"
OUTPUT_EXTENSION = "csv"

INPUT_FILE = f"{INPUT_DATA}.{INPUT_EXTENSION}"
INPUT_CONFIG_FILE = f"{INPUT_DATA_CONFIG}.{INPUT_CONFIG_EXTENSION}"
OUTPUT_FILE = f"{OUTPUT_DATA}.{OUTPUT_EXTENSION}"

INPUT_ABS_PATH = os.path.abspath(os.path.join(INPUT_PATH, INPUT_FILE))
INPUT_FILE_CONFIG = os.path.abspath(os.path.join(CONFIG_PATH, INPUT_CONFIG_FILE))
OUTPUT_ABS_PATH = os.path.abspath(os.path.join(OUTPUT_PATH, OUTPUT_FILE))


def main():
    """
    :return:
    """

    logging.info('####### MAIN FUNCTION STARTED #######')

    # LOAD CSV INPUT USING JSON CONFIG
    df, config = read_input_csv(INPUT_ABS_PATH, INPUT_FILE_CONFIG)

    logging.info('config JSON FOR CSV LOADED')
    logging.info('CSV LOADED')

    # TARGET DRIFT - produce global report
    get_target_drift_report(
        data_frame=df,
        target=config['MODEL']['TARGET'],
        prediction=config['MODEL']['PREDICTION'],
        datetime=config['MODEL']['DATETIME'],
        categorical_fatures=config['INPUTS']['CATEGORICAL_FEATURES'],
        columns_to_exclude=config['OUTPUTS']['COLUMNS_TO_EXCLUDE'],
        breaking_point_dt=config['OUTPUTS']['BREAKING_POINT_DT']
    )

    logging.info('TARGET DRIFT - produced global report')

    # TARGET DRIFT - produce weekly reports
    get_target_drift_report_weekly(
        data_frame=df,
        target=config['MODEL']['TARGET'],
        prediction=config['MODEL']['PREDICTION'],
        datetime=config['MODEL']['DATETIME'],
        categorical_fatures=config['INPUTS']['CATEGORICAL_FEATURES'],
        columns_to_exclude=config['OUTPUTS']['COLUMNS_TO_EXCLUDE']
    )

    logging.info('TARGET DRIFT - produced weekly reports')

    # DATA DRIFT - produce global report
    get_data_drift_report(
        data_frame=df,
        target=config['MODEL']['TARGET'],
        prediction=config['MODEL']['PREDICTION'],
        datetime=config['MODEL']['DATETIME'],
        categorical_fatures=config['INPUTS']['CATEGORICAL_FEATURES'],
        columns_to_exclude=config['OUTPUTS']['COLUMNS_TO_EXCLUDE'],
        breaking_point_dt=config['OUTPUTS']['BREAKING_POINT_DT']
    )

    logging.info('DATA DRIFT - produced global report')

    # DATA DRIFT - produce weekly reports
    get_data_drift_report_weekly(
        data_frame=df,
        target=config['MODEL']['TARGET'],
        prediction=config['MODEL']['PREDICTION'],
        datetime=config['MODEL']['DATETIME'],
        categorical_fatures=config['INPUTS']['CATEGORICAL_FEATURES'],
        columns_to_exclude=config['OUTPUTS']['COLUMNS_TO_EXCLUDE']
    )

    logging.info('DATA DRIFT - produced weekly reports')

    # CLASSIFICATION PERFORMANCE - produce global report
    get_classification_performance_report(
        data_frame=df,
        target=config['MODEL']['TARGET'],
        prediction=config['MODEL']['PREDICTION'],
        datetime=config['MODEL']['DATETIME'],
        categorical_fatures=config['INPUTS']['CATEGORICAL_FEATURES'],
        columns_to_exclude=config['OUTPUTS']['COLUMNS_TO_EXCLUDE'],
        breaking_point_dt=config['OUTPUTS']['BREAKING_POINT_DT']
    )

    logging.info('CLASSIFICATION PERFORMANCE - produced global report')

    # CLASSIFICATION PERFORMANCE - produce weekly reports
    get_classification_performance_report_weekly(
        data_frame=df,
        target=config['MODEL']['TARGET'],
        prediction=config['MODEL']['PREDICTION'],
        datetime=config['MODEL']['DATETIME'],
        categorical_fatures=config['INPUTS']['CATEGORICAL_FEATURES'],
        columns_to_exclude=config['OUTPUTS']['COLUMNS_TO_EXCLUDE']
    )

    logging.info('CLASSIFICATION PERFORMANCE - produced weekly reports')

    # PREDICT AUTOMATCH MONTHLY STABILITY - produce charts per each class for every variable by month
    get_monthly_stability_chart(
        data_frame=df,
        date_column="translated_when",
        column_to_count='predict_automatch',
        columns_to_exclude=[
            'period_end_date',
            'translated_when',
            'month_year',
            'predict_automatch',
            'class_acctual',
            'if_data_corrected',
            'freq_id'
        ]
    )

    logging.info('predict_automatch MONTHLY STABILITY - produced charts for all classes of variables by month')

    # CLASS ACCTUAL MONTHLY STABILITY - produce charts per each class for every variable by month
    get_monthly_stability_chart(
        data_frame=df,
        date_column="translated_when",
        column_to_count='class_acctual',
        columns_to_exclude=[
            'period_end_date',
            'translated_when',
            'month_year',
            'predict_automatch',
            'class_acctual',
            'if_data_corrected',
            'freq_id'
        ]
    )

    logging.info('class_acctual MONTHLY STABILITY - produced charts for all classes of variables by month')

    # PREDICT AUTOMATCH WEEKLY STABILITY - produce charts per each class for every variable by week
    get_weekly_stability_chart(
        data_frame=df,
        date_column="translated_when",
        column_to_count='predict_automatch',
        columns_to_exclude=[
            'period_end_date',
            'translated_when',
            'month_year',
            'predict_automatch',
            'class_acctual',
            'if_data_corrected',
            'freq_id'
        ]
    )

    logging.info('predict_automatch WEEKLY STABILITY - produced charts for all classes of variables by week')

    # CLASS ACCTUAL WEEKLY STABILITY - produce charts per each class for every variable by week
    get_weekly_stability_chart(
        data_frame=df,
        date_column="translated_when",
        column_to_count='class_acctual',
        columns_to_exclude=[
            'period_end_date',
            'translated_when',
            'month_year',
            'predict_automatch',
            'class_acctual',
            'if_data_corrected',
            'freq_id'
        ]
    )

    logging.info('class_acctual WEEKLY STABILITY - produced charts for all classes of variables by month')
    logging.info('####### MAIN FUNCTION DONE #######')


if __name__ == "__main__":
    main()
