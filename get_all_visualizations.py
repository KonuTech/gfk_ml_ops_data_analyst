#!/usr/bin/python

import os
import logging
from scripts.python.read_csv import read_input_csv
from scripts.python.get_reports import *


logging.basicConfig(filename='logs/logs.log', filemode='a', format='%(asctime)s - %(message)s', \
                    datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
logging.info('GLOBAL VARIABLES SET UP')


# SET GLOBAL VARIABLES
INPUT_PATH="input"
OUTPUT_PATH="output"
CONFIG_PATH="config"

INPUT_DATA="test_data2"
INPUT_DATA_CONFIG="input_config"
OUTPUT_DATA="output"

INPUT_EXTENSION="csv"
INPUT_CONFIG_EXTENSION="json"
OUTPUT_EXTENSION="csv"

INPUT_FILE=f"{INPUT_DATA}.{INPUT_EXTENSION}"
INPUT_CONFIG_FILE=f"{INPUT_DATA_CONFIG}.{INPUT_CONFIG_EXTENSION}"
OUTPUT_FILE=f"{OUTPUT_DATA}.{OUTPUT_EXTENSION}"

INPUT_ABS_APTH=os.path.abspath(os.path.join(INPUT_PATH, INPUT_FILE))
INPUT_FILE_CONFIG=os.path.abspath(os.path.join(CONFIG_PATH, INPUT_CONFIG_FILE))
OUTPUT_ABS_APTH=os.path.abspath(os.path.join(OUTPUT_PATH, OUTPUT_FILE))


def main():
    """
    :return:
    """


    logging.info('MAIN FUNCTION STARTED')


    # LOAD CSV INPUT USING JSON CONFIG
    df, CONFIG = read_input_csv(INPUT_ABS_APTH, INPUT_FILE_CONFIG)


    logging.info('CONFIG JSON FOR CSV LOADED')
    logging.info('CSV LOADED')


    # TARGET DRIFT - produce global report
    get_target_drift_report(
        data_frame=df,
        target=CONFIG['MODEL']['TARGET'],
        prediction=CONFIG['MODEL']['PREDICTION'],
        datetime=CONFIG['MODEL']['DATETIME'],
        categorical_fatures=CONFIG['INPUTS']['CATEGORICAL_FEATURES'],
        columns_to_exclude=CONFIG['OUTPUTS']['COLUMNS_TO_EXCLUDE'],
        breaking_point_dt=CONFIG['OUTPUTS']['BREAKING_POINT_DT']
    )


    logging.info('TARGET DRIFT - produced global report')


    # TARGET DRIFT - produce weekly reports
    get_target_drift_report_weekly(
        data_frame=df,
        target=CONFIG['MODEL']['TARGET'],
        prediction=CONFIG['MODEL']['PREDICTION'],
        datetime=CONFIG['MODEL']['DATETIME'],
        categorical_fatures=CONFIG['INPUTS']['CATEGORICAL_FEATURES'],
        columns_to_exclude=CONFIG['OUTPUTS']['COLUMNS_TO_EXCLUDE']
    )


    logging.info('TARGET DRIFT - produced weekly reports')


    # DATA DRIFT - produce global report
    get_data_drift_report(
        data_frame=df,
        target=CONFIG['MODEL']['TARGET'],
        prediction=CONFIG['MODEL']['PREDICTION'],
        datetime=CONFIG['MODEL']['DATETIME'],
        categorical_fatures=CONFIG['INPUTS']['CATEGORICAL_FEATURES'],
        columns_to_exclude=CONFIG['OUTPUTS']['COLUMNS_TO_EXCLUDE'],
        breaking_point_dt=CONFIG['OUTPUTS']['BREAKING_POINT_DT']
    )


    logging.info('DATA DRIFT - produced global report')


    # DATA DRIFT - produce weekly reports
    get_data_drift_report_weekly(
        data_frame=df,
        target=CONFIG['MODEL']['TARGET'],
        prediction=CONFIG['MODEL']['PREDICTION'],
        datetime=CONFIG['MODEL']['DATETIME'],
        categorical_fatures=CONFIG['INPUTS']['CATEGORICAL_FEATURES'],
        columns_to_exclude=CONFIG['OUTPUTS']['COLUMNS_TO_EXCLUDE']
    )


    logging.info('DATA DRIFT - produced weekly reports')


    # CLASSIFICATION PERFORMANCE - produce global report
    get_classification_performance_report(
        data_frame=df,
        target=CONFIG['MODEL']['TARGET'],
        prediction=CONFIG['MODEL']['PREDICTION'],
        datetime=CONFIG['MODEL']['DATETIME'],
        categorical_fatures=CONFIG['INPUTS']['CATEGORICAL_FEATURES'],
        columns_to_exclude=CONFIG['OUTPUTS']['COLUMNS_TO_EXCLUDE'],
        breaking_point_dt=CONFIG['OUTPUTS']['BREAKING_POINT_DT']
    )


    logging.info('CLASSIFICATION PERFORMANCE - produced global report')


    # CLASSIFICATION PERFORMANCE - produce weekly reports
    get_classification_performance_report_weekly(
        data_frame=df,
        target=CONFIG['MODEL']['TARGET'],
        prediction=CONFIG['MODEL']['PREDICTION'],
        datetime=CONFIG['MODEL']['DATETIME'],
        categorical_fatures=CONFIG['INPUTS']['CATEGORICAL_FEATURES'],
        columns_to_exclude=CONFIG['OUTPUTS']['COLUMNS_TO_EXCLUDE']
    )


    logging.info('CLASSIFICATION PERFORMANCE - produced weekly reports')
    logging.info('MAIN FUNCTION DONE')


if __name__ == "__main__":
    main()

