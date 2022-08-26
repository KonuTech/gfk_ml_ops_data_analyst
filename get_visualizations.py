#!/usr/bin/python

import os
import json
import pandas as pd
from scripts.python.get_reports import *

# VARIABLES
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


# LOAD JSON CONFIG
with open(INPUT_FILE_CONFIG, encoding='utf-8') as f:
    CONFIG = json.load(f)


def main():
    """
    :return:
    """

    # LOAD CSV
    df = pd.read_csv(
        INPUT_ABS_APTH,
        sep=CONFIG['INPUTS']['SEPARATOR'],
        encoding=CONFIG['INPUTS']['ENCODING'],
        infer_datetime_format=True,
        parse_dates=CONFIG['INPUTS']['DATE_COLUMNS'],
        engine="c",
        low_memory=False,
        skipinitialspace=True,
        dtype=CONFIG['INPUTS']['DTYPE']
    )


    # TARGET DRIFT - global report
    get_target_drift_report(
        data_frame=df,
        target=CONFIG['MODEL']['TARGET'],
        prediction=CONFIG['MODEL']['PREDICTION'],
        datetime=CONFIG['MODEL']['DATETIME'],
        categorical_fatures=CONFIG['INPUTS']['CATEGORICAL_FEATURES'],
        columns_to_exclude=CONFIG['OUTPUTS']['COLUMNS_TO_EXCLUDE'],
        breaking_point_dt=CONFIG['OUTPUTS']['BREAKING_POINT_DT']
    )

    # TARGET DRIFT - weekly report
    get_target_drift_report_weekly(
        data_frame=df,
        target=CONFIG['MODEL']['TARGET'],
        prediction=CONFIG['MODEL']['PREDICTION'],
        datetime=CONFIG['MODEL']['DATETIME'],
        categorical_fatures=CONFIG['INPUTS']['CATEGORICAL_FEATURES'],
        columns_to_exclude=CONFIG['OUTPUTS']['COLUMNS_TO_EXCLUDE']
    )

    # DATA DRIFT - global report
    get_data_drift_report(
        data_frame=df,
        target=CONFIG['MODEL']['TARGET'],
        prediction=CONFIG['MODEL']['PREDICTION'],
        datetime=CONFIG['MODEL']['DATETIME'],
        categorical_fatures=CONFIG['INPUTS']['CATEGORICAL_FEATURES'],
        columns_to_exclude=CONFIG['OUTPUTS']['COLUMNS_TO_EXCLUDE'],
        breaking_point_dt=CONFIG['OUTPUTS']['BREAKING_POINT_DT']
    )

    # DATA DRIFT - weekly report
    get_data_drift_report_weekly(
        data_frame=df,
        target=CONFIG['MODEL']['TARGET'],
        prediction=CONFIG['MODEL']['PREDICTION'],
        datetime=CONFIG['MODEL']['DATETIME'],
        categorical_fatures=CONFIG['INPUTS']['CATEGORICAL_FEATURES'],
        columns_to_exclude=CONFIG['OUTPUTS']['COLUMNS_TO_EXCLUDE']
    )

    # CLASSIFICATION PERFORMANCE - global report
    get_classification_performance_report(
        data_frame=df,
        target=CONFIG['MODEL']['TARGET'],
        prediction=CONFIG['MODEL']['PREDICTION'],
        datetime=CONFIG['MODEL']['DATETIME'],
        categorical_fatures=CONFIG['INPUTS']['CATEGORICAL_FEATURES'],
        columns_to_exclude=CONFIG['OUTPUTS']['COLUMNS_TO_EXCLUDE'],
        breaking_point_dt=CONFIG['OUTPUTS']['BREAKING_POINT_DT']
    )

    # CLASSIFICATION PERFORMANCE - weekly report
    get_classification_performance_report_weekly(
        data_frame=df,
        target=CONFIG['MODEL']['TARGET'],
        prediction=CONFIG['MODEL']['PREDICTION'],
        datetime=CONFIG['MODEL']['DATETIME'],
        categorical_fatures=CONFIG['INPUTS']['CATEGORICAL_FEATURES'],
        columns_to_exclude=CONFIG['OUTPUTS']['COLUMNS_TO_EXCLUDE']
    )

if __name__ == "__main__":
    main()
