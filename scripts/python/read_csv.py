import pandas as pd
from scripts.python.read_json import read_config_json


def read_input_csv(INPUT_ABS_APTH, INPUT_FILE_CONFIG):
    """
    :param INPUT_ABS_APTH:
    :param INPUT_FILE_CONFIG:
    :return:
    """

    # LOAD JSON CONFIG
    CONFIG = read_config_json(INPUT_FILE_CONFIG)

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

    return df, CONFIG
