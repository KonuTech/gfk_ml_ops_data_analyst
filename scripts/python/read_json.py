import json


def read_config_json(INPUT_FILE_CONFIG):
    """
    :param INPUT_FILE_CONFIG:
    :return:
    """

    # LOAD JSON CONFIG
    with open(INPUT_FILE_CONFIG, encoding='utf-8') as f:
        CONFIG = json.load(f)

    return CONFIG
