import copy
import itertools
import json
import logging as log
import os
import typing

import pandas as pd

import common

BROKER_IP = '127.0.0.1'
BROKER_PORT = '9999'
ENDPOINT = f'{BROKER_IP}:{BROKER_PORT}'

# Note: targetSensorAttributes: should be list of list
DATASET_DEPENDENT_SETTING = {
    'METR-LA': {
        'additionalTemporalEmbed': ['timestamp_in_day'],
        'targetSensorAttributes': [['speed']],
        'inputLength': [12], 'outputLength': [12],
        # 'inputLength': [6, 12, 18, 24], 'outputLength': [6, 12, 18, 24],
    },
    'PEMS-BAY': {
        'additionalTemporalEmbed': ['timestamp_in_day'],
        'targetSensorAttributes': [['speed']],
        'inputLength': [12], 'outputLength': [12],
        # 'inputLength': [6, 12, 18, 24], 'outputLength': [6, 12, 18, 24],
    },
    'PEMSD7': {
        'additionalTemporalEmbed': ['timestamp_in_day'],
        'targetSensorAttributes': [['speed']],
        'inputLength': [12], 'outputLength': [12],
        # 'inputLength': [6, 12, 18, 24], 'outputLength': [6, 12, 18, 24],
    },
    'PEMS03': {
        'additionalTemporalEmbed': ['timestamp_in_day'],
        'targetSensorAttributes': [['flow']],
        'inputLength': [12], 'outputLength': [12],
    },
    'PEMS04': {
        'additionalTemporalEmbed': ['timestamp_in_day'],
        'targetSensorAttributes': [['flow']],
        'inputLength': [12], 'outputLength': [12],
    },
    'PEMS07': {
        'additionalTemporalEmbed': ['timestamp_in_day'],
        'targetSensorAttributes': [['flow']],
        'inputLength': [12], 'outputLength': [12],
        # 'inputLength': [6, 12, 18, 24], 'outputLength': [6, 12, 18, 24],
    },
    'PEMS08': {
        'additionalTemporalEmbed': ['timestamp_in_day'],
        'targetSensorAttributes': [['flow']],
        'inputLength': [12], 'outputLength': [12],
    },
    'Chickenpox': {
        'additionalTemporalEmbed': ['timestamp_in_day'],
        'targetSensorAttributes': [['cases']],
        'inputLength': [8], 'outputLength': [10],
    },
    'JP-Pref': {
        'additionalTemporalEmbed': ['timestamp_in_day'],
        'targetSensorAttributes': [['cases']],
        'inputLength': [20], 'outputLength': [15],
    },
    'US-Regions': {
        'additionalTemporalEmbed': ['timestamp_in_day'],
        'targetSensorAttributes': [['cases']],
        'inputLength': [20], 'outputLength': [15],
    },
    'US-States': {
        'additionalTemporalEmbed': ['timestamp_in_day'],
        'targetSensorAttributes': [['cases']],
        'inputLength': [20], 'outputLength': [15],
    },
    'Uzel2022': {
        'additionalTemporalEmbed': ['timestamp_in_minute'],
        'targetSensorAttributes': [['intensity']],
        'inputLength': [15], 'outputLength': [15],
        # 'inputLength': [7, 15, 22, 30], 'outputLength': [7, 15, 22, 30],
    },
    'NOAA1000': {
        'additionalTemporalEmbed': ['timestamp_in_day'],
        'targetSensorAttributes': [['temperature']],
        'inputLength': [7], 'outputLength': [7],
        # 'inputLength': [3, 7, 11, 14], 'outputLength': [3, 7, 11, 14],
    },
}

MODEL_DEPENDENT_SETTING = {
    'ARMA': {'adjacencyMatrixLaplacianMatrix': [None]},
    'VAR': {'adjacencyMatrixLaplacianMatrix': [None]},
    'Seq2Seq_RNN': {'adjacencyMatrixLaplacianMatrix': [None]},
    'Seq2Seq_LSTM': {'adjacencyMatrixLaplacianMatrix': [None]},
    'Seq2Seq_GRU': {'adjacencyMatrixLaplacianMatrix': [None]},
    'DCRNN': {'adjacencyMatrixLaplacianMatrix': ['scaled_lap_sym']},
    'ASTGCN': {'adjacencyMatrixLaplacianMatrix': ['cheb_poly'], },
    'GWNet': {'adjacencyMatrixLaplacianMatrix': ['dual_rand_mat_asym']},
    'AGCRN': {'adjacencyMatrixLaplacianMatrix': [None]},
    'DSTAGNN': {'adjacencyMatrixLaplacianMatrix': ['cheb_poly']},
    'STGODE': {'adjacencyMatrixLaplacianMatrix': ['raw_thresholded_gaussian']},
    'DGCRN': {'adjacencyMatrixLaplacianMatrix': ['dual_rand_mat_asym']},
    'TGCRN': {'adjacencyMatrixLaplacianMatrix': [None]},
}

DEFAULT = {
    # about dataset
    'trainTestRatio': 0.7,
    'adjacencyMatrixThresholdValue': 0.7,

    # about model
    'maxEpoch': 100,
    # 'maxEpoch': 1,  # for SPATS testing
    'batchSize': 64,
    # 'batchSize': 64,  # for SPATS testing

    # about loss function
    'lossFunction': ["MAE", "MSE", "MAAPE"],  # should be list
    'targetLossFunction': "MSE",

    # about optimizer
    'optimizer': 'Adam',
    'learningRate': 0.001,
    'weightDecay': 0.0005,
}


VARIATION = {
    'datasetName': ['METR-LA', 'PEMS-BAY', 'PEMSD7', 'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08', 'Chickenpox', 'JP-Pref', 'US-Regions', 'US-States', 'Uzel2022', 'NOAA1000'],
    'modelName': ['AGCRN', 'ASTGCN', 'DCRNN', 'DGCRN', 'DSTAGNN', 'GWNet', 'STGODE', 'TGCRN', 'Seq2Seq_RNN', 'Seq2Seq_LSTM', 'Seq2Seq_GRU', 'ARMA', 'VAR'],
    # 'adjacencyMatrixThresholdValue': [0.0, 0.2, 0.4, 0.6, 0.8],
}


def stdin_generator(variation: dict) -> typing.Generator[dict, None, None]:
    # result = copy.deepcopy(default)
    variation_flat = pd.json_normalize(
        variation, sep='.').to_dict(orient='records')[0]

    # ignore list size is 0
    variation_flat = {key: value for (
        key, value) in variation_flat.items() if len(value) > 0}

    variation_flat_keys = [key for key in variation_flat.keys()]
    values = [values for (_, values) in variation_flat.items()]

    for element in itertools.product(*values):
        result = {}
        for (key, new_value) in zip(variation_flat_keys, element):
            if key in result.keys():
                if result[key] == new_value:
                    continue
                result[key] = new_value
                yield result
            else:
                result[key] = new_value
        yield result


def dependent_setting_generator(dataset_name: str, model_name: str) -> typing.Generator[typing.Tuple[str, typing.Any], None, None]:
    d_list = list(stdin_generator(DATASET_DEPENDENT_SETTING[dataset_name]))
    m_list = list(stdin_generator(MODEL_DEPENDENT_SETTING[model_name]))

    for d_setting, m_setting in itertools.product(d_list, m_list):
        # combine two dicts
        yield {**d_setting, **m_setting}


def main():
    log.basicConfig(
        format="[%(levelname)s][%(asctime)s.%(msecs)03d][%(filename)s:%(lineno)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=log.INFO)

    # condition check for dataset
    for dataset_name in VARIATION['datasetName']:
        assert dataset_name in DATASET_DEPENDENT_SETTING, \
            f'Error: datasetName {dataset_name} is not supported'

    # condition check for model
    for model_name in VARIATION['modelName']:
        assert model_name in MODEL_DEPENDENT_SETTING, \
            f'Error: modelName {model_name} is not supported'

    # generate all stdin from complete_variation
    for stdin in stdin_generator(VARIATION):
        stdin = {**DEFAULT, **stdin}
        dataset_name = stdin['datasetName']
        model_name = stdin['modelName']
        for dep_setting in dependent_setting_generator(dataset_name, model_name):
            stdin = {**stdin, **dep_setting}

            model_config_path = f"./common/model/{stdin['modelName']}.yaml"

            if os.path.exists(model_config_path):
                model_config = common.util.load_yaml(model_config_path)
                stdin['modelConfig'] = copy.deepcopy(model_config)
            else:
                stdin['modelConfig'] = {}

            # STDIN is generated.
            # for debugging, add print(stdin) here and comment out the following code.

            res = common.util.rest_json(
                'POST', ENDPOINT, '/api/job/control',
                query={
                    'request': 'insert',
                },
                body={
                    'argument': json.dumps(['training']),
                    'stdin': json.dumps(stdin),
                })

            if (res is None) or (res.status_code != 200):
                log.fatal('Broker seems down.')
                return

            res = res.json()
            log.info(f'Successfully insert job {res["jobID"]}')


if __name__ == '__main__':
    main()
