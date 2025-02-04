import json
import os
import sqlite3
import sys
import typing

import numpy as np
import polars as pl
import tqdm


def convert_stdin(stdin: json) -> typing.Dict[str, typing.Any]:
    return {
        'datasetName': stdin['datasetName'],
        'trainTestRatio': stdin['trainTestRatio'],
        'adjacencyMatrixThresholdValue': stdin['adjacencyMatrixThresholdValue'],
        'adjacencyMatrixLaplacianMatrix': stdin['adjacencyMatrixLaplacianMatrix'],
        'inputLength': stdin['inputLength'],
        'outputLength': stdin['outputLength'],
        'modelName': stdin['modelName'],
        'modelConfig': json.dumps(stdin['modelConfig']),
        'maxEpoch': stdin['maxEpoch'],
        'batchSize': stdin['batchSize'],
        'learningRate': stdin['learningRate'],
        'weightDecay': stdin['weightDecay'],
    }


def convert_for_success_state(stdin: json, stdout: json) -> typing.List[typing.Dict[str, typing.Any]]:
    result = []

    PK = convert_stdin(stdin)

    for row in stdout:
        result.append({**PK,
                       'epoch': row['epoch'],

                       'train_MAE': row['train_MAE_mean'],
                       'train_MSE': row['train_MSE_mean'],
                       'train_MAAPE': row['train_MAAPE_mean'],
                       'test_MAE': row['test_MAE_mean'],
                       'test_MSE': row['test_MSE_mean'],
                       'test_MAAPE': row['test_MAAPE_mean'],

                       'train_time': row['time_train'],
                       'test_time': row['time_test'],
                       'max_memory': row['cuda_mem_usage'],
                       })

    return result


def convert_for_failure_state(stdin: json, stderr: str, version: str) -> typing.Dict[str, typing.Any]:
    PK = convert_stdin(stdin, version)
    return {**PK,
            'stderr': stderr,
            }


def main():
    assert len(sys.argv) == 3, "Usage: python extractor.py <db_file> <STATUS>"

    DB_PATH = os.path.abspath(sys.argv[1])
    STATUS = sys.argv[2].lower()

    assert os.path.exists(DB_PATH), f"File not found: {DB_PATH}"
    assert os.path.isfile(DB_PATH), f"Invalid file: {DB_PATH}"
    assert STATUS in ['success', 'failure'], "Invalid STATUS"

    # separate file's directory and file name and file extension
    db_dir, _db_file = os.path.split(DB_PATH)
    db_name, _ = os.path.splitext(_db_file)

    TABLE_NAME = 'jobs'

    conn = sqlite3.connect(DB_PATH)
    print(f"Read the DB file: {DB_PATH}")

    df = pl.read_database(
        query=f"SELECT * FROM {TABLE_NAME} WHERE status = '{STATUS}'", connection=conn)

    if len(df) == 0:
        # retry
        df = pl.read_database(
            query=f"SELECT * FROM {TABLE_NAME} WHERE status = '{STATUS.upper()}'", connection=conn)

        if len(df) == 0:
            print("No selected data")
            return

    print(f"Parsing the data...")

    result = []
    if STATUS == 'success':
        for i in tqdm.tqdm(range(len(df))):
            stdin = json.loads(df[i]['stdin'].item())
            stdout = json.loads(df[i]['stdout'].item())

            if len(stdin) == 0 or len(stdout) == 0:
                continue

            result += convert_for_success_state(stdin, stdout)

    elif STATUS == 'failure':
        for i in tqdm.tqdm(range(len(df))):
            stdin = json.loads(df[i]['stdin'].item())
            stderr = df[i]['stderr'].item()

            if len(stdin) == 0 or len(stderr) == 0:
                continue

            result.append(convert_for_failure_state(stdin, stderr))

    if len(result) == 0:
        print("No data to save")
        return

    print(f"Convert dataframe to CSV...")
    csv_path = os.path.join(db_dir, db_name + f'_{STATUS.lower()}.csv')
    pl.DataFrame(result, infer_schema_length=10000).write_csv(csv_path)
    print(f"Save converted CSV to {csv_path}")


if __name__ == '__main__':
    main()
