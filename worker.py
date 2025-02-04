import argparse
import json
import logging as log
import multiprocessing
import multiprocessing.managers
import multiprocessing.shared_memory
import os
import socket
import subprocess
import time
import typing

import torch

import common

# global variables
# CURRENT_JOB_ID = multiprocessing.shared_memory.ShareableList( [None for _ in range(torch.cuda.device_count())])
MY_HOSTNAME = socket.gethostname()
BROKER_ADDRESS = '127.0.0.1:9999'
CONF_YAML_ABSPATH = os.path.abspath('./config.yaml')
POLLING_TIME = 0.1  # seconds


def report_result(my_state: typing.Dict[str, typing.Any], max_loop: int = None):
    assert max_loop is None or max_loop > 0

    if (my_state['jobID'] is None) or (my_state['jobID'] == '') or (my_state['status'] is None):
        return

    loop = 0
    while True:
        res = common.util.rest_json(
            'POST', BROKER_ADDRESS, '/api/job/process',
            query={'request': 'set'},
            body=my_state,
        )

        if (res is not None) and (res.status_code == 200):
            if my_state['status'] == 'success' or my_state['status'] == 'failure':
                log.info(
                    f"Report the job {my_state['jobID']} to Broker at {BROKER_ADDRESS}.")
            else:
                log.info(
                    f"Report the INCOMPLETE job {my_state['jobID']} to Broker at {BROKER_ADDRESS}. This job could be resumed.")
            return

        if max_loop is not None:
            loop += 1
            if loop >= max_loop:
                log.warning(
                    f"CANNOT report the job {my_state['jobID']} to Broker at {BROKER_ADDRESS}. Please manually check the job info such as the result folder.")
                return

        # wait until Broker is UP state
        time.sleep(POLLING_TIME)


def runner_spawner(
        JOB_STATUS: multiprocessing.shared_memory.ShareableList,
        JOB_ID: multiprocessing.shared_memory.ShareableList,
        gpu_id: int):

    log.info(f"Runner spawner for GPU {gpu_id} is started.")

    while True:
        # reset
        JOB_STATUS[gpu_id] = 'incomplete'
        JOB_ID[gpu_id] = None

        # get the job
        res = common.util.rest_json(
            'POST', BROKER_ADDRESS, '/api/job/process',
            query={
                'request': 'get',
            },
            body={
                'workerHostname': MY_HOSTNAME,
                'workerGPUID': gpu_id,
            },
        )

        if (res is None) or (res.status_code != 200):
            # wait until Broker is UP state
            time.sleep(POLLING_TIME)
            continue

        body = res.json()
        if body is None:
            # wait until Broker is UP state
            time.sleep(POLLING_TIME)
            continue

        JOB_ID[gpu_id] = str(body['jobID'])
        if JOB_ID[gpu_id] == '':
            # nothing to process, keep polling
            time.sleep(POLLING_TIME)
            continue

        # argument: typing.List[str] = json.loads(body['argument'])
        stdin: str = body['stdin']

        cmd = ['python3', './runner.py', JOB_ID[gpu_id]]  # + argument

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # run the process (TODO: receive stdout and stderr separately)
        log.info(
            f"Spawn a Runner on GPU {gpu_id} for a job {JOB_ID[gpu_id]}.")

        proc = subprocess.run(
            'bash -c "source activate base; ' +
            ' '.join(cmd) + '"',
            shell=True,
            input=bytes(stdin, 'utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        stdout = proc.stdout.decode('utf-8')
        stderr = proc.stderr.decode('utf-8')

        if ((len(stderr) > 0) and (len(stdout) == 0)):
            if 'cuda out of memory' in stderr.lower():
                batch_size = json.loads(stdin)['batchSize']
                if batch_size > 1:
                    log.info(
                        f"The job {JOB_ID[gpu_id]} is failed due to CUDA out of memory at batch size = {batch_size}. Join the Runner in GPU {gpu_id}. This job will be restarted by Broker with reduced batch size.")
                else:
                    log.info(
                        f"The job {JOB_ID[gpu_id]} is failed due to CUDA out of memory at batch size = {batch_size}. Join the Runner in GPU {gpu_id}. This job will be marked as failure.")
                    JOB_STATUS[gpu_id] = 'failure'
            else:
                log.info(
                    f"The job {JOB_ID[gpu_id]} is failed. Join the Runner in GPU {gpu_id}. Reason: {stderr}")
            JOB_STATUS[gpu_id] = 'failure'
        else:
            log.info(
                f"The job {JOB_ID[gpu_id]} is successful. Join the Runner in GPU {gpu_id}.")
            JOB_STATUS[gpu_id] = 'success'

        report_result({
            'status': JOB_STATUS[gpu_id],
            'jobID': JOB_ID[gpu_id],
            'workerHostname': MY_HOSTNAME,
            'workerGPUID': gpu_id,
            'stdout': stdout,
            'stderr': stderr,
        })


def main():
    # init --------------------------------------------------
    log.basicConfig(
        format="[%(levelname)s][%(asctime)s.%(msecs)03d][%(filename)s:%(lineno)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=log.INFO)

    total_gpus = torch.cuda.device_count()
    if total_gpus == 0:
        log.fatal('No GPU found, exiting...')
        return

    parser = argparse.ArgumentParser()
    parser.add_argument('--broker', dest='broker_address',
                        default='127.0.0.1:9999')
    parser.add_argument('--gpu', dest='gpu_ids',
                        default=','.join([str(i) for i in range(total_gpus)]))
    args = vars(parser.parse_args())

    global BROKER_ADDRESS
    BROKER_ADDRESS = args['broker_address']
    selected_gpu_ids = args['gpu_ids'].split(',')

    log.info(f'Broker address is {BROKER_ADDRESS}')
    log.info(
        f'{len(selected_gpu_ids)} GPU(s) are selected; IDs are {", ".join(selected_gpu_ids)}')

    JOB_STATUS = multiprocessing.shared_memory.ShareableList(
        ['incomplete' for _ in range(total_gpus)])
    JOB_ID = multiprocessing.shared_memory.ShareableList(
        [None for _ in range(total_gpus)])

    process = [multiprocessing.Process(
        target=runner_spawner, args=(JOB_STATUS, JOB_ID, int(gpu_id),)) for gpu_id in selected_gpu_ids]

    conf = common.util.load_yaml(CONF_YAML_ABSPATH)

    # run --------------------------------------------------
    try:
        for p in process:
            p.start()

        for p in process:
            p.join()

        log.info(f'All Runner spawners are terminated.')
    except:

        for p in process:
            if p.is_alive():
                p.terminate()

        for gpu_id in range(total_gpus):
            if JOB_ID[gpu_id] is None:
                continue

            log_path = os.path.join(
                os.path.dirname(CONF_YAML_ABSPATH),
                conf['location']['result'],
                JOB_ID[gpu_id],
                "log.json")

            stdout = None
            if os.path.exists(log_path) and os.path.isfile(log_path):
                with open(log_path, 'r') as f:
                    stdout = f.read()

            state = {
                'status': JOB_STATUS[gpu_id],
                'jobID': JOB_ID[gpu_id],
                'workerHostname': MY_HOSTNAME,
                'workerGPUID': gpu_id,
                'stdout': stdout,
                'stderr': None,
            }

            report_result(state, max_loop=5)

        JOB_STATUS.shm.close()
        JOB_STATUS.shm.unlink()
        JOB_ID.shm.close()
        JOB_ID.shm.unlink()


if __name__ == '__main__':
    main()
