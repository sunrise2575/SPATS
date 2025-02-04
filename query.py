import copy
import json
import logging as log
import os
import sys
import typing

import pandas

import common.util

BROKER_IP = '127.0.0.1'
BROKER_PORT = '9999'
ENDPOINT = f'{BROKER_IP}:{BROKER_PORT}'


def select_all(subcommand: typing.List[str]):
    if len(subcommand) == 0:
        # list all jobs
        res = common.util.rest_json(
            'GET', ENDPOINT, '/api/job',
            query={
                'pageStart': 0,
                'count': 100,
                'status': '',
            })

        if (res is None) or (res.status_code != 200):
            log.fatal('Broker seems down.')
            return

        res = res.json()
        if len(res['data']) == 0:
            log.info(f'No entry in Broker')
            return

        df = pandas.DataFrame(res['data'], columns=res['header'])
        df = df.set_index('id').sort_index()
        print(df)

    elif subcommand[0] in ['started', 'incomplete', 'success', 'failure']:
        # list specific jobs
        res = common.util.rest_json(
            'GET', ENDPOINT, '/api/job',
            query={
                'pageStart': 0,
                'count': 100,
                'status': subcommand[0],
            })

        if (res is None) or (res.status_code != 200):
            log.fatal('Broker seems down.')
            return

        res = res.json()
        if len(res['data']) == 0:
            log.info(f'No entry in Broker')
            return
        df = pandas.DataFrame(res['data'], columns=res['header'])
        df = df.set_index('id').sort_index()
        print(df)

    else:
        log.fatal(
            'wrong command, use: python ./query.py list <<None>|started|incomplete|success|failure>')
        exit(1)


def select(subcommand: typing.List[str]):
    if len(subcommand) == 0:
        log.fatal('wrong command, use: python ./query.py select <job_id>')
        return

    job_id = subcommand[0]

    # list a job
    res = common.util.rest_json(
        'GET', ENDPOINT, '/api/job/detail',
        query={
            'jobID': job_id,
        })

    if (res is None) or (res.status_code != 200):
        log.fatal('Broker seems down.')
        return

    res = res.json()
    if len(res['data']) == 0:
        log.info(f'No entry in Broker')
        return

    df = pandas.DataFrame(res['data'], columns=res['header'])
    df = df.set_index('id').sort_index()
    print(df)


def _parse_ranges(ranges_str: str) -> typing.List[int]:
    result = []
    ranges = ranges_str.split(',')
    for part in ranges:
        if part.isdigit():
            result.append(int(part))
        elif '..' in part:
            start, end = part.split('..')
            if not (start.isdigit() and end.isdigit()):
                continue
            start, end = int(start), int(end)
            if start > end:
                continue
            result.extend(range(int(start), int(end) + 1))
    return sorted(set(result))


def delete(subcommand: typing.List[str]):
    assert len(subcommand) > 0, 'Wrong argument; delete <job_id1>..<job_id2>'

    requested_job_ids = _parse_ranges(subcommand[0])
    assert len(requested_job_ids) > 0, 'Wrong argument; no jobs to delete.'

    job_id = subcommand[0]

    # delete a jobs
    for job_id in requested_job_ids:
        # delete a job
        res = common.util.rest_json(
            'POST', ENDPOINT, '/api/job/control',
            query={
                'request': 'delete',
            },
            body={
                'jobID': job_id,
            })

        if (res is None) or (res.status_code != 200):
            log.fatal('Broker seems down.')
            return

        log.info(f'Successfully delete job {job_id}')


def main():
    if len(sys.argv) < 2:
        log.fatal('wrong command, use: python ./query.py <list|select|delete>')
        exit(1)

    log.basicConfig(
        format="[%(levelname)s][%(asctime)s.%(msecs)03d][%(filename)s:%(lineno)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=log.INFO)

    if sys.argv[1] == 'list':
        select_all(sys.argv[2:])
    elif sys.argv[1] == 'select':
        select(sys.argv[2:])
    elif sys.argv[1] == 'delete':
        delete(sys.argv[2:])


if __name__ == '__main__':
    main()
