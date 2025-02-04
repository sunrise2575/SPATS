import yaml
import datetime
import typing
import requests
import json
import torch
import contextlib
import timeit
import numpy
import socket
import fcntl
import struct

def dict_put(d: dict, keys: str, item: any):
    if "." in keys:
        key, rest = keys.split(".", 1)
        if key not in d:
            d[key] = {}
        dict_put(d[key], rest, item)
    else:
        d[keys] = item


def dict_get(d: dict, keys: str) -> any:
    if "." in keys:
        key, rest = keys.split(".", 1)
        if key not in d:
            return None
        return dict_get(d[key], rest)
    else:
        return d[keys]


def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        a = yaml.load(f.read(), Loader=yaml.FullLoader)
    return a

def save_yaml(path: str, data: dict):
    with open(path, 'w') as f:
        yaml.dump(data, f, sort_keys=True)

def count_nan(x: numpy.ndarray) -> int:
    mask = numpy.isnan(x)
    nan_count = numpy.sum(mask)
    return nan_count
    
def count_nan_torch(x: torch.Tensor) -> int:
    mask = torch.isnan(x)
    mask_int= mask.int()
    nan_count = mask_int.sum().item()
    return nan_count

def gen_now_time_str() -> str:
    now = datetime.datetime.now(datetime.timezone.utc).astimezone()
    return now.strftime('%Y-%m-%dT%H:%M:%S')

def rest_json(method: str, endpoint: str, path: str, query: typing.Dict[str, str] = None, body: typing.Dict[str, str] = None) -> requests.Response:
    method = method.upper()
    url = f'http://{endpoint}{path}'
    headers = {"Content-Type": "application/json; charset=utf-8"}

    if body is not None:
        body = json.dumps(body)

    try:
        if method == 'GET':
            response = requests.get(url, params=query)
        elif method == 'POST':
            response = requests.post(
                url, headers=headers, params=query, data=body)
        elif method == 'PUT':
            response = requests.put(
                url, headers=headers, params=query, data=body)
        elif method == 'DELETE':
            response = requests.delete(url, params=query)
        else:
            raise ValueError(f'Invalid method: {method}')
    except requests.exceptions.ConnectionError:
        return None
    else:
        return response

@contextlib.contextmanager
def timer():
    start = timeit.default_timer()
    elapser = lambda: timeit.default_timer() - start
    yield lambda: elapser()
    end = timeit.default_timer()
    elapser = lambda: end-start

def get_tensor_byte_size(x: torch.Tensor):
    return x.numel() * x.element_size()
    
def get_state_dict_byte_size(x: typing.Dict[str, torch.Tensor]):
    return sum([get_tensor_byte_size(v) for v in x.values()])

def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', ifname[:15])
    )[20:24])
