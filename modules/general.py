'''
general functions which are often used
'''

import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable
from pathlib import PosixPath
from datetime import datetime
import numpy as np


def print_1line(str_: str, line_length=30):
    print(str_ + (line_length - len(str_)) * ' ', end='\r')


def realtime_output(cmd: str):
    '''
    a wrapper for shell commands which outputs all shell output to shell instantly
    '''
    with subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    ) as process:
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip(), flush=True)


def opener(inp: PosixPath) -> list:
    '''
    open text file as list of lines
    '''
    with open(inp, 'r', encoding='utf-8') as f:
        lines = [i.strip() for i in f.read().strip().split('\n')]
    return lines


def chunker(seq: list, size: int) -> list:
    '''
    returns sub-list of size "size"
    '''
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def multiproc(func: Callable, *args, n_workers: int = 8) -> dict:
    '''
    wrapper for ProcessPoolExecutor,
    gets function, values of arguments (as iterables) and max n of workers,
    gives dictionary {tuple of arguments: result, ...}
    '''
    result = {}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(func, *i): i for i in zip(*args)}
        for f in as_completed(futures.keys()):
            result[futures[f]] = f.result()

    return result


def flatten(lst: list) -> list:
    '''
    flattens list (as np.flatten())
    '''
    return [item for sublist in lst for item in sublist]


def get_keys_by_value(val, dict) -> tuple:
    '''
    get keys of dict (as tuple) by value
    '''
    res = []
    for k, v in dict.items():
        if val in v:
            res.append(k)
    return tuple(res)


def calc_1d_com(x, m):
    '''
    calculate center of mass of 1D array
    '''
    return np.sum(x * m) / np.sum(m)


def duration(func: Callable) -> Callable:
    '''
    decorator which shows time of execution of inner function
    '''
    def inner(*args, **kwargs):
        start_time = datetime.now()
        func(*args, **kwargs)
        end_time = datetime.now()
        print(f'\n⌛ duration: {end_time - start_time}')
    return inner


def sparkles(func: Callable) -> Callable:
    '''
    decorator which prints sparkles before and after output of inner function
    '''
    def inner(*args, **kwargs):
        print('\n' + '✨' * 30 + '\n')
        func(*args, **kwargs)
        print('\n' + '✨' * 30)
    return inner
