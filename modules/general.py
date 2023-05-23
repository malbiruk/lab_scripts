'''
general functions which are often used
'''

import logging
import multiprocessing
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from datetime import timedelta as td
from pathlib import PosixPath
from typing import Callable

import numpy as np
import rich
from modules.tg_bot import send_message
from rich.logging import RichHandler
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           ProgressColumn, SpinnerColumn, Text, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)


class SpeedColumn(ProgressColumn):
    '''speed column for progress bar'''

    def render(self, task: "Task") -> Text:
        if task.speed is None:
            return Text('- it/s', style='red')
        return Text(f'{task.speed:.2f} it/s', style='red')


progress_bar = Progress(
    TextColumn('[bold]{task.description}'),
    SpinnerColumn('simpleDots'),
    TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn('|'),
    SpeedColumn(),
    TextColumn('|'),
    TimeElapsedColumn(),
    TextColumn('|'),
    TimeRemainingColumn(),
)


def initialize_logging(fname: str = 'out.log', debug: bool = False) -> None:
    '''
    initialize logging (default to file 'out.log'
    in folder + rich to command line)
    '''
    filehandler = logging.FileHandler(fname)
    filehandler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)-8s %(message)s'))
    richhandler = RichHandler(show_path=False)
    richhandler.setFormatter(logging.Formatter('%(message)s'))
    handlers = [filehandler, richhandler]
    if debug:
        logging.basicConfig(level=logging.DEBUG, handlers=handlers)
    else:
        logging.basicConfig(level=logging.INFO, handlers=handlers)


def print_1line(str_: str, line_length=30):
    '''print in sngle line'''
    print(str_ + (line_length - len(str_)) * ' ', end='\r')


def realtime_output(cmd: str):
    '''
    a wrapper for shell commands which outputs all shell output
    to shell instantly
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


def multiproc_show_multiple_progress(
        func: Callable, *args, n_workers: int = 8,
        descr: str = 'Working',  messages: bool = False):
    '''
    helper function for multiproc, executed when show_progress == multiple
    '''
    #pylint: disable=too-many-locals
    result = {}

    with progress_bar as p:
        futures = {}

        with multiprocessing.Manager() as manager:
            _progress = manager.dict()
            overall_task = p.add_task(descr, total=len(args[0]))
            task = p.tasks[-1]

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(
                    func,
                    _progress,
                    p.add_task(f'task {c}', visible=False),  # tak_id
                    *i): i for c, i in enumerate(zip(*args))}

                completed = 0
                while (n_finished := sum(
                        (f.done() for f in futures))) < len(futures):
                    p.update(overall_task, completed=n_finished,
                             total=len(futures))
                    if n_finished != completed:
                        completed = n_finished
                        if messages:
                            send_message(
                                f'task "{task.description}": '
                                f'{task.completed}/{task.total} '
                                'steps completed')
                    for task_id, update_data in _progress.items():
                        # update the progress bar for this task:
                        p.update(
                            task_id,
                            completed=update_data['progress'],
                            total=update_data['total'],
                            visible=(update_data['progress']
                                     < update_data['total']),
                        )
                p.update(overall_task, completed=len(futures),
                         total=len(futures))
                if messages:
                    send_message(
                        f'task "{task.description}" completed\n'
                        f'finished time: {task.finished_time}\n'
                        f'finished speed: {task.finished_speed}')

                try:
                    for f, _ in futures.items():
                        result[futures[f]] = f.result()
                except TypeError as e:
                    rich.print(f'[bold red]Can\'t return results:[/]'
                               f'\n{str(e)}')
    return result


def multiproc(func: Callable, *args, n_workers: int = 8,
              descr: str = 'Working', show_progress: str = 'single',
              messages: bool = False) -> dict:
    '''
    wrapper for ProcessPoolExecutor,
    gets function, values of arguments (as iterables) and max n of workers,
    gives dictionary {tuple of arguments: result, ...}

    show_progress: single, multiple, no

    if show_progress multiple, func should have special structure
    and progress bar will show subtasks:

    def func(progress: dict, task_id: int, *args):
        len_of_task = len({iterable})
        for c, _ in enumerate({iterable}):
            ...
            progress[task_id] = {'progress': c+1, 'total': len_of_task}
        return ...
    '''

    if show_progress == 'multiple':
        return multiproc_show_multiple_progress(
            func, *args,
            n_workers=n_workers,
            descr=descr,
            messages=messages)

    result = {}

    with progress_bar as p:
        task_id = p.add_task(descr, total=len(
            args[0]), visible=show_progress == 'single')
        task = p.tasks[-1]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(func, *i): i for i in zip(*args)}
            for f in as_completed(futures.keys()):
                p.update(task_id, advance=1)
                if messages:
                    send_message(
                        f'task "{task.description}": '
                        f'{task.completed}/{task.total} steps completed')
                try:
                    result[futures[f]] = f.result()
                except TypeError as e:
                    rich.print(f'[bold red]Can\'t return results:[/]'
                               f'\n{str(e)}')
        if messages:
            send_message(
                f'task "{task.description}" completed\n'
                f'finished time: {str(td(seconds=task.finished_time))} s')
    return result


def find_optimal_n_workers(multiproc_func: Callable,
                           n_min: int, n_max: int, step: int,
                           *args, **kwargs) -> int:
    '''
    this function runs multiproc with all arguments several times with different
    n_workers and determines the best based on speed of completing the task
    '''
    def measure_performance(n_workers: int, multiproc_func: Callable,
                            *args, **kwargs):
        start_time = time.time()
        multiproc_func(n_workers=n_workers, *args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        rich.print(f'n: {n_workers}, time: {execution_time}')
        return execution_time

    results = {}
    # Test different values of n_workers
    for n_workers in range(n_min, n_max, step):
        execution_time = measure_performance(n_workers, multiproc_func,
                                             *args, **kwargs)
        results[n_workers] = execution_time
    # Find the configuration with the minimum execution time
    optimal_workers = min(results, key=results.get)

    return optimal_workers, results


def flatten(lst: list) -> list:
    '''
    flattens list (as np.flatten())
    '''
    return [item for sublist in lst for item in sublist]


def get_keys_by_value(val, dict_) -> tuple:
    '''
    get keys of dict (as tuple) by value
    '''
    res = []
    for k, v in dict_.items():
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
