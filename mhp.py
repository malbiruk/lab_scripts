'''
set of utilities to obtain and plot mhp data
'''
# pylint: disable = too-many-arguments

import logging
import os
import subprocess
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs
import seaborn as sns
import typer
from modules.constants import EXPERIMENTS, PATH
from modules.general import (flatten, initialize_logging, multiproc,
                             progress_bar)
from modules.tg_bot import run_or_send_error
from modules.traj import System, TrajectorySlice
from scipy import stats

app = typer.Typer(rich_markup_mode='rich', add_completion=False)


def obtain_mhpmap(trj: TrajectorySlice, force: bool = False) -> bool:
    '''
    obtain mhp map for specified trajectory slice
    '''
    desired_datafile = (PATH / trj.system.name /
                        f'mhp_{trj.b}-{trj.e}-{trj.dt}' / '1_data.nmp')

    if desired_datafile.is_file() and force is False:
        logging.info(
            'data for %s already calculated, skipping...', trj.system.name)
        return True

    (PATH / trj.system.name / f'mhp_{trj.b}-{trj.e}-{trj.dt}').mkdir(
        parents=True, exist_ok=True)
    os.chdir(PATH / trj.system.name / f'mhp_{trj.b}-{trj.e}-{trj.dt}')
    tpr = str(PATH / trj.system.name / 'md' / trj.system.tpr)
    xtc = str(PATH / trj.system.name / 'md' / trj.system.xtc)

    args = f'TOP={str(tpr)}\nTRJ={str(xtc)}' \
        f'\nBEG={int(trj.b*1000)}\nEND={int(trj.e*1000)}\nDT={trj.dt}' \
        '\nNX=150\nNY=150' \
        f'\nMAPVAL="M"\nMHPTBL="98"\nPRJ="P"\nUPLAYER=1\nMOL="lip///"' \
        '\nSURFSEL=$MOL\nPOTSEL=$MOL\nDUMPDATA=1\nNOIMG=1'

    with open('args', 'w', encoding='utf-8') as f:
        f.write(args)

    impulse = Path('/nfs/belka2/soft/impulse/dev/inst/runtask.py')
    prj = Path(
        '/home/krylov/Progs/IBX/AMMP/test/postpro/maps/galaxy/new/prj.json')
    cmd = f'{impulse} -f args -t {prj}'
    msg = f'couldn\'t obtain mhp data for `{trj.system.name}`'

    if run_or_send_error(cmd, msg,
                         stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT):
        logging.info('sucessfully calculated mhp data for %s', trj.system.name)
        return True

    logging.error('couldn\'t calculate mhp data for %s', trj.system.name)
    return False


@app.command()
def get(ctx: typer.Context,
        force: bool = typer.Option(
        False, help='override already calculated mhp data'),
        ):
    '''
    run [bold]impulse cont_stat[/] on all systems to obtain mhp info
    '''
    xtc, tpr, b, e, dt, n_workers, verbose, messages = ctx.obj

    initialize_logging('get_mhp.log', verbose)

    systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in flatten(EXPERIMENTS.values())])
    systems = list(dict.fromkeys(systems))
    trajectory_slices = [TrajectorySlice(System(PATH, s, xtc, tpr), b, e, dt)
                         for s in systems]

    logging.info('started mhp data calculation')
    multiproc(obtain_mhpmap,
              trajectory_slices,
              (force for _ in trajectory_slices),
              n_workers=n_workers,
              messages=messages,
              descr='get mhp data'
              )
    logging.info('mhp data calculation done')


def calc_fractions_all(trj_slices: list) -> None:
    '''
    calculate mhp fractions for all systems and save them as csv
    '''
    df_dict = {
        'index': [],
        'system': [],
        'timepoint': [],
        'CHL amount, %': [],
        'phob': [],
        'phil': [],
        'neutr': []}

    with progress_bar as p:
        for trj in p.track(trj_slices, description='collecting mhp data'):
            try:
                data = np.load(
                    PATH / trj.system.name /
                    f'mhp_{trj.b}-{trj.e}-{trj.dt}' / '1_data.nmp')['data']
            except FileNotFoundError:
                logging.error('couldn\'t find 1_data.nmp file for %s',
                              trj.system.name)
                continue
            for c, i in enumerate(data):
                df_dict['index'].append(trj.system.name)
                df_dict['system'].append(trj.system.name.split('_chol', 1)[0])
                if len(trj.system.name.split('_chol', 1)) == 1:
                    df_dict['CHL amount, %'].append(0)
                else:
                    df_dict['CHL amount, %'].append(
                        trj.system.name.split('_chol', 1)[1])
                df_dict['timepoint'].append(trj.b + trj.dt * c)
                i = i.flatten()
                phob = i[i >= 0.5].shape[0]
                phil = i[i <= -0.5].shape[0]
                neutr = i.shape[0] - phil - phob
                df_dict['phob'].append(phob / i.shape[0])
                df_dict['phil'].append(phil / i.shape[0])
                df_dict['neutr'].append(neutr / i.shape[0])
            logging.info('succesfully calculated mhp fractions for %s',
                         trj.system.name)

    df = pd.DataFrame(df_dict)

    df.to_csv(
        PATH / 'notebooks' / 'mhpmaps' /
        'for_hists_fractions_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv',
        index=False)

    df_stats = (
        df.groupby(['index', 'system', 'CHL amount, %'], as_index=False)
        .mean(numeric_only=True)
        .assign(phob_std=df.groupby(
            ['index', 'system', 'CHL amount, %'])
            .std(numeric_only=True)['phob'].values)
        .assign(phil_std=df.groupby(
                ['index', 'system', 'CHL amount, %'])
                .std(numeric_only=True)['phil'].values)
        .assign(neutr_std=df.groupby(
                ['index', 'system', 'CHL amount, %'])
                .std(numeric_only=True)['neutr'].values))

    df_stats.to_csv(
        PATH / 'notebooks' / 'mhpmaps' /
        'for_hists_fractions_stats_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv',
        index=False)

    logging.info('calculating mhp area fractions done')


def draw_mhp_area_bars(ax: plt.axis, df: pd.DataFrame, systs: list,
                       width: float, positions: tuple,
                       alpha: float, label: str):
    '''
    create single bar plot for df with columns
    "phob_fr", "phob_std", "phil_fr"...
    '''
    if label is not None:
        labels = ['hydrophobic, ' + label,
                  'neutral, ' + label,
                  'hydrophilic, ' + label,
                  ]
    else:
        labels = [None, None, None]

    ax.bar(positions[0], df.loc[systs, :]['phob'], width,
           yerr=df.loc[systs, :]['phob_std'], capsize=5,
           label=labels[0], color='C0', alpha=alpha)
    ax.bar(positions[1], df.loc[systs, :]['neutr'], width,
           yerr=df.loc[systs, :]['neutr_std'], capsize=5,
           label=labels[1], color='C1', alpha=alpha)
    ax.bar(positions[2], df.loc[systs, :]['phil'], width,
           yerr=df.loc[systs, :]['phil_std'], capsize=5,
           label=labels[2], color='C2', alpha=alpha)

    # single black edges independent on alpha
    ax.bar(positions[0], df.loc[systs, :]['phob'], width,
           yerr=df.loc[systs, :]['phob_std'], capsize=5, ec='k',
           fill=False, lw=2)
    ax.bar(positions[1], df.loc[systs, :]['neutr'], width,
           yerr=df.loc[systs, :]['neutr_std'], capsize=5, ec='k',
           fill=False, lw=2)
    ax.bar(positions[2], df.loc[systs, :]['phil'], width,
           yerr=df.loc[systs, :]['phil_std'], capsize=5, ec='k',
           fill=False, lw=2)


def plot_mhp_area_single_exp(ax: plt.axis, df: pd.DataFrame, exp: str,
                             show_label: bool):
    '''
    draw mhp area bars for single experiment
    '''
    x = np.arange(len(EXPERIMENTS[exp]))
    width = 0.07

    positions = (
        (x - 6 * width, x - 2 * width, x + 2 * width),
        (x - 5 * width, x - 1 * width, x + 3 * width),
        (x - 4 * width, x, x + 4 * width),
        (x - 3 * width, x + 1 * width, x + 5 * width),
    )

    alphas = (1, 0.5, 0.3, 0.1)

    systems = (
        EXPERIMENTS[exp],
        [i + '_chol10' for i in EXPERIMENTS[exp]],
        [i + '_chol30' for i in EXPERIMENTS[exp]],
        [i + '_chol50' for i in EXPERIMENTS[exp]],
    )

    if show_label:
        labels = ('0% CHL', '10% CHL', '30% CHL', '50% CHL')
    else:
        labels = (None, None, None, None)

    for syst, pos, alpha, label in zip(systems, positions, alphas, labels):
        draw_mhp_area_bars(ax, df, syst, width, pos, alpha, label)

    x = np.arange(len(EXPERIMENTS[exp]))
    ax.set_title(exp)
    ax.xaxis.set_ticks(x)
    ax.set_xticklabels(EXPERIMENTS[exp])


def plot_mhp_ratio_all(trj_slices):
    '''
    plot mhp fractions ratios for all systems
    '''
    logging.info('plotting area hists...')
    df = pd.read_csv(
        PATH / 'notebooks' / 'mhpmaps' /
        'for_hists_fractions_stats_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')

    subdf = df[['phob', 'phil',	'neutr', 'phob_std', 'phil_std', 'neutr_std']]
    subdf = subdf * 100
    df.set_index('index', inplace=True)

    fig, axs = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    for ax, exp in zip(axs, EXPERIMENTS):
        plot_mhp_area_single_exp(ax, df, exp, ax == axs[0])
        ticks, labels = ax.get_xticks(), ax.get_xticklabels()
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45,
                           ha='right', rotation_mode='anchor')
    axs[0].set_ylabel('% of area')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)

    plt.savefig(PATH / 'notebooks' / 'mhpmaps' / 'imgs' /
                f'mhp_hists_area_dt{trj_slices[0].dt}.png',
                bbox_inches='tight', dpi=300)
    logging.info('done.')


def stats_by_mhp(df: pd.DataFrame) -> pd.DataFrame:
    '''
    compare phob neutr phil columns in df
    using Kruskal-Wallis test followed by post-hoc Dunn test with Holm p-adjust
    '''
    dict_by_mhp = {
        'index': [],
        'system': [],
        'CHL amount, %': [],
        'phob vs neutr': [],
        'neutr vs phil': [],
        'phil vs phob': []
    }

    for syst in df['index'].unique():

        data = [df[df['index'] == syst]['phob'],
                df[df['index'] == syst]['neutr'],
                df[df['index'] == syst]['phil']]

        kruskal_p = stats.kruskal(*data).pvalue
        if kruskal_p < 0.01:
            p_values = scikit_posthocs.posthoc_dunn(data, p_adjust='holm')
        else:
            p_values = pd.DataFrame({1: [1, kruskal_p, kruskal_p],
                                     2: [kruskal_p, 1, kruskal_p],
                                     3: [kruskal_p, kruskal_p, 1]})
            p_values.index += 1

        dict_by_mhp['index'].append(syst)
        dict_by_mhp['system'].append(syst.split('_chol', 1)[0])
        dict_by_mhp['CHL amount, %'].append(
            '0'
            if len(syst.split('_chol', 1)) == 1
            else syst.split('_chol', 1)[1])
        dict_by_mhp['phob vs neutr'].append(p_values.loc[1, 2])
        dict_by_mhp['neutr vs phil'].append(p_values.loc[2, 3])
        dict_by_mhp['phil vs phob'].append(p_values.loc[1, 3])

    return pd.DataFrame(dict_by_mhp)


def stats_by_chl_amount(df: pd.DataFrame) -> pd.DataFrame:
    '''
    compare 0 10 30 50 CHL amount in df for phob, neutr, phil fractions
    using Kruskal-Wallis test followed by post-hoc Dunn test with Holm p-adjust
    '''
    dict_by_mhp = {
        'system': [],
        'mhp': [],
        '0 vs 10': [],
        '10 vs 30': [],
        '30 vs 50': [],
    }

    for syst in df['system'].unique():
        for mhp in ('phil', 'neutr', 'phob'):
            data = [df[(df['system'] == syst)
                       & (df['CHL amount, %'] == 0)][mhp],
                    df[(df['system'] == syst)
                       & (df['CHL amount, %'] == 10)][mhp],
                    df[(df['system'] == syst)
                       & (df['CHL amount, %'] == 30)][mhp],
                    df[(df['system'] == syst)
                       & (df['CHL amount, %'] == 50)][mhp]]
            kruskal_p = stats.kruskal(*data).pvalue
            if kruskal_p < 0.01:
                p_values = scikit_posthocs.posthoc_dunn(data, p_adjust='holm')
            else:
                p_values = pd.DataFrame(
                    {1: [1, kruskal_p, kruskal_p, kruskal_p],
                     2: [kruskal_p, 1, kruskal_p, kruskal_p],
                     3: [kruskal_p, kruskal_p, 1, kruskal_p],
                     4: [kruskal_p, kruskal_p, kruskal_p, 1]})
                p_values.index += 1


def calc_mhp_area_all_stats(trj_slices: list):
    '''
    compare 0 10 30 50 CHL for phob phil neutr
    compare phob phil neutr for 0 10 30 50 CHL
    compare systems in exp (each mhp each CHL amount)
    '''
    logging.info('calculating statistics...')
    df = pd.read_csv(
        PATH / 'notebooks' / 'mhpmaps' /
        'for_hists_fractions_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')

    # compare 0 10 30 50 CHL for phob phil neutr
    logging.info('by mhp difference...')
    df_by_mhp = stats_by_mhp(df)
    df_by_mhp.to_csv(
        PATH / 'notebooks' / 'mhpmaps' / 'stats' /
        'area_all_by_mhp_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv',
        index=False
    )

    # compare phob phil neutr for 0 10 30 50 CHL
    logging.info('by CHL amount difference...')
    df_by_mhp = stats_by_chl_amount(df)


# %%

def mhp_all(trj_slices: list,
            calculate: bool, plot: bool, stat: bool) -> None:
    '''
    calculate and plot ratio of areas with different mhp values
    also calculate statistics
    '''
    if calculate:
        calc_fractions_all(trj_slices)
    if plot:
        plot_mhp_ratio_all(trj_slices)
    if stat:
        calc_mhp_area_all_stats(trj_slices)


def mhp_chol(trj_slices: list,
             calculate: bool, plot: bool, stat: bool) -> None:
    '''
    calculate and plot ratio of areas with different mhp values
    composed by atoms of CHL on the surface
    also calculate statistics
    '''


@app.command()
def plot_area_fractions(
    ctx: typer.Context,
    fraction_types: Tuple[bool, bool, bool, bool] = typer.Option(
        (True, True, True, True),
        help='tasks to run [dim](pick from '
        '[bold]mhp_all, mhp_chol, chol_all, mhp_time[/])[/]'),
    calculate: bool = typer.Option(
        True, help='calculate and dump fractions from scratch'),
    plot: bool = typer.Option(
        True, help='plot fractions'),
    stat: bool = typer.Option(
        True, help='calculate statistics'),
):
    '''
    plot hydrophilic/neutral/hydrophobic area fractions ratios for bilayers
    '''
    sns.set(style='ticks', context='talk', palette='muted')

    xtc, tpr, b, e, dt, _, verbose, _ = ctx.obj
    initialize_logging('plot_mhp_fractions.log', verbose)
    systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in flatten(EXPERIMENTS.values())])
    systems = list(dict.fromkeys(systems))
    trajectory_slices = [TrajectorySlice(System(PATH, s, xtc, tpr), b, e, dt)
                         for s in systems]

    if fraction_types[0]:
        mhp_all(trajectory_slices, calculate, plot, stat)

    if fraction_types[1]:
        mhp_chol(trajectory_slices, calculate, plot, stat)


@app.command()
def plot_mhp_hists():
    pass


@app.command()
def clust():
    pass


@app.callback()
def callback(ctx: typer.Context,
             xtc: str = typer.Option(
                 'pbcmol_201.xtc', '-xtc', help='name of trajectory files',
                 rich_help_panel='Trajectory parameters'),
             tpr: str = typer.Option(
                 '201_ns.tpr', '-tpr', help='name of topology files',
                 rich_help_panel='Trajectory parameters'),
             b: float = typer.Option(
                 200, '-b', help='beginning of trajectories (in ns)',
                 rich_help_panel='Trajectory parameters'),
             e: float = typer.Option(
                 201, '-e', help='end of trajectories (in ns)',
                 rich_help_panel='Trajectory parameters'),
             dt: int = typer.Option(
                 1, '-dt', help='timestep of trajectories (in ps)',
                 rich_help_panel='Trajectory parameters'),
             n_workers: int = typer.Option(
                 8, help='n of processes to start for each task',
                 rich_help_panel='Script config'),
             verbose: bool = typer.Option(
                 False, '--verbose', '-v', help='print debug log',
                 rich_help_panel='Script config'),
             messages: bool = typer.Option(
                 True, help='send updates info in telegram',
                 rich_help_panel='Script config')):
    '''set of utilities to obtain and plot mhp data'''
    # store command line arguments
    ctx.obj = (xtc, tpr, b, e, dt, n_workers, verbose, messages)


# %%

if __name__ == '__main__':
    app()


# %%
# systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
#                    for i in flatten(EXPERIMENTS.values())])
# systems = list(dict.fromkeys(systems))
# trajectory_slices = [TrajectorySlice(System(
#     PATH, s, 'pbcmol_201.xtc', '201_ns.tpr'),
#     200, 201, 1) for s in systems]
trj_slices = trajectory_slices
