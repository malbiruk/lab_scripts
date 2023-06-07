'''
set of utilities to obtain and plot mhp data
'''


import logging
import os
import subprocess
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import scikit_posthocs
import seaborn as sns
import typer
from integral_parameters_script import plot_violins
from modules.constants import EXPERIMENTS, PATH
from modules.general import (flatten, initialize_logging, multiproc,
                             progress_bar)
from modules.mhp_clust import get_mhp_clusters_sizes_lifetimes
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


def calc_fractions_all(trj_slices: list) -> None:
    '''
    calculate mhp fractions for all systems and save them as csv
    '''
    if (PATH / 'notebooks' / 'mhpmaps' /
        'for_hists_fractions_'
            f'{trj_slices[0].b}-{trj_slices[0].e}-'
            f'{trj_slices[0].dt}.csv').is_file():
        logging.info('file exists, exiting...')
        return

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
        df.groupby(
            ['index', 'system', 'CHL amount, %'], as_index=False).agg(
            phob=('phob', 'mean'),
            phil=('phil', 'mean'),
            neutr=('neutr', 'mean'),
            phob_std=('phob', 'std'),
            neutr_std=('neutr', 'std'),
            phil_std=('phil', 'std'),
        ))

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
    # pylint: disable=too-many-arguments
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
                             show_label: bool, n_pos: int = 4):
    '''
    draw mhp area bars for single experiment
    '''
    x = np.arange(len(EXPERIMENTS[exp]))
    width = 0.1 - 0.03 * (n_pos - 3)

    positions = (
        (x - 6 * width, x - 2 * width, x + 2 * width),
        (x - 5 * width, x - 1 * width, x + 3 * width),
        (x - 4 * width, x, x + 4 * width),
        (x - 3 * width, x + 1 * width, x + 5 * width),
    ) if n_pos == 4 else (
        (x - 4 * width, x - width, x + 2 * width),
        (x - 3 * width, x, x + 3 * width),
        (x - 2 * width, x + width, x + 4 * width),
    ) if n_pos == 3 else None

    alphas = (1, 0.5, 0.3, 0.1)
    alphas = alphas[:n_pos]

    systems = (
        EXPERIMENTS[exp],
        [i + '_chol10' for i in EXPERIMENTS[exp]],
        [i + '_chol30' for i in EXPERIMENTS[exp]],
        [i + '_chol50' for i in EXPERIMENTS[exp]],
    )
    systems = systems[4 - n_pos:]

    if show_label:
        labels = ('0% CHL', '10% CHL', '30% CHL', '50% CHL')
    else:
        labels = (None, None, None, None)
    labels = labels[4 - n_pos:]

    for syst, pos, alpha, label in zip(systems, positions, alphas, labels):
        draw_mhp_area_bars(ax, df, syst, width, pos, alpha, label)

    x = np.arange(len(EXPERIMENTS[exp]))
    ax.set_title(exp)
    ax.xaxis.set_ticks(x)
    ax.set_xticklabels(EXPERIMENTS[exp])


def plot_mhp_ratio_all(trj_slices, chol: bool = False):
    '''
    plot mhp fractions ratios for all systems
    '''
    logging.info('plotting area hists...')
    df = pd.read_csv(
        PATH / 'notebooks' / 'mhpmaps' /
        'for_hists_fractions_stats_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-'
        f'{trj_slices[0].dt}.csv') if not chol else pd.read_csv(
            PATH / 'notebooks' / 'mhpmaps' /
            'for_hists_fractions_stats_chol_'
            f'{trj_slices[0].b}-{trj_slices[0].e}-'
            f'{trj_slices[0].dt}.csv')

    df[['phob', 'phil',	'neutr', 'phob_std', 'phil_std', 'neutr_std']] = (
        df[['phob', 'phil',	'neutr', 'phob_std', 'phil_std', 'neutr_std']]
        * 100)
    df.set_index('index', inplace=True)

    fig, axs = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    for ax, exp in zip(axs, EXPERIMENTS):
        n = 3 if chol else 4
        plot_mhp_area_single_exp(ax, df, exp, ax == axs[0], n)
        ticks, labels = ax.get_xticks(), ax.get_xticklabels()
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45,
                           ha='right', rotation_mode='anchor')
    axs[0].set_ylabel('% of area')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=n)

    if chol:
        fig.savefig(
            PATH / 'notebooks' / 'mhpmaps' / 'imgs' /
            f'mhp_hists_area_chol_dt{trj_slices[0].dt}.png',
            bbox_inches='tight', dpi=300)
    else:
        fig.savefig(
            PATH / 'notebooks' / 'mhpmaps' / 'imgs' /
            f'mhp_hists_area_dt{trj_slices[0].dt}.png',
            bbox_inches='tight', dpi=300)
    logging.info('done.')


def stats_by_mhp(df: pd.DataFrame,):
    # mode: str) -> pd.DataFrame:
    '''
    compare phob neutr phil columns in df
    using Kruskal-Wallis test followed by post-hoc Dunn test with Holm p-adjust

    mode can be: ci (confidence intervals), subsampling, stats (mean, std,
    25 and 75 quantiles, ci of mean and std)
    '''
    # TODO: add modes here
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

# TODO: add these 2 functions:
# def stats_by_chl_amount(df: pd.DataFrame, mode: str) -> pd.DataFrame:
#     '''
#     compare 0 10 30 50 CHL amount in df for phob, neutr, phil fractions
#     using Kruskal-Wallis test followed by post-hoc Dunn test with
# Holm p-adjust
#
#     mode can be: ci (confidence intervals), subsampling, stats (mean, std,
#     25 and 75 quantiles, ci of mean and std)
#     '''
#
#
# def stats_by_exp(df: pd.DataFrame, mode: str) -> pd.DataFrame:
#     '''
#     compare systs by experiment for 0 10 30 50 CHL and phob phil neutr in df
#     using Kruskal-Wallis test followed by post-hoc Dunn test with
# Holm p-adjust
#
#     mode can be: ci (confidence intervals), subsampling, stats (mean, std,
#     25 and 75 quantiles, ci of mean and std)
#     '''


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
    # logging.info('by CHL amount difference...')
    # df_by_mhp = stats_by_chl_amount(df)

    # compare syst1 syst2 syst3 for each of experiments
    # logging.info('by exp difference...')
    # df_by_mhp = stats_by_exp(df)


def get_detailed_mhp_data(trj_slices: list) -> None:
    '''
    calculate mhp fractions by molecule in all systems and save them as csv
    '''
    # pylint: disable = unsubscriptable-object
    # Universe.atoms is subscriptable

    if (PATH / 'notebooks' / 'mhpmaps' / 'info_mhp_atoms_'
            f'{trj_slices[0].b}-{trj_slices[0].e}-'
            f'{trj_slices[0].dt*10}.csv').is_file():
        logging.info('file exists, exiting...')
        return

    to_df = {'index': [],
             'system': [],
             'CHL amount, %': [],
             'timepoint': [],
             'mol_name': [],
             'mol_ind': [],
             'at_name': [],
             'at_ind': [],
             'mhp': []}

    with progress_bar as p:
        overall_task = p.add_task('system', total=len(trj_slices))
        for trj in trj_slices:
            prefix = PATH / trj.system.name / 'mhp_200.0-201.0-1'
            at_info = np.load(prefix / '1_pa.nmp')['data']
            mapp = np.load(prefix / '1_data.nmp')['data']

            logging.info('collecting data for %s...', trj.system.name)
            trj.generate_slice_with_gmx()

            u = mda.Universe(
                str(PATH / trj.system.name / 'md' / trj.system.tpr),
                str(PATH / trj.system.name / 'md' / trj.system.xtc),
                refresh_offsets=True)

            subtask = p.add_task('timestep', total=mapp.shape[0] / 10)
            for ts in range(0, mapp.shape[0], 10):
                map_ts = mapp[ts].flatten()
                at_ts = at_info[ts].flatten()
                to_df['index'].extend([trj.system.name for _ in map_ts])
                to_df['system'].extend([trj.system.name.split('_chol', 1)[0]
                                        for _ in map_ts])
                to_df['CHL amount, %'].extend(
                    [trj.system.name.split('_chol', 1)[1]
                     if '_chol' in trj.system.name else '0'
                     for _ in map_ts])
                to_df['timepoint'].extend([trj.b * 1000 + ts for _ in map_ts])
                to_df['mol_name'].extend([u.atoms[i].resname for i in at_ts])
                to_df['mol_ind'].extend([u.atoms[i].resid for i in at_ts])
                to_df['at_name'].extend([u.atoms[i].name for i in at_ts])
                to_df['at_ind'].extend(at_ts)
                to_df['mhp'].extend(map_ts)
                p.update(subtask, advance=1)
            p.remove_task(subtask)
            p.update(overall_task, advance=1)
        logging.info('saving results...')
        df = pd.DataFrame.from_dict(to_df)
        df.to_csv(
            PATH / 'notebooks' / 'mhpmaps' / 'info_mhp_atoms_'
            f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt*10}.csv')
        logging.info('done.')


def calc_fractions_chol(trj_slices: list) -> None:
    '''
    calculate mhp fractions of chl all systems and save them as csv
    '''
    logging.info('getting detailed mhp data...')
    get_detailed_mhp_data(trj_slices)

    logging.info('loading data...')
    df = pd.read_csv(PATH / 'notebooks' / 'mhpmaps' / 'info_mhp_atoms_'
                     f'{trj_slices[0].b}-{trj_slices[0].e}-'
                     f'{trj_slices[0].dt*10}.csv',
                     usecols=[1, 2, 3, 4, 5, 9])

    logging.info('calculating fractions...')
    df_chol = df[df['mol_name'] == 'CHL'].copy()

    df_chol['phob'] = df_chol['mhp'] >= 0.5
    df_chol['phil'] = df_chol['mhp'] <= -0.5
    df_chol['neutr'] = (-0.5 < df_chol['mhp']) & (df_chol['mhp'] < 0.5)

    for_hists_fractions_chol = df_chol.groupby(
        ['index', 'system', 'CHL amount, %', 'timepoint'], as_index=False
    )[['phob', 'phil', 'neutr']].agg(lambda x: np.sum(x) / len(x))

    for_hists_fractions_chol.to_csv(
        PATH / 'notebooks' / 'mhpmaps' /
        'for_hists_fractions_chol_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv',
        index=False)

    for_hists_fractions_stats_chol = (
        for_hists_fractions_chol.groupby(
            ['index', 'system', 'CHL amount, %'], as_index=False).agg(
            phob=('phob', 'mean'),
            phil=('phil', 'mean'),
            neutr=('neutr', 'mean'),
            phob_std=('phob', 'std'),
            neutr_std=('neutr', 'std'),
            phil_std=('phil', 'std'),
        ))

    for_hists_fractions_stats_chol.to_csv(
        PATH / 'notebooks' / 'mhpmaps' /
        'for_hists_fractions_stats_chol_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv',
        index=False)

    logging.info('done.')


def calc_chol_surf_fractions(trj_slices: list) -> None:
    '''
    calculate fraction of cholesterol on surface for all systems
    '''
    logging.info('getting detailed mhp data...')
    get_detailed_mhp_data(trj_slices)

    logging.info('loading data...')
    df = pd.read_csv(PATH / 'notebooks' / 'mhpmaps' / 'info_mhp_atoms_'
                     f'{trj_slices[0].b}-{trj_slices[0].e}-'
                     f'{trj_slices[0].dt*10}.csv',
                     usecols=[1, 2, 3, 4, 5, 9])

    logging.info('calculating chl surface fractions...')

    df2 = df[df['CHL amount, %'] != 0].copy()
    df2['chl_fraction'] = df2['mol_name'] == 'CHL'
    chol_fractions = df2.groupby(
        ['index', 'system', 'CHL amount, %', 'timepoint'], as_index=False
    )[['chl_fraction']].agg(lambda x: np.sum(x) / len(x))

    chol_fractions_stats = chol_fractions.groupby(
        ['index', 'system', 'CHL amount, %'], as_index=False).agg(
        chl_fraction=('chl_fraction', 'mean'),
        chl_fraction_std=('chl_fraction', 'std'))

    logging.info('saving_results...')

    chol_fractions.to_csv(
        PATH / 'notebooks' / 'mhpmaps' /
        'chol_from_all_fractions_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv',
        index=False)

    chol_fractions_stats.to_csv(
        PATH / 'notebooks' / 'mhpmaps' /
        'chol_from_all_fractions_stats_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv',
        index=False)

    logging.info('done.')


def plot_calc_chol_surf_fractions(trj_slices: list) -> None:
    '''
    plot fractions of chol on surface for all systems
    '''
    # pylint: disable=too-many-locals
    logging.info('plotting chol fractions of surface...')

    # barplots:
    df = pd.read_csv(PATH / 'notebooks' / 'mhpmaps' /
                     'chol_from_all_fractions_stats_'
                     f'{trj_slices[0].b}-{trj_slices[0].e}-'
                     f'{trj_slices[0].dt}.csv')

    df[['chl_fraction', 'chl_fraction_std']] = (
        df[['chl_fraction', 'chl_fraction_std']] * 100)

    df.set_index('index', inplace=True)

    fig, axs = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    for ax, exp in zip(axs, EXPERIMENTS):
        x = np.arange(len(EXPERIMENTS[exp]))
        width = 0.25
        positions = (x - width, x, x + width)

        palette = sns.color_palette('RdYlGn_r', 3)
        systems = (
            [i + '_chol10' for i in EXPERIMENTS[exp]],
            [i + '_chol30' for i in EXPERIMENTS[exp]],
            [i + '_chol50' for i in EXPERIMENTS[exp]],
        )
        if ax == axs[0]:
            labels = ('10% CHL', '30% CHL', '50% CHL')
        else:
            labels = (None, None, None)
        for systs, pos, color, label in zip(
                systems, positions, palette, labels):
            ax.bar(pos, df.loc[systs, :]['chl_fraction'], width,
                   yerr=df.loc[systs, :]['chl_fraction_std'], capsize=5,
                   label=label, color=color)

            # single black edges independent on alpha
            ax.bar(pos, df.loc[systs, :]['chl_fraction'], width,
                   yerr=df.loc[systs, :]['chl_fraction_std'],
                   capsize=5, ec='k',
                   fill=False, lw=2)
        x = np.arange(len(EXPERIMENTS[exp]))
        ax.set_title(exp)
        ax.xaxis.set_ticks(x)
        ax.set_xticklabels(EXPERIMENTS[exp])
        ticks, labels = ax.get_xticks(), ax.get_xticklabels()
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45,
                           ha='right', rotation_mode='anchor')

    axs[0].set_ylabel('% of area')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    fig.savefig(
        PATH / 'notebooks' / 'mhpmaps' / 'imgs' /
        f'chol_surf_fractions_dt{trj_slices[0].dt}.png',
        bbox_inches='tight', dpi=300)

    # violins
    plot_violins(PATH / 'notebooks' / 'mhpmaps' /
                 'chol_from_all_fractions_'
                 f'{trj_slices[0].b}-{trj_slices[0].e}-'
                 f'{trj_slices[0].dt}.csv', 'chl_fraction', '% of area')

    logging.info('done.')


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
    if calculate:
        calc_fractions_chol(trj_slices)
    if plot:
        plot_mhp_ratio_all(trj_slices, chol=True)


def chol_all(trj_slices: list,
             calculate: bool, plot: bool, stat: bool) -> None:
    '''
    calculate and plot ratio of surface areas
    consisting of CHL atoms
    also calculate statistics
    '''
    if calculate:
        calc_chol_surf_fractions(trj_slices)
    if plot:
        plot_calc_chol_surf_fractions(trj_slices)


def calculate_mhp_clusters(trj_slices: list) -> None:
    '''
    calculate sizes and lifetimes of hydrohobic and hydrophilic
    MHP clusters for each system
    '''
    for option in ['hydrophobic', 'hydrophilic']:
        logging.info('obtaining %s mhp clusters...', option)
        with progress_bar as p:
            task_id = p.add_task('mhp clusterization', total=len(trj_slices))
            for trj in trj_slices:
                nmp_file = (PATH / trj.system.name /
                            f'mhp_{trj.b}-{trj.e}-{trj.dt}' / '1_data.nmp')

                trj.generate_slice_with_gmx()

                u = mda.Universe(
                    str(PATH / trj.system.name / 'md' / trj.system.tpr),
                    str(PATH / trj.system.name / 'md' / trj.system.xtc),
                    refresh_offsets=True)

                box_side = u.dimensions[0]

                sizes, lifetimes = get_mhp_clusters_sizes_lifetimes(
                    nmp_file, box_side, trj.dt, option)

                np.savetxt(PATH / 'notebooks' / 'mhpmaps' / 'clust' /
                           f'{trj.system.name}_{trj.b}-{trj.e}-'
                           f'{trj.dt}_{option}.txt',
                           sizes)

                np.savetxt(PATH / 'notebooks' / 'mhpmaps' / 'clust' /
                           f'{trj.system.name}_{trj.b}-{trj.e}-'
                           f'{trj.dt}_{option}_lt.txt',
                           lifetimes + 1)

                p.update(task_id, advance=1)

    logging.info('done.')


def plot_mhp_clusters(trj_slices: list, option: str,
                      lifetimes: bool, bigger_lifetimes: bool) -> None:
    '''
    plot sizes or lifetimes of hydrohobic or hydrophilic
    MHP clusters for each system

    option - hydrophobic, hydrophilic
    if lifetimes plot lifetimes
    if bigger lifetimes plot only lifetimes > 10 ps
    '''
    # pylint: disable=too-many-locals

    logging.info('plotting %s mhp clusters...', option)
    logging.info('lifetimes: %s\nbigger_lifetimes: %s',
                 lifetimes, bigger_lifetimes)

    with progress_bar as p:
        for exp, systs in p.track(EXPERIMENTS.items(),
                                  description='plotting mhp clusters'):
            x = 'cluster lifetime, ps' if lifetimes else 'cluster size, Å'
            lt = '_lt' if lifetimes else ''

            systs_exp = flatten([
                [syst] + [syst + f'_chol{i}' for i in (10, 30, 50)]
                for syst in systs])
            trjs = [[i for i in trj_slices if i.system.name == s][0]
                    for s in systs_exp]

            df = pd.DataFrame.from_dict({
                'index': systs_exp,
                'system': [i.split('_chol', 1)[0] for i in systs_exp],
                'CHL amount, %': ['0' if len(i.split('_chol', 1)) == 1
                                  else i.split('_chol', 1)[1]
                                  for i in systs_exp],
                x: [np.loadtxt(
                    PATH / 'notebooks' / 'mhpmaps' / 'clust' /
                    f'{trj.system.name}_{trj.b}-{trj.e}-'
                    f'{trj.dt}_{option}{lt}.txt') for trj in trjs],
            })

            df = df.explode(x, ignore_index=True)
            fig, axs = plt.subplots(1, 3, figsize=(20, 7),
                                    sharey=True, sharex=True)
            for syst, ax in zip(df['system'].unique(), axs):
                data = df[df['system'] == syst]
                if lifetimes:
                    if bigger_lifetimes:
                        data = data[data[x] > 10]
                        binwidth = 5
                    else:
                        data = data[data[x] <= 10]
                        binwidth = 1

                    sns.histplot(data=data, x=x, alpha=.2, hue='CHL amount, %',
                                 stat='density', fill=True, binwidth=binwidth,
                                 legend=ax == axs[-1], common_norm=False,
                                 palette='RdYlGn_r', ax=ax)

                    sns.histplot(data=data, x=x, hue='CHL amount, %',
                                 stat='density', fill=False, binwidth=binwidth,
                                 legend=False, common_norm=False,
                                 palette='RdYlGn_r', ax=ax, lw=2)
                else:
                    sns.histplot(data=data, x=x, alpha=.2, lw=0,
                                 hue='CHL amount, %',
                                 palette='RdYlGn_r', stat='density', ax=ax,
                                 binwidth=.15, log_scale=True,
                                 common_norm=False,
                                 legend=ax == axs[-1])
                    sns.histplot(data=data, x=x, lw=2, fill=False, alpha=.5,
                                 legend=False,
                                 element='step', hue='CHL amount, %',
                                 palette='RdYlGn_r', stat='density', ax=ax,
                                 binwidth=.15, log_scale=True,
                                 common_norm=False,)
                    sns.kdeplot(data=data, x=x, lw=5,
                                hue='CHL amount, %',
                                palette='RdYlGn_r', ax=ax, common_norm=False,
                                log_scale=True,
                                legend=False)

                    ax.set_xlim(1)
                ax.set_title(syst)
            fig.suptitle(exp + ' ' + option)
            fname = ('bigger_lifetimes_' if bigger_lifetimes else
                     'lifetimes_' if lifetimes else '')
            fig.savefig(
                PATH / 'notebooks' / 'mhpmaps' / 'imgs' /
                f'mhp_clusters_{"_".join(exp.split())}_'
                f'{option}_{fname}dt{trj_slices[0].dt}.png',
                bbox_inches='tight', dpi=300)
    logging.info('done.')


def plot_mhp_hists_single_exp(progress: dict, task_id: int,
                              df: pd.DataFrame,
                              trj_slices: list, exp: str) -> None:
    '''
    plot overall distribution of mhp values per system for single experiment
    from EXPERIMENTS

    progress and tak_id are needed for multiproc
    df -- dataframe with all mhp values
    trj_slices -- list of all trajectories
    exp -- key from EXPERIMENTS dict
    '''

    len_of_task = len(EXPERIMENTS[exp])
    fig, axs = plt.subplots(1, 3, figsize=(20, 7), sharex=True, sharey=True)
    for c, (ax, syst) in enumerate(zip(axs, EXPERIMENTS[exp])):
        data = df[df['system'] == syst]
        sns.kdeplot(data=data, x='mhp', hue='CHL amount, %', ax=ax,
                    legend=ax == axs[-1], palette='RdYlGn_r',
                    common_norm=False, fill=True, alpha=.2)
        ax.axvline(-0.5, ls=':', c='k')
        ax.axvline(0.5, ls=':', c='k')
        ax.set_title(syst)
        progress[task_id] = progress[task_id] = {'progress': c + 1,
                                                 'total': len_of_task}

    fig.suptitle(exp)
    fig.savefig(
        PATH / 'notebooks' / 'mhpmaps' / 'imgs' /
        f'{"_".join(exp.split())}_mhp_values_kde_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-'
        f'{trj_slices[0].dt*10}.png',
        bbox_inches='tight', dpi=300)


@app.command()
def get(ctx: typer.Context,
        force: bool = typer.Option(
        False, help='override already calculated mhp data'),
        ):
    '''
    run [bold]impulse prj.json[/] on all systems to obtain mhp info
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


@ app.command()
def area_fractions(
    ctx: typer.Context,
    fraction_types: List[str] = typer.Argument(
        ..., help='tasks to run [dim](pick from '
        '[bold]mhp_all, mhp_chol, chol_all, mhp_time[/])[/]'),
    calculate: bool = typer.Option(
        False, help='calculate and dump fractions from scratch'),
    plot: bool = typer.Option(
        False, help='plot fractions'),
    stat: bool = typer.Option(
        False, help='calculate statistics'),
):
    '''
    plot hydrophilic/neutral/hydrophobic area fractions ratios for bilayers
    '''
    sns.set(style='ticks', context='talk', palette='muted')

    xtc, tpr, b, e, dt, _, verbose, _ = ctx.obj
    initialize_logging('mhp_fractions.log', verbose)
    systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in flatten(EXPERIMENTS.values())])
    systems = list(dict.fromkeys(systems))
    trajectory_slices = [TrajectorySlice(System(PATH, s, xtc, tpr), b, e, dt)
                         for s in systems]

    if 'mhp_all' in fraction_types:
        mhp_all(trajectory_slices, calculate, plot, stat)

    if 'mhp_chol' in fraction_types:
        mhp_chol(trajectory_slices, calculate, plot, stat)

    if 'chol_all' in fraction_types:
        chol_all(trajectory_slices, calculate, plot, stat)


@ app.command()
def plot_mhp_hists(ctx: typer.Context):
    '''
    plot overall distribution of mhp values per system
    '''
    sns.set(style='ticks', context='talk', palette='muted')

    xtc, tpr, b, e, dt, _, verbose, _ = ctx.obj
    initialize_logging('mhp_fractions.log', verbose)
    systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in flatten(EXPERIMENTS.values())])
    systems = list(dict.fromkeys(systems))
    trj_slices = [TrajectorySlice(System(PATH, s, xtc, tpr), b, e, dt)
                  for s in systems]

    logging.info('loading data...')
    df = pd.read_csv(PATH / 'notebooks' / 'mhpmaps' / 'info_mhp_atoms_'
                     f'{trj_slices[0].b}-{trj_slices[0].e}-'
                     f'{trj_slices[0].dt*10}.csv',
                     usecols=['system', 'CHL amount, %', 'mhp'])

    logging.info('drawing plots...')
    multiproc(plot_mhp_hists_single_exp,
              [df] * len(EXPERIMENTS),
              [tuple(trj_slices)] * len(EXPERIMENTS),
              EXPERIMENTS,
              descr='plotting mhp dists',
              n_workers=len(EXPERIMENTS),
              show_progress='multiple')
    logging.info('done.')


@ app.command()
def clust(ctx: typer.Context,
          calculate: bool = typer.Option(
              False, help='calculate and dump clusters from scratch'),
          plot: bool = typer.Option(
              False, help='plot cluster sizes and lifetimes'),
          stat: bool = typer.Option(
              False, help='calculate statistics'),
          ):
    '''
    calculate and plot clusterization of mhp data for hydrophobic/hydrophilic
    regions for bilayers
    '''
    sns.set(style='ticks', context='talk', palette='muted')

    xtc, tpr, b, e, dt, _, verbose, _ = ctx.obj
    initialize_logging('mhp_clust.log', verbose)
    systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in flatten(EXPERIMENTS.values())])
    systems = list(dict.fromkeys(systems))
    trajectory_slices = [TrajectorySlice(System(PATH, s, xtc, tpr), b, e, dt)
                         for s in systems]

    if calculate:
        calculate_mhp_clusters(trajectory_slices)

    if plot:
        for option in ['hydrophobic', 'hydrophilic']:
            for lifetimes in [True, False]:
                if lifetimes:
                    for bigger_lifetimes in [True, False]:
                        plot_mhp_clusters(trajectory_slices, option,
                                          lifetimes, bigger_lifetimes)
                else:
                    plot_mhp_clusters(trajectory_slices, option,
                                      lifetimes, False)


@ app.callback()
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
    # pylint: disable = too-many-arguments
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
#     200.0, 201.0, 1) for s in systems]
# trj_slices = trajectory_slices
