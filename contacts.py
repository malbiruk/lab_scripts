'''
set of utilities to obtain and plot contacts data
'''
# pylint: disable = too-many-arguments

import logging
import os
import subprocess
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from modules.constants import EXPERIMENTS, PATH
from modules.general import (duration, flatten, get_keys_by_value,
                             initialize_logging, multiproc, progress_bar)
from modules.tg_bot import run_or_send_error
from modules.traj import System, TrajectorySlice

app = typer.Typer(rich_markup_mode='rich', add_completion=False)


def calculate_contacts(trj: TrajectorySlice, grp1: str, grp2: str,
                       contact_type: str,
                       per_atom: int = 1) -> bool:
    '''
    wrapper for impulse to calculate specified contacts

    trj -- trajectory slice for which calculate contacts
    grp1, grp2 -- names of groups between which calculate contacts
    contact_type -- "hb" for hbonds, "dc" for contacts by distance
    per_atom -- if 1, calculate per atom, if 0 -- per molecule
    '''
    (Path(trj.system.dir) / 'contacts').mkdir(parents=True, exist_ok=True)
    os.chdir(Path(trj.system.dir) / 'contacts')

    grp2_name = ''
    if grp2 is not None:
        grp2_name = grp2 if '/' not in grp2 else grp2.replace('/', '')
        grp2_name = grp2 if '/' not in grp2 else grp2_name.replace('|', '')
        grp2_name = grp2_name + '_'

    if (Path(trj.system.dir) / 'contacts' /
        f'{grp1}_{grp2_name + contact_type}_{contact_type}_rcnames.csv'
        ).is_file():
        return True

    args = f'''
TOP={trj.system.dir}/md/{trj.system.tpr}
TRJ={trj.system.dir}/md/{trj.system.xtc}
BEG={int(trj.b*1000)}
END={int(trj.e*1000)}
DT={trj.dt}
SUBSET="///"
GRP="{grp1+'///' if grp2 is None else ''}"
GRP1="{grp1+'///' if grp2 is not None else ''}"
GRP2="{grp2+'///' if grp2 is not None else ''}"
CLIST={contact_type}
PERATOM={per_atom}
RCHIST=1
FMT=csv
OUT="{grp1}_{grp2_name + contact_type}"'''

    with open('args', 'w', encoding='utf-8') as f:
        f.write(args)

    msg = f'couldn\'t obtain contacts for `{trj.system.name}`'
    impulse = '/nfs/belka2/soft/impulse/dev/inst/runtask.py'
    cont_stat = '/home/krylov/Progs/IBX/AMMP/test/postpro/contacts/cont_stat.js'
    cmd = f'{impulse} -f args -t {cont_stat}'

    if run_or_send_error(cmd, msg,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.STDOUT):
        logging.info('sucessfully calculated contacts for %s', trj.system.name)
        return True

    logging.error('couldn\'t calculate contacts for %s', trj.system.name)
    return False


def import_ft_table(csv: Path, atoms: bool = True) -> pd.DataFrame:
    '''
    parse _ft.csv files for correct representation in pandas
    '''
    df = pd.read_csv(csv, sep=r'\s+|,', engine='python', header=None,
                     skiprows=1)
    if atoms:
        df.drop(columns=[0, 7, 2, 4, 8, 10, 12], inplace=True)
        df.rename(columns={
            1: 'dmi', 3: 'dmn', 5: 'dan', 6: 'dai', 9: 'ami', 11: 'amn',
            13: 'aan', 14: 'aai', 15: 'dt_fr', 16: 'dt_tm'}, inplace=True)
    else:
        df.drop(columns=[0, 1, 4, 5, 6, 7, 8, 11, 12], inplace=True)
        df.rename(columns={
            2: 'dmi', 3: 'dmn', 9: 'ami', 10: 'amn', 13: 'dt_fr', 14: 'dt_tm'
        }, inplace=True)
    return df


def int_to_selection(chl_lip: pd.DataFrame, lip_sol: pd.DataFrame) -> dict:
    '''
    returns dict where keys are desired interactions and
    values -- selections in dataframes obtained using import_ft_table()
    method

    supported interactions: 'CHL-CHL', 'CHL-PL', 'CHL-SOL', 'PL-SOL', 'PL-PL'
    '''
    return {
        'CHL-CHL': chl_lip[
            (chl_lip['dmn'] == 'CHL')
            & (chl_lip['amn'] == 'CHL')],
        'CHL-PL': chl_lip[
            ((chl_lip['dmn'] == 'CHL')
             & (chl_lip['amn'] != 'CHL'))
            | ((chl_lip['dmn'] != 'CHL')
               & (chl_lip['amn'] == 'CHL'))],
        'CHL-SOL': lip_sol[
            ((lip_sol['dmn'] == 'SOL')
             & (lip_sol['amn'] == 'CHL'))
            | ((lip_sol['dmn'] == 'CHL')
               & (lip_sol['amn'] == 'SOL'))],
        'PL-SOL': lip_sol[
            ((lip_sol['dmn'] == 'SOL')
             & (lip_sol['amn'] != 'CHL'))
            | ((lip_sol['dmn'] != 'CHL')
               & (lip_sol['amn'] == 'SOL'))],
        'PL-PL': chl_lip[
            (chl_lip['dmn'] != 'CHL')
            & (chl_lip['amn'] != 'CHL')]}


def create_contacts_mean_std_df(trj_slices: list) -> None:
    '''
    creates df with means and stds of hb probabilities
    '''
    filename = (PATH / 'notebooks' / 'contacts' / 'contacts_mean_std_df_'
                f'{trj_slices[0].b}-{trj_slices[0].e}-'
                f'{trj_slices[0].dt}.csv')
    if filename.is_file():
        return

    to_df = []

    with progress_bar as p:
        for trj in p.track(trj_slices, description='mean std hb lt'):
            try:
                chl_lip = import_ft_table(
                    PATH / trj.system.name / 'contacts' / 'lip_hb_hb_ft.csv')
            except pd.errors.EmptyDataError:
                chl_lip = pd.DataFrame([
                    {'dmn': 'CHL', 'amn': 'LIP', 'dt_fr': 0}
                ])
            lip_sol = import_ft_table(
                PATH / trj.system.name / 'contacts' / 'lip_SOL_hb_hb_ft.csv')

            int_to_sel = int_to_selection(chl_lip, lip_sol)

            row = {
                'index': trj.system.name,
                'system': trj.system.name.split('_chol', 1)[0],
                'CHL amount, %': (
                    '0' if len(trj.system.name.split('_chol', 1)) == 1
                    else trj.system.name.split('_chol', 1)[1])}

            for key in ['CHL-CHL', 'CHL-PL', 'CHL-SOL', 'PL-SOL', 'PL-PL']:
                values = int_to_sel[key]['dt_fr']
                row[key] = values.mean()
                row[key + ' std'] = values.std()

            to_df.append(row)
    df = pd.DataFrame(to_df)
    df.to_csv(filename, index=False)


def draw_hb_prob_bars(ax: plt.axis, df: pd.DataFrame, systs: list,
                      width: float, positions: tuple,
                      alpha: float, label: str):
    '''
    create single bar plot for df with columns
    "CHL-CHL", "CHL-CHL std", "CHL-PL"...
    '''
    # pylint: disable=too-many-arguments
    if label is not None:
        labels = [i + ', ' + label for i in df.columns[2::2]]
    else:
        labels = [None for _ in df.columns[2::2]]

    for c, col in enumerate(df.columns[2::2]):
        ax.bar(positions[c], df.loc[systs, :][col], width,
               yerr=df.loc[systs, :][f'{col} std'],
               label=labels[c], color=f'C{c}', alpha=alpha)
        # single black edges independent on alpha
        ax.bar(positions[c], df.loc[systs, :][col], width,
               yerr=df.loc[systs, :][f'{col} std'], ec='k',
               fill=False, lw=2)


def plot_hb_prob_single_exp(ax: plt.axis, df: pd.DataFrame, exp: str,
                            show_label: bool):
    '''
    draw mhp area bars for single experiment
    '''
    x = np.arange(len(EXPERIMENTS[exp]))
    width = 0.05

    positions = (
        (x - 10 * width, x - 6 * width,
         x - 2 * width, x + 2 * width, x + 6 * width),
        (x - 9 * width, x - 5 * width,
         x - 1 * width, x + 3 * width, x + 7 * width),
        (x - 8 * width, x - 4 * width,
         x, x + 4 * width, x + 8 * width),
        (x - 7 * width, x - 3 * width,
         x + 1 * width, x + 5 * width, x + 9 * width),
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
        draw_hb_prob_bars(ax, df, syst, width, pos, alpha, label)

    x = np.arange(len(EXPERIMENTS[exp]))
    ax.set_title(exp)
    ax.xaxis.set_ticks(x)
    ax.set_xticklabels(EXPERIMENTS[exp])


def plot_hb_probabilities(trj_slices):
    '''
    plot hb probabilities for all systems
    '''
    logging.info('plotting hb probabilities hists...')
    df = pd.read_csv(
        PATH / 'notebooks' / 'contacts' /
        f'contacts_mean_std_df_{trj_slices[0].b}-{trj_slices[0].e}'
        f'-{trj_slices[0].dt}.csv')

    df.fillna(0, inplace=True)
    df[df.columns[3:]] = df[df.columns[3:]] * 100
    df.set_index('index', inplace=True)

    for exp in EXPERIMENTS:
        fig, ax = plt.subplots(1, 1, figsize=(20, 7), sharey=True)
        plot_hb_prob_single_exp(ax, df, exp, True)
        ax.set_ylim(0)
        ticks, labels = ax.get_xticks(), ax.get_xticklabels()
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45,
                           ha='right', rotation_mode='anchor')
        ax.set_ylabel('probability of hydrogen bond formantion')
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)

        fig.savefig(
            PATH / 'notebooks' / 'contacts' / 'imgs' /
            f'hb_probabilities_{"_".join(exp.split())}'
            f'_dt{trj_slices[0].dt}.png',
            bbox_inches='tight', dpi=300)
    logging.info('done.')


def obtain_hb_lt_distribution(
        progress, task_id,
        trj_slices: list, interaction: str) -> None:
    '''
    create and save dataframe with lifetimes distribution of
    hydrogen bonds of interaction specified
    (should be in ['CHL-CHL', 'CHL-PL', 'CHL-SOL', 'PL-SOL', 'PL-PL'])
    '''
    filename = (PATH / 'notebooks' / 'contacts' /
                f'{interaction}_hb_lt_distr_'
                f'{trj_slices[0].b}-{trj_slices[0].e}-'
                f'{trj_slices[0].dt}.csv')

    if filename.is_file():
        return

    to_df = {
        'index': [],
        'system': [],
        'CHL amount, %': [],
        interaction: [],
        interaction + ' prob': []
    }

    len_of_task = len(trj_slices)

    for c, trj in enumerate(trj_slices):
        try:
            chl_lip = import_ft_table(
                PATH / trj.system.name / 'contacts' / 'lip_hb_hb_ft.csv')
        except pd.errors.EmptyDataError:
            chl_lip = pd.DataFrame([
                {'dmn': 'LIP', 'amn': 'LIP', 'dt_fr': 0, 'dt_tm': 0}
            ])
        lip_sol = import_ft_table(
            PATH / trj.system.name / 'contacts' / 'lip_SOL_hb_hb_ft.csv')
        int_to_sel = int_to_selection(chl_lip, lip_sol)

        dt_tm_values = int_to_sel[interaction]['dt_tm'].tolist()
        dt_fr_values = int_to_sel[interaction]['dt_fr'].tolist()

        to_df['index'].extend([trj.system.name] * len(dt_tm_values))
        to_df['system'].extend([trj.system.name.split('_chol', 1)[0]]
                               * len(dt_tm_values))
        to_df['CHL amount, %'].extend([
            '0' if len(trj.system.name.split('_chol', 1)) == 1
            else trj.system.name.split('_chol', 1)[1]]
            * len(dt_tm_values))
        to_df[interaction].extend(dt_tm_values)
        to_df[interaction + ' prob'].extend(dt_fr_values)

        progress[task_id] = {'progress': c + 1, 'total': len_of_task}

    df = pd.DataFrame.from_dict(to_df)
    df.to_csv(filename, index=False)


def plot_hb_lt_distr_single_interaction(progress: dict, task_id: int,
                                        trj_slices: list, interaction: str):
    '''
    plot distributions of hydrogen bonds lifetimes for each experiment
    from EXPERIMENTS for specified interaction
    (should be in ['CHL-CHL', 'CHL-PL', 'CHL-SOL', 'PL-SOL', 'PL-PL'])
    '''
    df = pd.read_csv(PATH / 'notebooks' / 'contacts' /
                     f'{interaction}_hb_lt_distr_'
                     f'{trj_slices[0].b}-{trj_slices[0].e}-'
                     f'{trj_slices[0].dt}.csv')
    if 'CHL' in interaction:
        df = df[df['CHL amount, %'] != 0]
    df.rename(columns={interaction: 'lifetime, ps'}, inplace=True)
    df = df[df['lifetime, ps'] != 0]

    len_of_task = len(EXPERIMENTS)
    for c, (exp, systs) in enumerate(EXPERIMENTS.items()):
        fname = (PATH / 'notebooks' / 'contacts' / 'imgs' /
                 f'{interaction}_{exp}_hb_lt_distr_'
                 f'{trj_slices[0].b}-{trj_slices[0].e}-'
                 f'{trj_slices[0].dt}.png')

        if fname.is_file():
            continue

        fig, axs = plt.subplots(1, 3, figsize=(20, 7),
                                sharey=True, sharex=True)
        for syst, ax in zip(systs, axs):
            data = df[df['system'] == syst]
            sns.histplot(data=data, x='lifetime, ps', alpha=.2,
                         hue='CHL amount, %',
                         palette='RdYlGn_r', stat='density', ax=ax,
                         binwidth=.15, log_scale=True,
                         legend=ax == axs[-1], common_norm=False,
                         kde=True, line_kws={'lw': 5})
            sns.kdeplot(data=data, x='lifetime, ps', lw=5,
                        hue='CHL amount, %',
                        palette='RdYlGn_r', ax=ax, common_norm=False,
                        log_scale=True, legend=False)
            ax.set_title(syst)
        fig.suptitle(exp)
        fig.savefig(fname, bbox_inches='tight', dpi=300)
        progress[task_id] = {'progress': c + 1, 'total': len_of_task}


def plot_hb_lt_distributions(
        trj_slices: list, interactions: tuple =
        ('CHL-CHL', 'CHL-PL', 'CHL-SOL', 'PL-SOL', 'PL-PL')) -> None:
    '''
    plot distributions of hydrogen bonds lifetimes for each experiment
    from EXPERIMENTS for each of specified interactions
    '''
    logging.info('plotting hb lifetimes distributions...')
    multiproc(obtain_hb_lt_distribution,
              [tuple(trj_slices) for _ in interactions],
              interactions,
              descr='getting hb lt dists',
              n_workers=len(interactions),
              show_progress='multiple')

    multiproc(plot_hb_lt_distr_single_interaction,
              [tuple(trj_slices) for _ in interactions],
              interactions,
              descr='plotting hb lt dists',
              n_workers=len(interactions),
              show_progress='multiple')
    logging.info('done.')


def retrieve_chl_lip_over_lip_lip_ratios(trj_slices: list) -> None:
    '''
    calculate ratios of n of CHL-PL contacts over all LIP-LIP contacts
    in each frame
    '''
    fname = (PATH / 'notebooks' / 'contacts' /
             f'chl_lip_over_lip_lip_ratios_'
             f'{trj_slices[0].b}-{trj_slices[0].e}-'
             f'{trj_slices[0].dt}.csv')

    if fname.is_file():
        return

    dfs = []
    for trj in trj_slices:
        grp2 = (trj.system.pl_selector()[1:-4]
                if 'dopc_dops50' not in trj.system.name
                else trj.system.pl_selector(0)[1:-4]
                + trj.system.pl_selector(1)[1:-6])

        chl_lip = pd.read_csv(PATH / trj.system.name / 'contacts' /
                              f'CHOL_{grp2}_dc_dc_ncr.csv',
                              header=None, skiprows=1,
                              names=['timepoint', 'CHL-PL'])
        lip_lip = pd.read_csv(PATH / trj.system.name / 'contacts' /
                              'lip_dc_dc_ncr.csv',
                              header=None, skiprows=1,
                              names=['timepoint', 'LIP-LIP'])

        df = chl_lip.merge(lip_lip, on=['timepoint'])
        c = {'index': trj.system.name,
             'system': trj.system.name.split('_chol', 1)[0],
             'CHL amount, %': trj.system.name.split('_chol', 1)[1]}
        df = df.assign(**c)
        columns_to_move = ['index', 'system', 'CHL amount, %']
        new_columns = columns_to_move + [col for col in df.columns
                                         if col not in columns_to_move]
        df = df[new_columns]
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    df_all['CHL-PL / LIP-LIP, %'] = df_all['CHL-PL'] / df_all['LIP-LIP'] * 100
    df_all.to_csv(fname, index=False)


def retrieve_chl_chl_over_lip_lip_ratios(trj_slices: list) -> None:
    '''
    calculate ratios of ft of CHL-CHL contacts over all LIP-LIP contacts
    ci obtained using bootstrapping
    '''
    fname = (PATH / 'notebooks' / 'contacts' /
             f'chl_chl_over_lip_lip_ratios_'
             f'{trj_slices[0].b}-{trj_slices[0].e}-'
             f'{trj_slices[0].dt}.csv')

    if fname.is_file():
        return

    logging.info('retrieving ratios...')
    to_df = []
    with progress_bar as p:
        for trj in p.track(trj_slices, description='trj'):
            df = import_ft_table(PATH / trj.system.name / 'contacts' /
                                 'lip_dc_dc_ft.csv', False)

            # bootstrapping
            bootstrapped_values = []
            for _ in range(1000):
                sampled = df.sample(frac=1, replace=True)
                bootstrapped_values.append(
                    sampled[(sampled['dmn'] == 'CHL')
                            & (sampled['amn'] == 'CHL')]['dt_tm'].sum()
                    / sampled['dt_tm'].sum() * 100)

            bootstrapped_values = np.array(bootstrapped_values)
            std = np.std(bootstrapped_values)
            confidence_interval = 1.96 * (std
                                          / np.sqrt(len(bootstrapped_values)))

            row = {'index': trj.system.name,
                   'system': trj.system.name.split('_chol', 1)[0],
                   'CHL amount, %': trj.system.name.split('_chol', 1)[1],
                   'CHL-CHL / LIP-LIP, %': (
                       df[(df['dmn'] == 'CHL')
                          & (df['amn'] == 'CHL')]['dt_tm'].sum()
                       / df['dt_tm'].sum() * 100),
                   'ci': confidence_interval,
                   'std': std
                   }
            to_df.append(row)

    df_all = pd.DataFrame(to_df)
    df_all.to_csv(fname, index=False)
    logging.info('done.')


def plot_chl_chl_over_lip_lip_ratios(trj_slices: list) -> None:
    '''
    plot ratios of CHL-CHL contacts by distance and all LIP-LIP contacts
    for all systems
    '''
    logging.info('plotting ratios...')

    df = pd.read_csv(PATH / 'notebooks' / 'contacts' /
                     f'chl_chl_over_lip_lip_ratios_'
                     f'{trj_slices[0].b}-{trj_slices[0].e}-'
                     f'{trj_slices[0].dt}.csv')
    df.set_index('index', inplace=True)

    fig, axs = plt.subplots(1, 3, figsize=(20, 7),
                            sharey=True)

    width = 0.25
    palette = sns.color_palette('RdYlGn_r', 3)

    for ax, exp in zip(axs, EXPERIMENTS):
        x = np.arange(len(EXPERIMENTS[exp]))
        positions = (x - width, x, x + width)
        for c, chl_amount in enumerate([10, 30, 50]):
            systs = [i + f'_chol{chl_amount}' for i in EXPERIMENTS[exp]]
            label = chl_amount if ax == axs[0] else None
            ax.bar(positions[c], df.loc[systs, 'CHL-CHL / LIP-LIP, %'],
                   yerr=df.loc[systs, 'std'], width=width, ec='k',
                   color=palette[c], capsize=5,
                   error_kw={'elinewidth': 2}, label=label)
        ax.set_title(exp)
        ax.xaxis.set_ticks(x)
        ax.set_xticklabels(EXPERIMENTS[exp])
        ax.set_ylim(0)
        if ax == axs[0]:
            ax.set_ylabel('CHL-CHL / LIP-LIP, %')
    fig.legend(loc='upper center', title='CHL amount, %',
               bbox_to_anchor=(0.5, 0), ncol=3)
    fig.savefig(PATH / 'notebooks' / 'contacts' / 'imgs' /
                f'chl_chl_over_lip_lip_ratios_'
                f'{trj_slices[0].b}-'
                f'{trj_slices[0].e}-'
                f'{trj_slices[0].dt}.png',
                bbox_inches='tight')
    logging.info('done.')


def retrieve_pl_pl_over_lip_lip_ratios(trj_slices: list) -> None:
    '''
    calculate ratios of ft of CHL-CHL contacts over all LIP-LIP contacts
    ci obtained using bootstrapping
    '''
    fname = (PATH / 'notebooks' / 'contacts' /
             f'pl_pl_over_lip_lip_ratios_'
             f'{trj_slices[0].b}-{trj_slices[0].e}-'
             f'{trj_slices[0].dt}.csv')

    if fname.is_file():
        return

    logging.info('retrieving ratios...')
    to_df = []
    with progress_bar as p:
        for trj in p.track(trj_slices, description='trj'):
            df = import_ft_table(PATH / trj.system.name / 'contacts' /
                                 'lip_dc_dc_ft.csv', False)

            # bootstrapping
            bootstrapped_values = []
            for _ in range(1000):
                sampled = df.sample(frac=1, replace=True)
                bootstrapped_values.append(
                    sampled[(sampled['dmn'] != 'CHL')
                            & (sampled['amn'] != 'CHL')]['dt_tm'].sum()
                    / sampled['dt_tm'].sum() * 100)

            bootstrapped_values = np.array(bootstrapped_values)
            std = np.std(bootstrapped_values)
            confidence_interval = 1.96 * (std
                                          / np.sqrt(len(bootstrapped_values)))

            row = {'index': trj.system.name,
                   'system': trj.system.name.split('_chol', 1)[0],
                   'CHL amount, %': trj.system.name.split('_chol', 1)[1],
                   'PL-PL / LIP-LIP, %': (
                       df[(df['dmn'] != 'CHL')
                          & (df['amn'] != 'CHL')]['dt_tm'].sum()
                       / df['dt_tm'].sum() * 100),
                   'ci': confidence_interval,
                   'std': std
                   }
            to_df.append(row)

    df_all = pd.DataFrame(to_df)
    df_all.to_csv(fname, index=False)
    logging.info('done.')


def plot_pl_pl_over_lip_lip_ratios(trj_slices: list) -> None:
    '''
    plot ratios of PL-PL contacts by distance and all LIP-LIP contacts
    for all systems
    '''
    logging.info('plotting ratios...')

    df = pd.read_csv(PATH / 'notebooks' / 'contacts' /
                     f'pl_pl_over_lip_lip_ratios_'
                     f'{trj_slices[0].b}-{trj_slices[0].e}-'
                     f'{trj_slices[0].dt}.csv')
    df.set_index('index', inplace=True)

    fig, axs = plt.subplots(1, 3, figsize=(20, 7),
                            sharey=True)

    width = 0.25
    palette = sns.color_palette('RdYlGn_r', 3)

    for ax, exp in zip(axs, EXPERIMENTS):
        x = np.arange(len(EXPERIMENTS[exp]))
        positions = (x - width, x, x + width)
        for c, chl_amount in enumerate([10, 30, 50]):
            systs = [i + f'_chol{chl_amount}' for i in EXPERIMENTS[exp]]
            label = chl_amount if ax == axs[0] else None
            ax.bar(positions[c], df.loc[systs, 'PL-PL / LIP-LIP, %'],
                   yerr=df.loc[systs, 'std'], width=width, ec='k',
                   color=palette[c], capsize=5,
                   error_kw={'elinewidth': 2}, label=label)
        ax.set_title(exp)
        ax.xaxis.set_ticks(x)
        ax.set_xticklabels(EXPERIMENTS[exp])
        ax.set_ylim(0)
        if ax == axs[0]:
            ax.set_ylabel('PL-PL / LIP-LIP, %')
    fig.legend(loc='upper center', title='CHL amount, %',
               bbox_to_anchor=(0.5, 0), ncol=3)
    fig.savefig(PATH / 'notebooks' / 'contacts' / 'imgs' /
                f'pl_pl_over_lip_lip_ratios_'
                f'{trj_slices[0].b}-'
                f'{trj_slices[0].e}-'
                f'{trj_slices[0].dt}.png',
                bbox_inches='tight')
    logging.info('done.')


def retrieve_chl_pl_over_lip_lip_ratios(trj_slices: list) -> None:
    '''
    calculate ratios of ft of CHL-PL contacts over all LIP-LIP contacts
    ci obtained using bootstrapping
    '''
    fname = (PATH / 'notebooks' / 'contacts' /
             f'chl_pl_over_lip_lip_ratios_'
             f'{trj_slices[0].b}-{trj_slices[0].e}-'
             f'{trj_slices[0].dt}.csv')

    if fname.is_file():
        return

    logging.info('retrieving ratios...')
    to_df = []
    with progress_bar as p:
        for trj in p.track(trj_slices, description='trj'):
            df = import_ft_table(PATH / trj.system.name / 'contacts' /
                                 'lip_dc_dc_ft.csv', False)

            # bootstrapping
            bootstrapped_values = []
            for _ in range(1000):
                sampled = df.sample(frac=1, replace=True)
                bootstrapped_values.append(
                    sampled[(sampled['dmn'] == 'CHL')
                            & (sampled['amn'] == 'CHL')]['dt_tm'].sum()
                    / sampled['dt_tm'].sum() * 100)

            bootstrapped_values = np.array(bootstrapped_values)
            std = np.std(bootstrapped_values)
            confidence_interval = 1.96 * (std
                                          / np.sqrt(len(bootstrapped_values)))

            row = {'index': trj.system.name,
                   'system': trj.system.name.split('_chol', 1)[0],
                   'CHL amount, %': trj.system.name.split('_chol', 1)[1],
                   'CHL-PL / LIP-LIP, %': (
                       df[((df['dmn'] != 'CHL') & (df['amn'] == 'CHL'))
                          | ((df['dmn'] == 'CHL') & (df['amn'] != 'CHL'))
                          ]['dt_tm'].sum()
                       / df['dt_tm'].sum() * 100),
                   'ci': confidence_interval,
                   'std': std
                   }
            to_df.append(row)

    df_all = pd.DataFrame(to_df)
    df_all.to_csv(fname, index=False)
    logging.info('done.')


def plot_chl_pl_over_lip_lip_ratios(trj_slices: list) -> None:
    '''
    plot ratios of CHL-PL contacts by distance and all LIP-LIP contacts
    for all systems
    '''
    logging.info('plotting ratios...')

    df = pd.read_csv(PATH / 'notebooks' / 'contacts' /
                     f'chl_pl_over_lip_lip_ratios_'
                     f'{trj_slices[0].b}-{trj_slices[0].e}-'
                     f'{trj_slices[0].dt}.csv')
    df.set_index('index', inplace=True)

    fig, axs = plt.subplots(1, 3, figsize=(20, 7),
                            sharey=True)

    width = 0.25
    palette = sns.color_palette('RdYlGn_r', 3)

    for ax, exp in zip(axs, EXPERIMENTS):
        x = np.arange(len(EXPERIMENTS[exp]))
        positions = (x - width, x, x + width)
        for c, chl_amount in enumerate([10, 30, 50]):
            systs = [i + f'_chol{chl_amount}' for i in EXPERIMENTS[exp]]
            label = chl_amount if ax == axs[0] else None
            ax.bar(positions[c], df.loc[systs, 'CHL-PL / LIP-LIP, %'],
                   yerr=df.loc[systs, 'std'], width=width, ec='k',
                   color=palette[c], capsize=5,
                   error_kw={'elinewidth': 2}, label=label)
        ax.set_title(exp)
        ax.xaxis.set_ticks(x)
        ax.set_xticklabels(EXPERIMENTS[exp])
        ax.set_ylim(0)
        if ax == axs[0]:
            ax.set_ylabel('CHL-PL / LIP-LIP, %')
    fig.legend(loc='upper center', title='CHL amount, %',
               bbox_to_anchor=(0.5, 0), ncol=3)
    fig.savefig(PATH / 'notebooks' / 'contacts' / 'imgs' /
                f'chl_pl_over_lip_lip_ratios_'
                f'{trj_slices[0].b}-'
                f'{trj_slices[0].e}-'
                f'{trj_slices[0].dt}.png',
                bbox_inches='tight')
    logging.info('done.')


def plot_chl_lip_over_lip_lip_ratios(trj_slices: list) -> None:
    '''
    plot ratios of CHL-PL contacts by distance and all LIP-LIP contacts
    for all systems
    '''
    logging.info('plotting ratios...')

    df = pd.read_csv(PATH / 'notebooks' / 'contacts' /
                     f'chl_lip_over_lip_lip_ratios_'
                     f'{trj_slices[0].b}-'
                     f'{trj_slices[0].e}-'
                     f'{trj_slices[0].dt}.csv')

    df = df[df.columns.intersection(
        ['system', 'experiment', 'CHL amount, %', 'CHL-PL / LIP-LIP, %'])]

    df['experiment'] = df['system'].apply(
        lambda x: get_keys_by_value(x, EXPERIMENTS))
    df = df.explode('experiment')
    order = {'dmpc': 0, 'dppc_325': 1, 'dspc': 2, 'popc': 3,
             'dopc': 4, 'dopc_dops50': 5, 'dops': 6}
    df = df.sort_values(by=['system'], key=lambda x: x.map(order))

    g = sns.FacetGrid(df, col='experiment', height=7,
                      aspect=0.75, sharex=False, dropna=True)
    g.map_dataframe(sns.barplot, x='system', y='CHL-PL / LIP-LIP, %',
                    hue='CHL amount, %', saturation=1, ci='sd',
                    errwidth=2, capsize=.1,
                    palette='RdYlGn_r', edgecolor='k', errcolor='k')
    g.axes[0][1].legend(ncol=len(df['CHL amount, %'].unique()),
                        title='CHL amount, %',
                        loc='upper center', bbox_to_anchor=(0.5, -0.15))
    # g.add_legend(title=legend_title)
    g.set_titles(col_template='{col_name}')

    plt.savefig(PATH / 'notebooks' / 'contacts' / 'imgs' /
                f'chl_lip_over_lip_lip_ratios_'
                f'{trj_slices[0].b}-'
                f'{trj_slices[0].e}-'
                f'{trj_slices[0].dt}.png',
                bbox_inches='tight')
    logging.info('done.')


def create_chl_lip_hb_with_groups(trj_slices: list) -> None:
    '''
    obtain hb probabilities and lifetimes between CHL and PL
    from all systems by groups
    '''
    fname = (PATH / 'notebooks' / 'contacts' /
             'chl_lip_hbonds_pl_groups_'
             f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')

    if fname.is_file():
        return

    dfs = []
    for trj in trj_slices:
        chl_lip = import_ft_table(
            PATH / trj.system.name / 'contacts' / 'lip_hb_hb_ft.csv')
        df = chl_lip[(chl_lip['dmn'] == 'CHL')
                     & (chl_lip['amn'] != 'CHL')
                     ].copy().reset_index(drop=True)
        new_columns = pd.DataFrame(
            {'index': [trj.system.name] * len(df),
             'system': [trj.system.name.split('_chol', 1)[0]] * len(df),
             'CHL amount, %': [trj.system.name.split('_chol', 1)[1]] * len(df)
             })
        df = pd.concat([new_columns, df], axis=1)
        dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        determine_pl_group = {'Ser': ['OH9', 'OH10', 'NH7'],
                              'PO4': ['OH2', 'OH3', 'OH4', 'OG3'],
                              "ROR'": ['OG1', 'OG2'],
                              "RC(O)R'": ['OA1', 'OB1']}
        # need to reverse this dict for vectorized
        # assigning of groups later
        reverse_lookup_dict = {}
        for key, values in determine_pl_group.items():
            for value in values:
                reverse_lookup_dict[value] = key
                df['PL group'] = np.vectorize(
                    reverse_lookup_dict.get)(df['aan'])

                df.to_csv(fname, index=False)


def bootstrap_chl_lip_hb_by_group(grouped, n: int = 1000) -> list:
    '''
    get confidence interval of mean values of ratios of
    all lifetimes of hb between CHL and current PL group
    and all lifetimes of hb between CHL and all PL groups

    n -- cycles of bootstrapping
    '''
    # bootstrapping for confidence intervals
    sampled_values = []

    def bootstrap_func(group):
        sample = group.sample(frac=1, replace=True)  # Bootstrap sampling
        return sample

    with progress_bar as p:
        for _ in p.track(range(n), description='bootstrapping'):
            resampled_df = grouped.apply(bootstrap_func).reset_index(drop=True)
            grouped_sum = resampled_df.groupby(
                ['index', 'system', 'CHL amount, %']
            )['dt_tm'].sum().reset_index()
            grouped_by_group_sum = resampled_df.groupby(
                ['index', 'system', 'CHL amount, %', 'PL group'])['dt_tm'].sum(
            ).reset_index()
            merged_df = pd.merge(grouped_by_group_sum, grouped_sum,
                                 on=['index', 'system', 'CHL amount, %'],
                                 suffixes=['', '_sum'])
            merged_df['dt_fr'] = (
                merged_df['dt_tm'] / merged_df['dt_tm_sum'] * 100)
            sampled_values.append(merged_df['dt_fr'].values)

    sampled_values = np.array(sampled_values).T

    confidence_level = 0.95
    percentile_low = (1 - confidence_level) / 2 * 100
    percentile_high = 100 - percentile_low

    quantiles = []
    for i in sampled_values:
        quantiles.append(np.percentile(i, [percentile_low, percentile_high]))

    return quantiles


def process_df_for_chl_lip_groups_hb(
        df: pd.DataFrame, n: int = 1000) -> pd.DataFrame:
    '''
    drop unnecessary columns,
    calculate ratio of all lifetimes of each PL group per all ligfetimes
    of hb between CHL and PL

    n -- cycles of bootstrapping
    '''
    df.drop(columns=['dmi', 'dmn', 'dan', 'dai', 'ami', 'amn', 'aan', 'aai'],
            inplace=True)

    grouped = df.groupby(['index', 'system', 'CHL amount, %', 'PL group'],
                         as_index=False)

    # creating sum of dt_tm by 'system', 'CHL amount, %'
    grouped_sum = df.groupby(
        ['index', 'system', 'CHL amount, %'])['dt_tm'].sum().reset_index()

    grouped_by_group_sum = df.groupby(
        ['index', 'system', 'CHL amount, %', 'PL group'])['dt_tm'].sum(
    ).reset_index()
    merged_df = pd.merge(grouped_by_group_sum, grouped_sum,
                         on=['index', 'system', 'CHL amount, %'],
                         suffixes=['', '_sum'])
    merged_df['dt_fr'] = merged_df['dt_tm'] / merged_df['dt_tm_sum'] * 100

    quantiles = bootstrap_chl_lip_hb_by_group(grouped, n)
    merged_df[['lq', 'hq']] = pd.DataFrame(quantiles)
    df = merged_df.set_index('index')
    df['dt_fr_std'] = ((df['hq'] - df['dt_fr']) + (df['dt_fr'] - df['lq'])) / 2
    return df


def plot_chl_lip_groups_hb(trj_slices: list) -> None:
    '''
    plot hydrogen bonds probabilities of hb formation between CHL
    and different PL groups
    '''
    #pylint: disable = too-many-locals
    df = pd.read_csv(PATH / 'notebooks' / 'contacts' /
                     'chl_lip_hbonds_pl_groups_'
                     f'{trj_slices[0].b}-{trj_slices[0].e}-'
                     f'{trj_slices[0].dt}.csv')

    logging.info('processing dataset...')
    df = process_df_for_chl_lip_groups_hb(df, 1000)

    logging.info('drawing plot...')
    fig, axs = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    for ax, exp in zip(axs, EXPERIMENTS):
        x = np.arange(len(EXPERIMENTS[exp]))
        width = 0.07
        positions = (
            (x - 6 * width, x - 3 * width, x,             x + 3 * width),
            (x - 5 * width, x - 2 * width, x + 1 * width, x + 4 * width),
            (x - 4 * width, x - 1 * width, x + 2 * width, x + 5 * width))
        alphas = (1, 0.66, 0.33)

        if ax == axs[0]:
            chl_labels = ['10% CHL', '30% CHL', '50% CHL']
        else:
            chl_labels = [None, None, None]

        groups = ['PO4', "ROR'", "RC(O)R'", 'Ser']

        for pos, alpha, label, chl_amount in zip(
                positions, alphas, chl_labels, df['CHL amount, %'].unique()):

            systs = [i + f'_chol{chl_amount}' for i in EXPERIMENTS[exp]]

            if label is not None:
                labels = [i + ', ' + label for i in groups]
            else:
                labels = [None for _ in groups]

            for c, group in enumerate(groups):
                values_to_draw = df.loc[systs, :][
                    df.loc[systs, :]['PL group'] == group]
                for syst in systs:
                    if syst not in values_to_draw.index:
                        values_to_draw.loc[syst, 'dt_fr'] = 0
                        values_to_draw.loc[syst, 'dt_fr_std'] = 0
                values_to_draw = values_to_draw.loc[systs]

                ax.bar(pos[c], values_to_draw['dt_fr'], width,
                       label=labels[c], color=f'C{c}', alpha=alpha)
                # single black edges independent on alpha
                ax.bar(pos[c], values_to_draw['dt_fr'], width,
                       yerr=values_to_draw['dt_fr_std'], ec='k',
                       fill=False, lw=1, error_kw={'elinewidth': 1})

        ax.set_title(exp)
        ax.xaxis.set_ticks(x)
        ax.set_xticklabels(EXPERIMENTS[exp])
        ax.set_ylim(0)
        if ax == axs[0]:
            ax.set_ylabel('probability of hydrogen bond formantion')
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3)
        fig.savefig(
            PATH / 'notebooks' / 'contacts' / 'imgs' /
            f'hb_pl_groups_dt{trj_slices[0].dt}.png',
            bbox_inches='tight', dpi=300)
    logging.info('done.')


@ app.command()
def plot(ctx: typer.Context,
         hbonds: bool = typer.Option(
             False, help='водородные связи '
             '(ХС-липид, ХС-вода, липид-липид, липид-вода): '
             'вероятность возникновения связи  '
             '(общее число контактов на мол за 1 пс) + время жизни'),
         dcontacts: bool = typer.Option(
             False, help='контакты липид-ХС как % от всех лип-лип контактов')):
    '''
    plot contacts
    '''
    # TODO: add средняя площадь экспонирования для каждой молекулы
    trj, tpr, b, e, dt, _, verbose, _ = ctx.obj
    sns.set(style='ticks', context='talk', palette='muted')
    initialize_logging('plot_contacts.log', verbose)
    systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in flatten(EXPERIMENTS.values())])
    systems = list(dict.fromkeys(systems))
    trajectory_slices = [TrajectorySlice(System(PATH, s, trj, tpr), b, e, dt)
                         for s in systems]
    trajectory_slices_only_chl = [trj for trj in trajectory_slices
                                  if 'chol' in trj.system.name]

    if hbonds:
        create_contacts_mean_std_df(trajectory_slices)
        plot_hb_probabilities(trajectory_slices)
        plot_hb_lt_distributions(trajectory_slices)
        plot_chl_lip_groups_hb(trajectory_slices_only_chl)

    if dcontacts:
        # retrieve_chl_lip_over_lip_lip_ratios(trajectory_slices_only_chl)
        # plot_chl_lip_over_lip_lip_ratios(trajectory_slices_only_chl)
        retrieve_chl_chl_over_lip_lip_ratios(trajectory_slices_only_chl)
        plot_chl_chl_over_lip_lip_ratios(trajectory_slices_only_chl)
        retrieve_pl_pl_over_lip_lip_ratios(trajectory_slices_only_chl)
        plot_pl_pl_over_lip_lip_ratios(trajectory_slices_only_chl)
        retrieve_chl_pl_over_lip_lip_ratios(trajectory_slices_only_chl)
        plot_chl_pl_over_lip_lip_ratios(trajectory_slices_only_chl)


@duration
@app.command()
def get(ctx: typer.Context,
        contact_types: List[str] = typer.Argument(
        ...,
        help='tasks to run '
        '[dim](pick from [bold]hb_lip, hb_sol, dc_lip, dc_sol[/])[/]'),
        ):
    '''
    run [bold]impulse cont_stat[/] on all systems to obtain contacts info
    '''
    trj, tpr, b, e, dt, n_workers, verbose, messages = ctx.obj

    initialize_logging('get_contacts.log', verbose)
    systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in flatten(EXPERIMENTS.values())])
    systems = list(dict.fromkeys(systems))

    trajectory_slices = [TrajectorySlice(System(PATH, s, trj, tpr), b, e, dt)
                         for s in systems]

    trajectory_slices_only_chl = [trj for trj in trajectory_slices
                                  if 'chol' in trj.system.name]

    grp2_selectors = [trj.system.pl_selector()[1:-4]
                      if 'dopc_dops50' not in trj.system.name
                      else trj.system.pl_selector(0)[1:-4] + '///|' +
                      trj.system.pl_selector(1)[1:-6]
                      for trj in trajectory_slices_only_chl]

    if 'hb_lip' in contact_types:
        logging.info('started intrabilayer hbonds calculation')
        multiproc(calculate_contacts,
                  trajectory_slices,
                  ('lip' for _ in trajectory_slices),
                  (None for _ in trajectory_slices),
                  ('hb' for _ in trajectory_slices),
                  (1 for _ in trajectory_slices),
                  n_workers=n_workers,
                  messages=messages,
                  descr='intrabilayer hbonds'
                  )
        logging.info('intrabilayer hbonds calculation done')
        logging.info('')

    if 'hb_sol' in contact_types:
        logging.info('started hbonds with water calculation')
        multiproc(calculate_contacts,
                  trajectory_slices,
                  ('lip' for _ in trajectory_slices),
                  ('SOL' for _ in trajectory_slices),
                  ('hb' for _ in trajectory_slices),
                  (1 for _ in trajectory_slices),
                  n_workers=n_workers,
                  messages=messages,
                  descr='hbonds with water'
                  )
        logging.info('hbonds with water calculation done')
        logging.info('')

    if 'dc_lip' in contact_types:
        logging.info('started intrabilayer contacts calculation')
        # multiproc(calculate_contacts,
        #           trajectory_slices_only_chl,
        #           ('CHOL' for _ in trajectory_slices_only_chl),
        #           (grp2_selectors),
        #           ('dc' for _ in trajectory_slices_only_chl),
        #           (0 for _ in trajectory_slices_only_chl),
        #           n_workers=n_workers,
        #           messages=messages,
        #           descr='intrabilayer contacts'
        #           )
        multiproc(calculate_contacts,
                  trajectory_slices_only_chl,
                  ('lip' for _ in trajectory_slices_only_chl),
                  (None for _ in trajectory_slices_only_chl),
                  ('dc' for _ in trajectory_slices_only_chl),
                  (0 for _ in trajectory_slices_only_chl),
                  n_workers=n_workers,
                  messages=messages,
                  descr='intrabilayer contacts'
                  )
        logging.info('intrabilayer contacts calculation done')
        logging.info('')

    if 'dc_sol' in contact_types:
        logging.info('started contacts with water calculation')
        multiproc(calculate_contacts,
                  trajectory_slices_only_chl,
                  ('CHOL' for _ in trajectory_slices_only_chl),
                  ('SOL' for _ in trajectory_slices_only_chl),
                  ('dc' for _ in trajectory_slices_only_chl),
                  (0 for _ in trajectory_slices_only_chl),
                  n_workers=n_workers,
                  messages=messages,
                  descr='contacts with water'
                  )
        logging.info('contacts with water calculation done')
        logging.info('')
    logging.info('done.')


@app.callback()
def callback(ctx: typer.Context,
             trj: str = typer.Option(
                 'pbcmol_201.xtc', '-trj', help='name of trajectory files',
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
    '''set of utilities to obtain and plot contacts data'''
    ctx.obj = (trj, tpr, b, e, dt, n_workers, verbose, messages)
    # store command arguments


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
# trj = trj_slices[0]
