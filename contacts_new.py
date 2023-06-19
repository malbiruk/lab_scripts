'''
n of SOL interactions per PL (hb, dc)

chl without hbonds, ratio of hbonds mols

sol inside bilayer -- hbonds (chl vs pl)
'''
import logging
from typing import List

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from contacts import import_ft_table
from modules.constants import EXPERIMENTS, PATH
from modules.general import flatten, initialize_logging, multiproc
from modules.traj import System, TrajectorySlice
from z_slices_angles_contacts import (contacts_to_single_tables, get_n_chl_df,
                                      update_contacts_tables_single_trj)

app = typer.Typer(rich_markup_mode='rich', add_completion=False)


def get_n_pl_df(trj_slices: list):
    '''
    read or create file with n of PL molecules in systems
    '''
    n_pl_molecules_file = (PATH / 'notebooks' / 'integral_parameters' /
                           f'n_pl_{trj_slices[0].b}-{trj_slices[0].e}-'
                           f'{trj_slices[0].dt}.csv')

    if n_pl_molecules_file.is_file():
        n_pl_df = pd.read_csv(n_pl_molecules_file)

    else:
        logging.info('calculating n of PL molecules...')
        n_pl_to_df = {
            'index': [],
            'system': [],
            'CHL amount, %': [],
            'timepoint': [],
            'n_pl': []
        }
        for trj in trj_slices:
            trj.generate_slice_with_gmx()
            u = mda.Universe(
                f'{trj.system.dir}/md/{trj.system.tpr}',
                f'{trj.system.dir}/md/pbcmol_{trj.b}-{trj.e}-{trj.dt}.xtc')

            for ts in u.trajectory:
                system_name = ''.join([i for i in trj.system.name
                                       if not i.isdigit()]).upper()
                pls_in_syst = [i for i in system_name.split('_') if i != 'CHOL']
                pls_in_syst[:] = [i for i in pls_in_syst if i]
                selection_string = ' or '.join(
                    [f'resname {i}' for i in pls_in_syst])
                pl_group = u.select_atoms(selection_string)
                n_pl_to_df['index'].append(trj.system.name)
                n_pl_to_df['system'].append(
                    trj.system.name.split('_chol', 1)[0])
                n_pl_to_df['CHL amount, %'].append(
                    trj.system.name.split('_chol', 1)[1]
                    if 'chol' in trj.system.name
                    else '0')
                n_pl_to_df['timepoint'].append(int(ts.time))
                n_pl_to_df['n_pl'].append(len(pl_group.residues))

        n_pl_df = pd.DataFrame(n_pl_to_df)
        n_pl_df.to_csv(n_pl_molecules_file, index=False)
    return n_pl_df


def hb_n_sol_per_pl(trj_slices, n_workers, messages):
    '''
    plot n hbonds sol per pl in all systems
    '''
    logging.info('loading data...')
    lip_sol_rchist_full = (PATH / 'notebooks' / 'contacts' /
                           'lip_SOL_hb_hb_full_'
                           f'{trj_slices[0].b}-{trj_slices[0].e}'
                           f'-{trj_slices[0].dt}_'
                           'rchist_full.csv')

    if not lip_sol_rchist_full.is_file():
        logging.warning('%s does not exist. creating...',
                        lip_sol_rchist_full.name)
        multiproc(update_contacts_tables_single_trj,
                  trj_slices,
                  [['lip_SOL_hb_hb']] * len(trj_slices),
                  show_progress='multiple',
                  n_workers=n_workers,
                  messages=messages,
                  descr='updating contact tables')
        contacts_to_single_tables(trj_slices, '_full',
                                  ['lip_SOL_hb_hb'])

    df_sol = pd.read_csv(lip_sol_rchist_full)
    n_pl_df = get_n_pl_df(trj_slices)

    logging.info('processing data...')
    table_with_probs_sol = (df_sol.groupby(
        ['system', 'CHL amount, %', 'timepoint'],
        as_index=False)['dmn']
        .agg('count')
        .rename(columns={'dmn': 'n_contacts'})
    )
    table_with_probs_sol = table_with_probs_sol.merge(
        n_pl_df,
        on=['system', 'CHL amount, %', 'timepoint'], how='left')

    table_with_probs_sol['n contacts per molecule'] = (
        table_with_probs_sol['n_contacts']
        / table_with_probs_sol['n_pl'])

    logging.info('drawing plots...')
    fig, axs = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

    for ax, exp in zip(axs, EXPERIMENTS):
        data = table_with_probs_sol[
            table_with_probs_sol['system'].isin(EXPERIMENTS[exp])]
        sns.barplot(data=data,
                    x='system', y='n contacts per molecule',
                    hue='CHL amount, %', ax=ax,
                    edgecolor='k', palette='RdYlGn_r', errorbar='sd'
                    )
        ax.set_title(exp)
        if ax != axs[1]:
            ax.legend([], [], frameon=False)
        if ax != axs[0]:
            ax.set_ylabel('')

    sns.move_legend(axs[1], loc='upper center',
                    bbox_to_anchor=(0.5, -0.2), ncol=6)
    fig.patch.set_facecolor('white')
    fig.savefig(
        PATH / 'notebooks' / 'contacts' / 'imgs' /
        'hb_sol_per_pl_n_contacts_'
        f'dt{trj_slices[0].dt}.png',
        bbox_inches='tight', dpi=300)


def hb_n_pl_per_pl(trj_slices, n_workers, messages):
    '''
    plot n hbonds chl per pl in all systems
    '''
    logging.info('loading data...')
    lip_rchist_full = (PATH / 'notebooks' / 'contacts' /
                       'lip_hb_hb_full_'
                       f'{trj_slices[0].b}-{trj_slices[0].e}'
                       f'-{trj_slices[0].dt}_'
                       'rchist_full.csv')

    if not lip_rchist_full.is_file():
        logging.warning('%s does not exist. creating...',
                        lip_rchist_full.name)
        multiproc(update_contacts_tables_single_trj,
                  trj_slices,
                  [['lip_hb_hb']] * len(trj_slices),
                  show_progress='multiple',
                  n_workers=n_workers,
                  messages=messages,
                  descr='updating contact tables')
        contacts_to_single_tables(trj_slices, '_full',
                                  ['lip_hb_hb'])

    df_pl = pd.read_csv(lip_rchist_full)
    n_pl_df = get_n_pl_df(trj_slices)

    logging.info('processing data...')
    table_with_probs_pl = (df_pl[(df_pl['dmn'] != 'CHL') &
                                 (df_pl['dmn'] != 'SOL') &
                                 (df_pl['amn'] != 'CHL') &
                                 (df_pl['amn'] != 'SOL')].groupby(
        ['system', 'CHL amount, %', 'timepoint'],
        as_index=False)['dmn']
        .agg('count')
        .rename(columns={'dmn': 'n_contacts'})
    )

    table_with_probs_pl = table_with_probs_pl.merge(
        n_pl_df,
        on=['system', 'CHL amount, %', 'timepoint'], how='left')

    table_with_probs_pl['n contacts per molecule'] = (
        table_with_probs_pl['n_contacts'] * 2
        / table_with_probs_pl['n_pl'])

    logging.info('drawing plots...')
    fig, axs = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

    for ax, exp in zip(axs, EXPERIMENTS):
        data = table_with_probs_pl[
            table_with_probs_pl['system'].isin(EXPERIMENTS[exp])]
        try:
            sns.barplot(data=data,
                        x='system', y='n contacts per molecule',
                        hue='CHL amount, %', ax=ax,
                        edgecolor='k', palette='RdYlGn_r', errorbar='sd'
                        )
            ax.yaxis.set_tick_params(labelbottom=True)
        except ValueError:
            continue
        finally:
            ax.set_title(exp)
            if ax != axs[2]:
                ax.legend([], [], frameon=False)

    sns.move_legend(axs[2], loc='upper center',
                    bbox_to_anchor=(0.5, -0.2), ncol=6)
    fig.patch.set_facecolor('white')
    fig.savefig(
        PATH / 'notebooks' / 'contacts' / 'imgs' /
        'hb_pl_per_pl_n_contacts_'
        f'dt{trj_slices[0].dt}.png',
        bbox_inches='tight', dpi=300)


def hb_n_chl_per_pl(trj_slices):
    '''
    plot n hbonds chl per pl in all systems
    '''
    logging.info('loading data...')
    lip_rchist_full = (PATH / 'notebooks' / 'contacts' /
                       'lip_hb_hb_chl_angle_z_mhp_'
                       f'{trj_slices[0].b}-{trj_slices[0].e}'
                       f'-{trj_slices[0].dt}_'
                       'rchist_full.csv')

    if not lip_rchist_full.is_file():
        raise ValueError(
            '%s does not exist. use z_slices_angles_contacts script.')

    df_chl = pd.read_csv(lip_rchist_full)
    n_pl_df = get_n_pl_df(trj_slices)

    logging.info('processing data...')
    table_with_probs_chl = (df_chl[(df_chl['other_name'] != 'CHL') &
                                   (df_chl['other_name'] != 'SOL')].groupby(
        ['system', 'CHL amount, %', 'timepoint'],
        as_index=False)['other_name']
        .agg('count')
        .rename(columns={'other_name': 'n_contacts'})
    )

    table_with_probs_chl = table_with_probs_chl.merge(
        n_pl_df,
        on=['system', 'CHL amount, %', 'timepoint'], how='left')

    table_with_probs_chl['n contacts per molecule'] = (
        table_with_probs_chl['n_contacts']
        / table_with_probs_chl['n_pl'])

    logging.info('drawing plots...')
    fig, axs = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

    for ax, exp in zip(axs, EXPERIMENTS):
        data = table_with_probs_chl[
            table_with_probs_chl['system'].isin(EXPERIMENTS[exp])]
        sns.barplot(data=data,
                    x='system', y='n contacts per molecule',
                    hue='CHL amount, %', ax=ax,
                    edgecolor='k', palette='RdYlGn_r', errorbar='sd'
                    )
        ax.set_title(exp)
        if ax != axs[1]:
            ax.legend([], [], frameon=False)
        if ax != axs[0]:
            ax.set_ylabel('')

    sns.move_legend(axs[1], loc='upper center',
                    bbox_to_anchor=(0.5, -0.2), ncol=6)
    fig.patch.set_facecolor('white')
    fig.savefig(
        PATH / 'notebooks' / 'contacts' / 'imgs' /
        'hb_chl_per_pl_n_contacts_'
        f'dt{trj_slices[0].dt}.png',
        bbox_inches='tight', dpi=300)


def dc_n_chl_per_pl(trj_slices):
    '''
    plot n dc chl per pl in all systems
    '''
    logging.info('loading data...')
    lip_rchist_full = (PATH / 'notebooks' / 'contacts' /
                       'lip_dc_dc_chl_angle_z_mhp_'
                       f'{trj_slices[0].b}-{trj_slices[0].e}'
                       f'-{trj_slices[0].dt}_'
                       'rchist_full.csv')

    if not lip_rchist_full.is_file():
        raise ValueError(
            '%s does not exist. use z_slices_angles_contacts script.')

    df_chl = pd.read_csv(lip_rchist_full)
    n_pl_df = get_n_pl_df(trj_slices)

    logging.info('processing data...')
    table_with_probs_chl = (df_chl[(df_chl['other_name'] != 'CHL') &
                                   (df_chl['other_name'] != 'SOL')].groupby(
        ['system', 'CHL amount, %', 'timepoint'],
        as_index=False)['other_name']
        .agg('count')
        .rename(columns={'other_name': 'n_contacts'})
    )

    table_with_probs_chl = table_with_probs_chl.merge(
        n_pl_df,
        on=['system', 'CHL amount, %', 'timepoint'], how='left')

    table_with_probs_chl['n contacts per molecule'] = (
        table_with_probs_chl['n_contacts']
        / table_with_probs_chl['n_pl'])

    logging.info('drawing plots...')
    fig, axs = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

    for ax, exp in zip(axs, EXPERIMENTS):
        data = table_with_probs_chl[
            table_with_probs_chl['system'].isin(EXPERIMENTS[exp])]
        sns.barplot(data=data,
                    x='system', y='n contacts per molecule',
                    hue='CHL amount, %', ax=ax,
                    edgecolor='k', palette='RdYlGn_r', errorbar='sd'
                    )
        ax.set_title(exp)
        if ax != axs[1]:
            ax.legend([], [], frameon=False)
        if ax != axs[0]:
            ax.set_ylabel('')

    sns.move_legend(axs[1], loc='upper center',
                    bbox_to_anchor=(0.5, -0.2), ncol=6)
    fig.patch.set_facecolor('white')
    fig.savefig(
        PATH / 'notebooks' / 'contacts' / 'imgs' /
        'dc_chl_per_pl_n_contacts_'
        f'dt{trj_slices[0].dt}.png',
        bbox_inches='tight', dpi=300)


def bootstr_mol_sol_over_lip_sol_ratios(trj: TrajectorySlice, chl: bool):
    '''
    helper function for retrieve_mol_sol_over_lip_sol_ratios
    '''
    df = import_ft_table(PATH / trj.system.name / 'contacts' /
                         'lip_SOL_hb_hb_ft.csv', True)
    acceptors = (df['amn'] == 'CHL') if chl else (df['amn'] != 'CHL')
    mol = 'CHL' if chl else 'PL'

    # bootstrapping
    bootstrapped_values = []
    for _ in range(1000):
        sampled = df.sample(frac=1, replace=True)
        sampled_acceptors = ((sampled['amn'] == 'CHL')
                             if chl else (sampled['amn'] != 'CHL'))
        bootstrapped_values.append(
            sampled[(sampled['dmn'] == 'SOL')
                    & sampled_acceptors]['dt_tm'].sum()
            / sampled[(sampled['dmn'] == 'SOL')]['dt_tm'].sum() * 100)

    bootstrapped_values = np.array(bootstrapped_values)
    std = np.std(bootstrapped_values)
    confidence_interval = 1.96 * (std
                                  / np.sqrt(len(bootstrapped_values)))

    row = {'index': trj.system.name,
           'system': trj.system.name.split('_chol', 1)[0],
           'CHL amount, %': (
               trj.system.name.split('_chol', 1)[1]
               if 'chol' in trj.system.name else '0'),
           f'{mol}-SOL / LIP-SOL, %': (
               df[(df['dmn'] == 'SOL')
                  & acceptors]['dt_tm'].sum()
               / df[(df['dmn'] == 'SOL')]['dt_tm'].sum() * 100),
           'ci': confidence_interval,
           'std': std
           }

    return row


def retrieve_mol_sol_over_lip_sol_ratios(trj_slices: list, chl: bool) -> None:
    '''
    calculate ratios of hb lt of CHL-SOL contacts over all LIP-SOL contacts
    ci obtained using bootstrapping
    '''
    fname = (PATH / 'notebooks' / 'contacts' /
             f'chl_sol_over_lip_sol_ratios_'
             f'{trj_slices[0].b}-{trj_slices[0].e}-'
             f'{trj_slices[0].dt}.csv' if chl else
             PATH / 'notebooks' / 'contacts' /
             f'pl_sol_over_lip_sol_ratios_'
             f'{trj_slices[0].b}-{trj_slices[0].e}-'
             f'{trj_slices[0].dt}.csv')

    if fname.is_file():
        logging.info('%s already calculated, skipping,,,', fname.name)
        return

    logging.info('retrieving ratios...')
    res = multiproc(bootstr_mol_sol_over_lip_sol_ratios,
                    trj_slices,
                    (chl for _ in range(len(trj_slices))),
                    n_workers=len(trj_slices))

    df_all = pd.DataFrame(list(res.values()))
    df_all.to_csv(fname, index=False)
    logging.info('done.')


def plot_chl_sol_over_lip_sol_ratios(trj_slices: list, chl: bool) -> None:
    '''
    plot ratios of CHL-SOL or PL-SOL hbonds lt and all LIP-SOL contacts
    for all systems
    '''
    # pylint: disable = too-many-locals
    logging.info('plotting ratios...')

    fname = (PATH / 'notebooks' / 'contacts' /
             f'chl_sol_over_lip_sol_ratios_'
             f'{trj_slices[0].b}-{trj_slices[0].e}-'
             f'{trj_slices[0].dt}.csv' if chl else
             PATH / 'notebooks' / 'contacts' /
             f'pl_sol_over_lip_sol_ratios_'
             f'{trj_slices[0].b}-{trj_slices[0].e}-'
             f'{trj_slices[0].dt}.csv')

    mol = 'CHL' if chl else 'PL'

    df = pd.read_csv(fname)
    df.set_index('index', inplace=True)

    fig, axs = plt.subplots(1, 3, figsize=(20, 7),
                            sharey=True)

    width = 0.25
    palette = sns.color_palette('RdYlGn_r', 3)

    for ax, exp in zip(axs, EXPERIMENTS):
        x = np.arange(len(EXPERIMENTS[exp]))
        positions = (x - width, x, x + width)
        for c, chl_amount in enumerate([10, 30, 50]):
            systs = (
                EXPERIMENTS[exp] if chl_amount == 0
                else [i + f'_chol{chl_amount}' for i in EXPERIMENTS[exp]])
            label = chl_amount if ax == axs[0] else None
            ax.bar(positions[c], df.loc[systs, f'{mol}-SOL / LIP-SOL, %'],
                   yerr=df.loc[systs, 'std'], width=width, ec='k',
                   color=palette[c], capsize=5,
                   error_kw={'elinewidth': 2}, label=label)
        ax.set_title(exp)
        ax.xaxis.set_ticks(x)
        ax.set_xticklabels(EXPERIMENTS[exp])
        ax.set_ylim(0)
        if ax == axs[0]:
            ax.set_ylabel(f'{mol}-SOL / LIP-SOL, %')
    fig.legend(loc='upper center', title='CHL amount, %',
               bbox_to_anchor=(0.5, 0), ncol=3)
    fig.savefig(PATH / 'notebooks' / 'contacts' / 'imgs' /
                f'hb_{mol.lower()}_sol_over_lip_sol_ratios_'
                f'{trj_slices[0].b}-'
                f'{trj_slices[0].e}-'
                f'{trj_slices[0].dt}.png',
                bbox_inches='tight')
    logging.info('done.')


def chl_percentage_hbonds_interactors(trj_slices: list):
    '''
    plot percentage of CHL molecules with no hbonds, PL and SOL hbonds
    '''
    logging.info('loading data...')
    df = pd.read_csv(PATH / 'notebooks' / 'contacts' /
                     'hb_chl_angle_z_mhp_'
                     f'{trj_slices[0].b}-{trj_slices[0].e}'
                     f'-{trj_slices[0].dt}.csv',
                     usecols=['system', 'CHL amount, %', 'timepoint',
                              'chl_index', 'other_name'],
                     low_memory=False)
    df = df[df['CHL amount, %'] != 0]
    n_chl_df = get_n_chl_df(trj_slices)

    logging.info('preprocessing data...')
    new_df = df.groupby(['system', 'CHL amount, %', 'timepoint', 'other_name'],
                        as_index=False)['chl_index'].nunique().rename(
        columns={'chl_index': 'n_chl_with_hbonds'}
    )
    probs_df = new_df.merge(
        n_chl_df,
        on=['system', 'CHL amount, %', 'timepoint'], how='left')
    probs_df['% of CHL'] = (
        probs_df['n_chl_with_hbonds'] / probs_df['n_chl'] * 100)
    new_df2 = df.groupby(['system', 'CHL amount, %', 'timepoint'],
                         as_index=False)['chl_index'].nunique().rename(
        columns={'chl_index': 'n_chl_with_hbonds'})
    no_hb_df = new_df2.merge(
        n_chl_df,
        on=['system', 'CHL amount, %', 'timepoint'], how='left')
    no_hb_df['% of CHL'] = (
        1 - no_hb_df['n_chl_with_hbonds'] / no_hb_df['n_chl']) * 100
    no_hb_df['other_name'] = 'No hbonds'
    final_df = pd.concat([probs_df, no_hb_df],
                         ignore_index=True).sort_values(
        ['system', 'CHL amount, %', 'timepoint', 'other_name'],
        ignore_index=True)

    logging.info('plotting data...')
    for exp, systs in EXPERIMENTS.items():
        fig, axs = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
        for ax, syst in zip(axs, systs):
            data = final_df[final_df['system'] == syst]
            sns.barplot(data=data,
                        x='other_name', y='% of CHL',
                        hue='CHL amount, %',
                        order=['No hbonds', 'PL', 'SOL'], ax=ax,
                        edgecolor='k', palette='RdYlGn_r', errorbar='sd'
                        )
            if ax != axs[1]:
                ax.legend([], [], frameon=False)
            ax.set_xlabel('contacting molecule')
            ax.set_ylim(0)
            ax.set_title(syst)
            if ax != axs[0]:
                ax.set_ylabel('')
        sns.move_legend(axs[1], loc='upper center',
                        bbox_to_anchor=(0.5, -0.2), ncol=6)
        fig.suptitle(exp)
        fig.patch.set_facecolor('white')
        fig.savefig(
            PATH / 'notebooks' / 'contacts' / 'imgs' /
            f'{exp}_chl_hb_perc_no_sol_pl_'
            f'dt{trj_slices[0].dt}.png',
            bbox_inches='tight', dpi=300)


@ app.command()
def plot(
    ctx: typer.Context,
    to_plot: List[str] = typer.Argument(
        ..., help='what to plot. available values: '
        'hb_n_chl_per_pl, hb_n_sol_per_pl, '
        'hb_lt_chl_sol_over_lip_sol, hb_lt_pl_sol_over_lip_sol')):
    '''
    plot contacts
    '''
    # pylint: disable = too-many-locals
    trj, tpr, b, e, dt, n_workers, verbose, messages = ctx.obj
    sns.set(style='ticks', context='talk', palette='muted')
    initialize_logging('plot_contacts_new.log', verbose)
    systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in flatten(EXPERIMENTS.values())])
    systems = list(dict.fromkeys(systems))
    trajectory_slices = [TrajectorySlice(System(PATH, s, trj, tpr), b, e, dt)
                         for s in systems]
    trajectory_slices_only_chl = [trj for trj in trajectory_slices
                                  if 'chol' in trj.system.name]

    to_exec = {
        'hb_n_chl_per_pl': [
            (hb_n_chl_per_pl, (trajectory_slices_only_chl,))],
        'dc_n_chl_per_pl': [
            (dc_n_chl_per_pl, (trajectory_slices_only_chl,))],
        'hb_n_sol_per_pl': [
            (hb_n_sol_per_pl, (trajectory_slices, n_workers, messages))],
        'hb_n_pl_per_pl': [
            (hb_n_pl_per_pl, (trajectory_slices, n_workers, messages))],
        'hb_lt_chl_sol_over_lip_sol': [
            (retrieve_mol_sol_over_lip_sol_ratios, (trajectory_slices, True)),
            (plot_chl_sol_over_lip_sol_ratios, (trajectory_slices, True))
        ],
        'hb_lt_pl_sol_over_lip_sol': [
            (retrieve_mol_sol_over_lip_sol_ratios, (trajectory_slices, False)),
            (plot_chl_sol_over_lip_sol_ratios, (trajectory_slices, False))
        ],
        'hb_chl_perc': [
            (chl_percentage_hbonds_interactors, (trajectory_slices,))],
    }

    for task in to_plot:
        for func, arguments in to_exec[task]:
            func(*arguments)


@ app.callback()
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
    # pylint: disable = too-many-arguments
    ctx.obj = (trj, tpr, b, e, dt, n_workers, verbose, messages)
    # store command arguments


# %%
if __name__ == '__main__':
    app()
