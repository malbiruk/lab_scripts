'''
This script is created to determine how angles and contacts
of CHL vary in different "zones" by z-axis.

<3, 3-6, 6-9, 9-12, 12-15, >15 A distances from bilayer center were chosen,
also CHL exposed on surface were analyzed separately

also analysis of MHP and components depening on CHL angle components
are implemented here
'''
# pylint: disable = too-many-lines

import logging
import pickle

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from chl_angle_components_density import generate_coords_comps_table
from modules.constants import EXPERIMENTS, PATH
from modules.general import flatten, initialize_logging, multiproc
from modules.traj import System, TrajectorySlice
from pandas import DataFrame

app = typer.Typer(rich_markup_mode='rich', add_completion=False)


def contacts_system_name(system_name: str) -> str:
    '''
    reformat system name to contacts dc filename
    '''
    chol = 'CHOL_' if 'chol' in system_name else ''
    base = system_name.split('_chol', 1)[0].upper().replace('_', '')
    return chol + ''.join([i for i in base if not i.isdigit()])


def read_rcnames_file(trj: TrajectorySlice, prefix: str) -> DataFrame:
    '''
    read_rcnames_file as pd.DataFrame
    '''
    if 'dc_dc' in prefix:
        usecols = [2, 3, 9, 10]
        names = ['dmi', 'dmn', 'ami', 'amn']
    elif 'lip_SOL_hb_hb' in prefix:
        usecols = [2, 3, 10, 11]
        names = ['dmi', 'dmn', 'ami', 'amn']
    elif 'lip_hb_hb' in prefix:
        usecols = [2, 3, 5, 6, 10, 11, 13, 14]
        names = ['dmi', 'dmn', 'dan', 'dai',
                 'ami', 'amn', 'aan', 'aai']
    else:
        raise ValueError(f'unrecognized prefix "{prefix}"')

    return pd.read_csv(PATH / trj.system.name / 'contacts' /
                       f'{prefix}_rcnames.csv',
                       sep=r'\s+|,', engine='python',
                       header=None,
                       skiprows=1,
                       usecols=usecols,
                       names=names)


def update_contacts_tables_single_trj(progress: dict,
                                      task_id: int,
                                      trj: TrajectorySlice) -> None:
    '''
    combine rchist and rcnames files for contact list for single trj
    '''

    logging.debug('%s...', trj.system.name)
    list_of_prefixes = [
        'lip_hb_hb', 'lip_SOL_hb_hb', 'CHOL_SOL_dc_dc', 'lip_dc_dc']

    for c, prefix in enumerate(list_of_prefixes):
        rchist_full_file = (PATH / trj.system.name / 'contacts' /
                            f'{prefix}_rchist_full.csv')

        if rchist_full_file.is_file():
            logging.debug('%s file exists, skipping...', prefix)
            return

        rcnames = read_rcnames_file(trj, prefix)
        rchist = pd.read_csv(PATH / trj.system.name / 'contacts' /
                             f'{prefix}_rchist.csv', header=None)

        # process rchist df
        rchist = rchist.T.reset_index()
        rchist['index'] = rchist['index'] + np.array(trj.b * 1000,
                                                     dtype=np.int32)
        rchist.rename(columns={'index': 'timepoint'}, inplace=True)

        # create rchist_full df
        row_indices, column_indices = np.where(rchist.iloc[:, 1:] == 1)
        column_names = rchist.columns[1:][column_indices]
        selected_rows = rcnames.loc[column_names].copy()
        selected_rows['timepoint'] = (rchist['timepoint']
                                      .iloc[row_indices].values)
        col = selected_rows.pop('timepoint')
        selected_rows.insert(0, col.name, col)
        rchist_full = selected_rows
        rchist_full.reset_index(drop=True, inplace=True)

        # save df
        rchist_full.to_csv(rchist_full_file, index=False)
        logging.debug('%s done', {prefix})
        progress[task_id] = {'progress': c + 1, 'total': len(list_of_prefixes)}
    logging.info('%s done', trj.system.name)


def get_angle_mhp_z_chol_table(trj_slices: list) -> None:
    '''
    get and save df with angle, components, mhp and z data
    per CHL molecule
    '''
    fname = (PATH / 'notebooks' / 'integral_parameters' /
             'angle_mhp_z_'
             f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')

    if fname.is_file():
        logging.info('angle_mhp_z table already calculated')
        return

    logging.info('reading coords comps table...')
    coords_comps = (PATH / 'notebooks' / 'integral_parameters' /
                    f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-'
                    f'{trj_slices[0].dt}_coords_with_comps.csv')

    if not coords_comps.is_file():
        generate_coords_comps_table(trj_slices, 100, 200, 100)

    df = pd.read_csv(coords_comps)

    logging.info('reading chol mhp table...')
    mhp_df = pd.read_csv(
        PATH / 'notebooks' / 'mhpmaps' / 'info_mhp_atoms_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}'
        '_chol_per_mol.csv')
    mhp_df.set_index('index', inplace=True)

    logging.info('updating chl indices...')
    first_chl_index = {}

    for trj in trj_slices:
        trj.generate_slice_with_gmx()
        u = mda.Universe(
            f'{trj.system.dir}/md/{trj.system.tpr}',
            f'{trj.system.dir}/md/pbcmol_{trj.b}-{trj.e}-{trj.dt}.xtc')
        ind_corr = np.min(u.select_atoms('resname CHL').resids)
        first_chl_index[trj.system.name] = ind_corr

        mhp_df.loc[trj.system.name, 'mol_ind'] = (
            mhp_df.loc[trj.system.name, 'mol_ind'] - ind_corr)

    mhp_df.rename(columns={'mol_ind': 'chl_index'}, inplace=True)
    mhp_df.reset_index(inplace=True)

    logging.info('merging tables...')
    merged_df = df.merge(
        mhp_df,
        on=['system', 'CHL amount, %', 'timepoint', 'chl_index'],
        how='left')
    merged_df['surface'] = merged_df['mhp'].notnull().astype(int)
    merged_df = merged_df.drop(['index'], axis=1)

    angle_mhp_z = merged_df[
        ['system', 'CHL amount, %', 'timepoint',
         'chl_index', 'α, °', 'z_com', 'zmem',
         '1', '2', 'nan',
         'mhp', 'surface']].copy()

    angle_mhp_z['distance to bilayer center'] = (
        angle_mhp_z['z_com'] - angle_mhp_z['zmem']).abs()

    angle_mhp_z.to_csv(fname, index=False)
    with open(PATH / 'notebooks' / 'integral_parameters' /
              'chl_first_index.pkl', 'wb') as f:
        pickle.dump(first_chl_index, f)
    logging.info('preprocessing complete.')


def z_slices_angles(trj_slices: list) -> None:
    '''
    plot chl angle distributions for different distances from bilayer center
    '''
    angle_mhp_z = pd.read_csv(
        PATH / 'notebooks' / 'integral_parameters' /
        'angle_mhp_z_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')

    angle_mhp_z['distance to bilayer center, Å'] = pd.cut(
        angle_mhp_z['distance to bilayer center'],
        bins=[0, 3, 6, 9, 12, 15, 100],
        labels=['<= 3', '3-6', '6-9', '9-12', '12-15', '> 15'])
    angle_mhp_z['α, °'] = angle_mhp_z['α, °'].abs()

    for exp, systs in EXPERIMENTS.items():
        logging.info('plotting %s...', exp)
        fig, axs = plt.subplots(3, 3, figsize=(20, 20),
                                sharex=True, sharey=True)
        c = 0
        for syst in systs:
            for chl_amount in [10, 30, 50]:
                data = angle_mhp_z[
                    (angle_mhp_z['system'] == syst) &
                    (angle_mhp_z['CHL amount, %'] == chl_amount)]
                sns.kdeplot(data=data, x='α, °',
                            hue='distance to bilayer center, Å',
                            ax=axs.flatten()[c], palette='crest_r', fill=True,
                            common_norm=False, legend=c == 7)
                axs.flatten()[c].set_title(f'{syst}, {chl_amount} % CHL')
                data = data[data['surface'] == 1]
                c += 1

        sns.move_legend(axs.flatten()[7], loc='upper center',
                        bbox_to_anchor=(0.5, -0.2), ncol=6)

        fig.suptitle(exp)
        fig.savefig(
            PATH / 'notebooks' / 'chol_tilt' / 'surface_z' /
            'z_slices_angles_'
            f'{"_".join(exp.split())}_'
            f'dt{trj_slices[0].dt}.png',
            bbox_inches='tight', dpi=300)
    logging.info('done.')


def surface_angles(trj_slices: list) -> None:
    '''
    plot chl angle distributions depending on surface exposure of chl
    '''
    angle_mhp_z = pd.read_csv(
        PATH / 'notebooks' / 'integral_parameters' /
        'angle_mhp_z_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')

    angle_mhp_z['α, °'] = angle_mhp_z['α, °'].abs()
    angle_mhp_z['surface'] = angle_mhp_z['surface'].map({1: 'yes', 0: 'no'})

    for exp, systs in EXPERIMENTS.items():
        logging.info('plotting %s...', exp)
        fig, axs = plt.subplots(3, 3, figsize=(20, 20),
                                sharex=True, sharey=True)
        axs = axs.flatten()
        c = 0
        for syst in systs:
            for chl_amount in [10, 30, 50]:
                data = angle_mhp_z[
                    (angle_mhp_z['system'] == syst) &
                    (angle_mhp_z['CHL amount, %'] == chl_amount)]
                sns.kdeplot(data=data, x='α, °',
                            hue='surface',
                            hue_order=['no', 'yes'],
                            ax=axs[c], fill=True,
                            common_norm=False, legend=c == 7)
                axs[c].set_title(f'{syst}, {chl_amount} % CHL')
                data = data[data['surface'] == 1]
                c += 1

        sns.move_legend(axs[7], loc='upper center',
                        bbox_to_anchor=(0.5, -0.2), ncol=6)

        fig.suptitle(exp)
        fig.savefig(
            PATH / 'notebooks' / 'chol_tilt' / 'surface_z' /
            'surface_angles_'
            f'{"_".join(exp.split())}_'
            f'dt{trj_slices[0].dt}.png',
            bbox_inches='tight', dpi=300)
    logging.info('done.')


def calculate_chol_surface_exposure_comps(trj_slices: list) -> None:
    '''
    calculate % of area occupied by CHL on surface depending on
    CHL tilt component
    '''
    fname = (PATH / 'notebooks' / 'mhpmaps' /
             'chol_from_all_fractions_comps_stats_'
             f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')
    # if fname.is_file():
    #     logging.info('chol fractions of surface already calculated')
    #     return

    logging.info('loading data...')
    df = pd.read_csv(PATH / 'notebooks' / 'mhpmaps' / 'info_mhp_atoms_'
                     f'{trj_slices[0].b}-{trj_slices[0].e}-'
                     f'{trj_slices[0].dt*10}.csv',
                     usecols=[1, 2, 3, 4, 5, 6])

    angle_mhp_z = pd.read_csv(
        PATH / 'notebooks' / 'integral_parameters' /
        'angle_mhp_z_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')

    with open(PATH / 'notebooks' / 'integral_parameters' /
              'chl_first_index.pkl', 'rb') as f:
        chl_first_index = pickle.load(f)

    logging.info('updating CHL indices...')
    df2 = df[df['CHL amount, %'] != 0].copy()
    df2.set_index('index', inplace=True)
    series = df2[df2['mol_name'] == 'CHL']['mol_ind']
    df2.loc[df2['mol_name'] == 'CHL', 'mol_ind'] = (
        series - series.index.map(chl_first_index))
    df2.reset_index(inplace=True)

    logging.info('adding components...')
    df_filtered = df2[df2['mol_name'] == 'CHL']
    df_merged = df_filtered.reset_index().merge(angle_mhp_z[
        ['system', 'CHL amount, %', 'timepoint', 'chl_index', '1', '2']],
        left_on=['system', 'CHL amount, %', 'timepoint', 'mol_ind'],
        right_on=['system', 'CHL amount, %', 'timepoint', 'chl_index'],
        how='left').set_index('level_0')
    df_merged = df_merged.drop(['chl_index'], axis=1)

    df2.loc[df_merged.index, ['1', '2']] = df_merged[['1', '2']]

    logging.info('calculating fractions...')
    chol_fractions = df2.groupby(
        ['index', 'system', 'CHL amount, %', 'timepoint'], as_index=False
    )[['1', '2']].agg(lambda x: np.sum(x) / len(x) * 100)

    chol_fractions.rename(columns={
        '1': 'CHL, vertical component, % of area',
        '2': 'CHL, horizontal component, % of area'
    }, inplace=True)

    chol_fractions_stats = chol_fractions.groupby(
        ['index', 'system', 'CHL amount, %'], as_index=False).agg(
        vert_perc_of_area=('CHL, vertical component, % of area', 'mean'),
        vert_perc_of_area_std=('CHL, vertical component, % of area', 'std'),
        hor_perc_of_area=('CHL, horizontal component, % of area', 'mean'),
        hor_perc_of_area_std=('CHL, horizontal component, % of area', 'std'),
    )

    logging.info('saving_results...')
    chol_fractions.to_csv(
        PATH / 'notebooks' / 'mhpmaps' /
        'chol_from_all_fractions_comps_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv',
        index=False)
    chol_fractions_stats.to_csv(fname, index=False)
    logging.info('done.')


def plot_chol_surface_exposure_comps(trj_slices: list) -> None:
    '''
    plot % of area occupied by CHL on surface depending on
    CHL tilt component
    '''
    logging.info('plotting chol_surface_exposure_comps...')
    df = pd.read_csv(
        PATH / 'notebooks' / 'mhpmaps' /
        'chol_from_all_fractions_comps_stats_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')

    df.set_index('index', inplace=True)
    palette = sns.color_palette('Paired')

    fig, axs = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

    for ax, exp in zip(axs, EXPERIMENTS):
        systems = (
            [i + '_chol10' for i in EXPERIMENTS[exp]],
            [i + '_chol30' for i in EXPERIMENTS[exp]],
            [i + '_chol50' for i in EXPERIMENTS[exp]],
        )
        width = 0.25
        x = np.arange(len(EXPERIMENTS[exp]))
        positions = [x - width, x, x + width]
        chl_amounts = [10, 30, 50]
        for c, systs in enumerate(systems):
            data = df.loc[systs, :]
            ax.bar(positions[c], data['vert_perc_of_area'],
                   yerr=data['vert_perc_of_area_std'],
                   width=width, ec='k', capsize=5,
                   color=palette[c * 2 + 1],
                   label=f'vertical component, {chl_amounts[c]}% of CHL')
            ax.bar(positions[c], data['hor_perc_of_area'],
                   yerr=data['hor_perc_of_area_std'],
                   width=width, ec='k',
                   color=palette[c * 2], capsize=5,
                   bottom=data['vert_perc_of_area'],
                   label=f'horizontal component, {chl_amounts[c]}% of CHL')

        ax.set_title(exp)
        ax.xaxis.set_ticks(x)
        ax.set_xticklabels(EXPERIMENTS[exp])

    axs[0].set_ylabel('% of area')
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    fig.savefig(
        PATH / 'notebooks' / 'mhpmaps' / 'imgs' /
        f'chol_surf_fractions_comps_dt{trj_slices[0].dt}.png',
        bbox_inches='tight', dpi=300)
    logging.info('done.')


def calculate_chol_mhp_fractions_comps(trj_slices: list) -> None:
    '''
    calculate area occupied by different mhp fractions of CHL
    on surface depending on CHL tilt component
    '''
    fname = (PATH / 'notebooks' / 'mhpmaps' /
             'for_hists_fractions_stats_chol_comps_'
             f'{trj_slices[0].b}-{trj_slices[0].e}-'
             f'{trj_slices[0].dt}.csv')
    # if fname.is_file():
    #     return

    logging.info('preprocessing data...')
    df = pd.read_csv(
        PATH / 'notebooks' / 'integral_parameters' /
        'angle_mhp_z_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')

    df_surf = df.loc[df['surface'] == 1].copy()
    df_surf['mhp_label'] = pd.cut(df_surf['mhp'],
                                  bins=[-100, -0.5, 0.5, 100],
                                  labels=['phil', 'neutr', 'phob']
                                  )

    mhp_comp = df_surf.groupby(
        ['system', 'CHL amount, %', 'timepoint', 'mhp_label'],
        as_index=False
    )[['1', '2']].mean().fillna(0)

    label_counts = df_surf.groupby(
        ['system', 'CHL amount, %', 'timepoint'], as_index=False
    )['mhp_label'].value_counts(sort=False)

    label_counts['ratio'] = label_counts.groupby(
        ['system', 'CHL amount, %', 'timepoint'],
        as_index=False)['count'].transform(lambda x: x / x.sum())

    mhp_comp[['1', '2']] = mhp_comp[['1', '2']].multiply(
        label_counts['ratio'], axis='index')

    mhp_comp.to_csv(PATH / 'notebooks' / 'mhpmaps' /
                    'for_hists_fractions_chol_comps_'
                    f'{trj_slices[0].b}-{trj_slices[0].e}-'
                    f'{trj_slices[0].dt}.csv', index=False)

    mhp_comp_stats = mhp_comp.groupby(
        ['system', 'CHL amount, %', 'mhp_label'], as_index=False).agg(
        vertical=('1', 'mean'),
        horizontal=('2', 'mean'),
        vertical_std=('1', 'std'),
        horizontal_std=('2', 'std'),
    )

    mhp_comp_stats.to_csv(fname, index=False)
    logging.info('done.')


def plot_chol_mhp_fractions_comps(trj_slices: list) -> None:
    '''
    plot area occupied by different mhp fractions of CHL
    on surface depending on CHL tilt component
    '''
    # pylint: disable = too-many-locals

    logging.info('plotting chol_mhp_fractions_comps...')
    df = pd.read_csv(PATH / 'notebooks' / 'mhpmaps' /
                     'for_hists_fractions_stats_chol_comps_'
                     f'{trj_slices[0].b}-{trj_slices[0].e}-'
                     f'{trj_slices[0].dt}.csv')

    df[['vertical', 'horizontal', 'vertical_std', 'horizontal_std']] *= 100

    palette = sns.color_palette('Paired')

    fig, axs = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    width = 0.1

    for ax, exp in zip(axs, EXPERIMENTS):
        x = np.arange(len(EXPERIMENTS[exp]))
        positions = (
            (x - 4 * width, x - width, x + 2 * width),
            (x - 3 * width, x, x + 3 * width),
            (x - 2 * width, x + width, x + 4 * width),
        )
        alphas = [1, 0.5, 0.3]
        mhp_labels = ['phob', 'neutr', 'phil']
        labels = ['hydrophobic', 'neutral', 'hydrophilic']

        for pos, chl_amount, alpha in zip(positions, [10, 30, 50], alphas):
            data = df[
                (df['system'].isin(EXPERIMENTS[exp])) &
                (df['CHL amount, %'] == chl_amount)]

            for c, mhp in enumerate(mhp_labels):
                ax.bar(pos[c],
                       data[data['mhp_label'] == mhp]['vertical'],
                       yerr=data[data['mhp_label'] == mhp]['vertical_std'],
                       width=width,
                       label=f'{labels[c]}, vertical, {chl_amount}% CHL',
                       color=palette[c * 2 + 1], alpha=alpha)

                ax.bar(pos[c],
                       data[data['mhp_label'] == mhp]['horizontal'],
                       yerr=data[data['mhp_label'] == mhp]['horizontal_std'],
                       bottom=data[data['mhp_label'] == mhp]['vertical'],
                       width=width,
                       label=f'{labels[c]}, horizontal, {chl_amount}% CHL',
                       color=palette[c * 2], alpha=alpha)

                # single black edges independent on alpha
                ax.bar(pos[c],
                       data[data['mhp_label'] == mhp]['vertical'],
                       yerr=data[data['mhp_label'] == mhp]['vertical_std'],
                       width=width, ec='k', lw=2, fill=False)

                ax.bar(pos[c],
                       data[data['mhp_label'] == mhp]['horizontal'],
                       yerr=data[data['mhp_label'] == mhp]['horizontal_std'],
                       bottom=data[data['mhp_label'] == mhp]['vertical'],
                       width=width, ec='k', lw=2, fill=False)
        ax.set_title(exp)
        ax.set_ylim(0)
        ax.xaxis.set_ticks(x)
        ax.set_xticklabels(EXPERIMENTS[exp])

    axs[0].set_ylabel('% of area')
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    fig.savefig(
        PATH / 'notebooks' / 'mhpmaps' / 'imgs' /
        f'mhp_hists_area_chol_comps_dt{trj_slices[0].dt}.png',
        bbox_inches='tight', dpi=300)
    logging.info('done.')


def contacts_to_single_tables(trj_slices: list) -> None:
    '''
    collect contacts (hb and dc) of CHL with PL and SOL to single tables
    containing all systems
    '''
    list_of_prefixes = ['lip_hb_hb', 'lip_SOL_hb_hb',
                        'CHOL_SOL_dc_dc', 'lip_dc_dc']

    for pr in list_of_prefixes:
        fname = (
            PATH / 'notebooks' / 'contacts' /
            f'{pr}_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}'
            '_rchist_full.csv')
        # if fname.is_file():
        #     return
        logging.info('saving %s contacts data...', pr)

        dfs = []
        for trj in trj_slices:
            df = pd.read_csv(PATH / trj.system.name / 'contacts' /
                             f'{pr}_rchist_full.csv')
            df.insert(0, 'index', trj.system.name)
            df.insert(1, 'system', trj.system.name.split('_chol', 1)[0])
            df.insert(2, 'CHL amount, %', trj.system.name.split('_chol', 1)[1])
            dfs.append(df)

        df_all = pd.concat(dfs, ignore_index=True)
        df_all.to_csv(fname, index=False)
    logging.info('done.')


def create_chl_angle_z_mhp_table_for_contacts(trj_slices: list,
                                              pr: str) -> None:
    '''
    pr - type of contact:
    supported values -- ['lip_hb_hb', 'lip_SOL_hb_hb',
                        'CHOL_SOL_dc_dc', 'lip_dc_dc']
    '''
    fname = (PATH / 'notebooks' / 'contacts' /
             f'{pr}_chl_angle_z_mhp_'
             f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}'
             '_rchist_full.csv')

    # if fname.is_file():
    #     logging.info('%s already preprocessed, skipping...', pr)
    #     return

    logging.info('%s: preprocessing angle_mhp_z table...', pr)
    angle_mhp_z = pd.read_csv(
        PATH / 'notebooks' / 'integral_parameters' / 'angle_mhp_z_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')
    angle_mhp_z['distance to bilayer center, Å'] = pd.cut(
        angle_mhp_z['distance to bilayer center'],
        bins=[0, 3, 6, 9, 12, 15, 100],
        labels=['<= 3', '3-6', '6-9', '9-12', '12-15', '> 15'])
    angle_mhp_z['α, °'] = angle_mhp_z['α, °'].abs()

    logging.info('%s: loading data...', pr)
    df = pd.read_csv(
        PATH / 'notebooks' / 'contacts' /
        f'{pr}_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}'
        '_rchist_full.csv')
    df['chl_index'] = np.where(
        df['dmn'] == 'CHL', df['dmi'],
        np.where(df['amn'] == 'CHL', df['ami'], df['dmi']))
    df['other_index'] = np.where(
        df['dmn'] != 'CHL', df['dmi'],
        np.where(df['amn'] != 'CHL', df['ami'], df['ami']))
    df['other_name'] = np.where(
        df['dmn'] != 'CHL', df['dmn'],
        np.where(df['amn'] != 'CHL', df['amn'], df['amn']))

    df_dup = df.loc[df['other_name'] == 'CHL'].copy()
    (df_dup['other_index'], df_dup['chl_index']) = (
        df_dup['chl_index'], df_dup['other_index'])
    df = pd.concat([df, df_dup], ignore_index=True)

    df.dropna(inplace=True)

    logging.info('%s: updating CHL indices...', pr)
    with open(PATH / 'notebooks' / 'integral_parameters' /
              'chl_first_index.pkl', 'rb') as f:
        chl_first_index = pickle.load(f)
    df.set_index('index', inplace=True)
    df['chl_index'] = (df['chl_index']
                       - df['chl_index'].index.map(chl_first_index))
    df.reset_index(inplace=True)

    logging.info('%s: merging data...', pr)
    df_merged = df.merge(
        angle_mhp_z,
        on=['system', 'CHL amount, %', 'timepoint', 'chl_index'],
        how='left')

    logging.info('%s: saving data...', pr)
    df_merged.to_csv(fname, index=False)
    logging.info('%s done.', pr)


def merge_hb_or_dc_tables(trj_slices: list, hbonds: bool):
    '''
    create single dataframe from chl_angle_z_mhp with contacts
    (hbonds or dc), combining chl-lip and chl-sol
    '''
    if hbonds:
        pr1 = 'lip_hb_hb'
        pr2 = 'lip_SOL_hb_hb'
        postfix = 'hb'
    else:
        pr1 = 'lip_dc_dc'
        pr2 = 'CHOL_SOL_dc_dc'
        postfix = 'dc'

    fname = (PATH / 'notebooks' / 'contacts' /
             f'{postfix}_chl_angle_z_mhp_'
             f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')

    # if fname.is_file():
    #     logging.info('loading %s data...', postfix)
    #     return postfix, pd.read_csv(fname, low_memory=False)

    logging.info('preprocessing %s data...', postfix)

    df1 = pd.read_csv(
        PATH / 'notebooks' / 'contacts' /
        f'{pr1}_chl_angle_z_mhp_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}'
        '_rchist_full.csv')
    df2 = pd.read_csv(
        PATH / 'notebooks' / 'contacts' /
        f'{pr2}_chl_angle_z_mhp_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}'
        '_rchist_full.csv')

    df = pd.concat([df1, df2], ignore_index=True)
    df.loc[df['other_name'].str.endswith(('PC', 'PS')), 'other_name'] = 'PL'
    df['tilt component'] = df['1'].apply(
        lambda x: 'vertical' if x == 1 else ('horizontal' if x == 0 else x))
    df['surface'] = df['surface'].map({1: 'yes', 0: 'no'})
    df.to_csv(fname, index=False)
    return postfix, df


def plot_contacts_per_chl_mol_by_hue(trj_slices: list,
                                     df: DataFrame,
                                     n_chols_df: DataFrame,
                                     hue: str,
                                     postfix: str):
    '''
    plot figures of contacts between CHL and other molecles
    from df by experiments for single hue.

    df - from merge_hb_or_dc_tables
    n_chols_df - df grouped by time and system count of chl molecules
    hue should be in ['surface', 'tilt_component',
        'distance to bilayer center, Å']
    postfix -- hb or dc, used only in filename of plots
    '''
    # pylint: disable = too-many-locals
    logging.info('%s: calculating n contacts per molecule...', hue)
    table_with_probs = (df.groupby(
        ['system', 'CHL amount, %', 'timepoint', 'other_name', hue],
        as_index=False)['chl_index']
        .agg('count')
        .rename(columns={'chl_index': 'n_contacts'})
    )
    table_with_probs = table_with_probs.merge(
        n_chols_df,
        on=['system', 'CHL amount, %', 'timepoint'], how='left')

    table_with_probs['n contacts per molecule'] = (
        table_with_probs['n_contacts']
        / table_with_probs['n_chols'])

    order = (['<=3', '3-6', '6-9', '9-12', '12-15', '>15']
             if hue == 'distance to bilayer center, Å'
             else ['vertical', 'horizontal'] if hue == 'tilt component'
             else None)
    palette = 'crest_r' if hue == 'distance to bilayer center, Å' else 'muted'

    logging.info('%s: drawing plots...', hue)
    for exp, systs in EXPERIMENTS.items():
        logging.info('plotting %s...', exp)
        fig, axs = plt.subplots(3, 3, figsize=(20, 20),
                                sharex=True, sharey=True)
        axs = axs.flatten()

        c = 0
        for syst in systs:
            for chl_amount in [10, 30, 50]:
                data = table_with_probs[
                    (table_with_probs['system'] == syst) &
                    (table_with_probs['CHL amount, %'] == chl_amount)]
                sns.barplot(data=data,
                            x='other_name', y='n contacts per molecule',
                            order=['CHL', 'PL', 'SOL'],
                            hue=hue, hue_order=order, ax=axs[c],
                            edgecolor='k', palette=palette, ci='sd'
                            )
                if c != 7:
                    axs[c].legend([], [], frameon=False)
                if c not in [0, 3, 6]:
                    axs[c].set_ylabel('')
                else:
                    axs[c].set_ylabel('n contacts per CHL molecule')

                if c not in [6, 7, 8]:
                    axs[c].set_xlabel('')
                else:
                    axs[c].set_xlabel('contacting molecule')
                axs[c].xaxis.set_tick_params(which='both', labelbottom=True)

                axs[c].set_title(f'{syst}, {chl_amount} % CHL')
                c += 1

        sns.move_legend(axs[7], loc='upper center',
                        bbox_to_anchor=(0.5, -0.2), ncol=6)
        if postfix == 'hb':
            fig.suptitle(f'{exp}, hydrogen bonds')
        elif postfix == 'hb':
            fig.suptitle(f'{exp}, contacts by distance (6 Å)')
        fig.savefig(
            PATH / 'notebooks' / 'contacts' / 'imgs' /
            'contacts_per_chl_mol' /
            f'{postfix}_{hue}_'
            f'{"_".join(exp.split())}_'
            f'dt{trj_slices[0].dt}.png',
            bbox_inches='tight', dpi=300)
        plt.close()


def plot_chl_pl_hbonds_groups_by_hue(trj_slices: list,
                                     df: DataFrame,
                                     n_chols_df: DataFrame,
                                     hue: str,
                                     postfix: str):
    '''
    plot figures of contacts between CHL and PL groups
    from df by experiments for single hue.

    df - from merge_hb_or_dc_tables
    n_chols_df - df grouped by time and system count of chl molecules
    hue should be in ['surface', 'tilt_component',
        'distance to bilayer center, Å']
    postfix -- hb or dc, used only in filename of plots
    '''
    # pylint: disable = too-many-locals
    logging.info('%s: calculating n contacts per molecule (PL groups)...', hue)
    df = df[df['other_name'] == 'PL']
    table_with_probs = (df.groupby(
        ['system', 'CHL amount, %', 'timepoint', 'aan', hue],
        as_index=False)['chl_index']
        .agg('count')
        .rename(columns={'chl_index': 'n_contacts'})
    )

    # aan to PL group
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

    table_with_probs['PL group'] = table_with_probs['aan'].map(
        reverse_lookup_dict)

    table_with_probs = table_with_probs.merge(
        n_chols_df,
        on=['system', 'CHL amount, %', 'timepoint'], how='left')

    table_with_probs['n contacts per molecule'] = (
        table_with_probs['n_contacts']
        / table_with_probs['n_chols'])

    hue_order = (['<=3', '3-6', '6-9', '9-12', '12-15', '>15']
                 if hue == 'distance to bilayer center, Å'
                 else ['vertical', 'horizontal'] if hue == 'tilt component'
                 else None)
    palette = 'crest_r' if hue == 'distance to bilayer center, Å' else 'muted'

    logging.info('%s: drawing plots...', hue)

    for exp, systs in EXPERIMENTS.items():

        logging.info('plotting %s...', exp)
        fig, axs = plt.subplots(3, 3, figsize=(20, 20),
                                sharey=True)
        axs = axs.flatten()

        c = 0
        for syst in systs:
            order = (['PO4', "RC(O)R'", "ROR'", 'Ser'] if 'dops' in syst
                     else ['PO4', "RC(O)R'", "ROR'"])
            for chl_amount in [10, 30, 50]:
                data = table_with_probs[
                    (table_with_probs['system'] == syst) &
                    (table_with_probs['CHL amount, %'] == chl_amount)]

                sns.barplot(data=data,
                            x='PL group', y='n contacts per molecule',
                            order=order, hue=hue, hue_order=hue_order,
                            ax=axs[c],
                            edgecolor='k', palette=palette, ci='sd'
                            )
                if c != 7:
                    axs[c].legend([], [], frameon=False)
                if c not in [0, 3, 6]:
                    axs[c].set_ylabel('')
                else:
                    axs[c].set_ylabel('n contacts per CHL molecule')

                if c not in [6, 7, 8]:
                    axs[c].set_xlabel('')
                else:
                    axs[c].set_xlabel('contacting PL group')
                axs[c].xaxis.set_tick_params(which='both', labelbottom=True)

                axs[c].set_title(f'{syst}, {chl_amount} % CHL')
                c += 1

        sns.move_legend(axs[7], loc='upper center',
                        bbox_to_anchor=(0.5, -0.2), ncol=6)
        if postfix == 'hb':
            fig.suptitle(f'{exp}, hydrogen bonds')
        elif postfix == 'dc':
            fig.suptitle(f'{exp}, contacts by distance (6 Å)')

        fig.savefig(
            PATH / 'notebooks' / 'contacts' / 'imgs' /
            'contacts_per_chl_mol' /
            f'chl_pl_groups_{postfix}_{hue}_'
            f'{"_".join(exp.split())}_'
            f'dt{trj_slices[0].dt}.png',
            bbox_inches='tight', dpi=300)
        plt.close()


@app.command()
def plot_contacts(
        ctx: typer.Context):
    '''
    plot n contacts per CHL molecule
    - hbonds and dc
    - hue by surface, z-slices and angle components
    '''
    trj_slices, _, verbose, _ = ctx.obj
    initialize_logging('plot_contacts.log', verbose)
    sns.set(style='ticks', context='talk', palette='muted')
    chl_indexes = pd.read_csv(PATH / 'notebooks' / 'integral_parameters' /
                              f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-'
                              f'{trj_slices[0].dt}_coords_with_comps.csv',
                              usecols=[0, 4, 1, 2])
    n_chols_df = chl_indexes.groupby(
        ['system', 'CHL amount, %', 'timepoint'],
        as_index=False)['chl_index'].count().rename(
        columns={'chl_index': 'n_chols'}
    )
    for hbonds in [True, False]:
        postfix, df = merge_hb_or_dc_tables(trj_slices, hbonds)
        for hue in ['surface', 'tilt component',
                    'distance to bilayer center, Å']:
            plot_contacts_per_chl_mol_by_hue(trj_slices,
                                             df,
                                             n_chols_df,
                                             hue,
                                             postfix)
            if hbonds:
                plot_chl_pl_hbonds_groups_by_hue(trj_slices,
                                                 df,
                                                 n_chols_df,
                                                 hue,
                                                 postfix)
    logging.info('done.')


@app.command()
def mhp_components(
    ctx: typer.Context,
    chol_all: bool = typer.Option(
        True, help='plot % of area occupied by CHL on surface depending on '
        'CHL tilt component'),
    chol_mhp: bool = typer.Option(
        True, help='plot % of area occupied by different mhp fractions of CHL '
        'on surface depending on CHL tilt component')):
    '''
    plot CHL % of area on surface and % of different mhp fractions of CHL
    on surface depending on CHL tit component
    '''
    trj_slices, _, verbose, _ = ctx.obj
    initialize_logging('mhp_components.log', verbose)
    sns.set(style='ticks', context='talk', palette='muted')
    if chol_all:
        calculate_chol_surface_exposure_comps(trj_slices)
        plot_chol_surface_exposure_comps(trj_slices)
    if chol_mhp:
        calculate_chol_mhp_fractions_comps(trj_slices)
        plot_chol_mhp_fractions_comps(trj_slices)


@app.command()
def plot_angles(
        ctx: typer.Context,
        surface: bool = typer.Option(
            True, help='plot angles of CHL molecules on surface'),
        z_slices: bool = typer.Option(
            True, help='plot angles of CHL molecules on slices of z')):
    '''
    plot chl angles distributions of surface CHL molecules and by distance from
    bilayer center
    '''
    trj_slices, _, verbose, _ = ctx.obj
    initialize_logging('plot_angles.log', verbose)
    sns.set(style='ticks', context='talk', palette='muted')
    get_angle_mhp_z_chol_table(trj_slices)
    if z_slices:
        z_slices_angles(trj_slices)
    if surface:
        surface_angles(trj_slices)


@app.command()
def update_contacts_tables(ctx: typer.Context):
    '''
    combine rchist and rcnames files for all contacts
    '''
    trj_slices, n_workers, verbose, messages = ctx.obj
    initialize_logging('upd_ct.log', verbose)
    multiproc(update_contacts_tables_single_trj,
              trj_slices,
              show_progress='multiple',
              n_workers=n_workers,
              messages=messages,
              descr='updating contact tables')
    contacts_to_single_tables(trj_slices)
    list_of_prefixes = ['lip_hb_hb',
                        'lip_SOL_hb_hb',
                        'CHOL_SOL_dc_dc',
                        'lip_dc_dc']
    multiproc(create_chl_angle_z_mhp_table_for_contacts,
              [trj_slices] * len(list_of_prefixes),
              list_of_prefixes,
              n_workers=n_workers,
              messages=messages,
              descr='processing contact tables')


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
    '''
    set of utilities to determine how angles and contacts
    of CHL vary in different "zones" by z-axis
    '''
    # pylint: disable = too-many-arguments
    systems = flatten([(i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in flatten(EXPERIMENTS.values())])
    systems = list(dict.fromkeys(systems))
    trj_slices = [TrajectorySlice(System(
        PATH, s, trj, tpr), b, e, dt) for s in systems]
    ctx.obj = (trj_slices, n_workers, verbose, messages)


# %%
if __name__ == '__main__':
    app()


# %%
# systems = flatten([(i + '_chol10', i + '_chol30', i + '_chol50')
#                    for i in flatten(EXPERIMENTS.values())])
# systems = list(dict.fromkeys(systems))
# trj_slices = [TrajectorySlice(System(
#     PATH, s, 'pbcmol_201.xtc', '201_ns.tpr'),
#     200.0, 201.0, 1) for s in systems]
#
# trj = trj_slices[10]

# %% RDF
# systs = ['popc_chol10', 'popc_chol30', 'popc_chol50']
# syst=systs[0]
#
# fig, axs = plt.subplots(1, 3, figsize=(20, 7),
#                         sharex=True, sharey=True)
# for ax, syst in zip(axs, systs):
#     df_n = pd.read_csv(PATH / syst / 'a' / 'N_rdf.csv')
#     df_p = pd.read_csv(PATH / syst / 'a' / 'P_rdf.csv')
#     ax.plot(df_n['# r'], df_n['g'], label='N')
#     ax.plot(df_p['# r'], df_p['g'], label='P')
#     ax.set_xlabel('distance to CHL O, Å')
#     sname, chl = syst.split('_chol', 1)
#     ax.set_title(f'{sname}, {chl}% CHL')
# axs[0].set_ylabel('g(r)')
# axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
#
#
# # %%
# fig.savefig(PATH / 'notebooks' / 'rdf' / 'popc_P_N_chl_O.png',
# bbox_inches='tight', dpi=300)


# # %%
# df = pd.read_csv(PATH / 'notebooks' / 'integral_parameters' /
#                  'angle_mhp_z_'
#                  f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')
#
#
# # %%
# df[df['system'] == 'dops'].groupby(
#     ['system', 'CHL amount, %'], as_index=False)['timepoint'].nunique()
