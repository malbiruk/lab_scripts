'''
Script to perform lateral clusterization in each monolayer and
plot cluster sizes in each experiment depending on CHL amount
'''

import argparse
import sys

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import scikit_posthocs
import seaborn as sns
from chl_angle_components_density import generate_coords_comps_table
from MDAnalysis.analysis import leaflet
from modules.constants import EXPERIMENTS, PATH
from modules.general import (duration, flatten, print_1line, progress_bar,
                             sparkles)
from modules.traj import System, TrajectorySlice
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering


def obtain_pl_coords(trj_slices: list) -> None:
    '''
    obtain coords of P of phospholipids in all systems listed in trj_slices
    also assigns P coords to monolayers
    '''

    records = []
    print('collecting coords...')

    for trj in trj_slices:
        print(trj)

        trj.generate_slice_with_gmx()
        u = mda.Universe(f'{trj.system.dir}/md/md.tpr',
                         f'{trj.system.dir}/md/pbcmol_{trj.b}-{trj.e}-{trj.dt}.xtc')

        for ts in u.trajectory:
            cutoff, n = leaflet.optimize_cutoff(
                u, 'name P* or name O3', dmin=7, dmax=17)
            print_1line(f'cutoff {cutoff} A, {n} groups')
            leaflet_ = leaflet.LeafletFinder(
                u, 'name P* or name O3', pbc=True, cutoff=cutoff)
            if len(leaflet_.groups()) != 2:
                print(f'{len(leaflet_.groups())} groups found...')
            leaflet_0 = leaflet_.group(0)
            leaflet_1 = leaflet_.group(1)
            zmem = 0.5 * (leaflet_1.centroid() + leaflet_0.centroid())[2]
            phosphors = u.select_atoms("name P*")
            for i in phosphors:
                records.append((
                    trj.system.name.split(
                        '_chol', 1)[0] if '_chol' in trj.system.name else trj.system.name,
                    int(trj.system.name.split(
                        '_chol', 1)[1]) if '_chol' in trj.system.name else 0,
                    ts.time,
                    i.resname,
                    i.position[0],
                    i.position[1],
                    i.position[2],
                    'upper' if i in leaflet_0 else (
                        'lower' if i in leaflet_1 else 'na'),
                    zmem,
                    ts.dimensions[0],
                    ts.dimensions[1],
                    ts.dimensions[2]))

    df = pd.DataFrame.from_records(records, columns=[
        'system',
        'CHL amount, %',
        'timepoint',
        'lipid',
        'x_p',
        'y_p',
        'z_p',
        'monolayer',
        'zmem',
        'x_box',
        'y_box',
        'z_box'
    ])

    df.to_csv(PATH / 'notebooks' / 'gclust' /
              f'PL_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}_coords.csv')


def perform_pbc_clusterization(df: pd.DataFrame, x='x_o', y='y_o',
                               box='x_box', thresh: int = 6) -> pd.DataFrame:
    '''
    Single linkage geometric clusterization by distance threshold taking into account
    periodic boundary conditions.
    X - array of data (coords of points which should be clusterized)
    L - box size (for pbc)
    thresh - distance threshold
    '''
    xy = np.hstack((df[x].values[:, None], df[y].values[:, None]))
    box_side = df.iloc[0][box]

    for d in range(xy.shape[1] - 1):
        # find all 1-d distances
        pd_ = pdist(xy[:, d].reshape(xy.shape[0], 1))
        # apply boundary conditions
        pd_[pd_ > box_side * 0.5] -= box_side
        try:
            # sum
            total += pd_**2
        except NameError:
            # or define the sum if not previously defined
            total = pd_**2
    # transform the condensed distance matrix...
    total = np.sqrt(total)
    # ...into a square distance matrix
    square = squareform(total)
    ac = AgglomerativeClustering(n_clusters=None, linkage='single',
                                 distance_threshold=thresh, affinity='precomputed').fit(square)
    df['label'] = ac.labels_
    return df


def obtain_cluster_labels(trj_slices: list, mol: str = 'CHL',
                          thresh: int = 6, x: str = 'x_o', y: str = 'y_o',
                          b_comp=100, e_comp=200, dt_comp=100) -> None:
    '''
    perform clusterization for each point of _coords_with_comps.csv df columns
    'timepoint', 'system', 'CHL amount, %', 'monolayer' should be presented
    trj_slices -- list of TrajectorySlice() objects
    mol -- molecule, which clusters are analyzed, should be CHL or PL
    thresh -- threshold for clusterization (A)
    x, y - columns of coords for clusterization
    '''

    if mol == 'CHL':
        if not (PATH / 'notebooks' / 'integral_parameters' /
                f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-'
                f'{trj_slices[0].dt}_coords_with_comps.csv').is_file():
            generate_coords_comps_table(trj_slices, b_comp, e_comp, dt_comp)

        df = pd.read_csv(PATH / 'notebooks' / 'integral_parameters' /
                         f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-'
                         f'{trj_slices[0].dt}_coords_with_comps.csv')

    elif mol == 'PL':
        if not (PATH / 'notebooks' / 'gclust' /
                f'PL_{trj_slices[0].b}-{trj_slices[0].e}-'
                f'{trj_slices[0].dt}_coords.csv').is_file():
            obtain_pl_coords(trj_slices)

        df = pd.read_csv(PATH / 'notebooks' / 'gclust' /
                         f'PL_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}_coords.csv')

    else:
        raise ValueError(f'--molecule value "{mol}" is not recognized. '
                         'Acceptable options are "CHL" or "PL".')

    print('running clusterization...')
    all_counts = df.groupby(
        ['system', 'CHL amount, %', 'timepoint', 'monolayer'],
        as_index=False, group_keys=False).apply(
        perform_pbc_clusterization, x=x, y=y, thresh=thresh)
    print('saving results...')
    all_counts.to_csv(PATH / 'notebooks' / 'gclust' /
                             f'{mol}_clusters_{trj_slices[0].b}-{trj_slices[0].e}-'
                             f'{trj_slices[0].dt}_thresh_{thresh}.csv', index=False)
    print('done.')


def plot_cluster_sizes(trj_slices: list, mol: str, thresh=3, max_clsize=70) -> None:
    '''
    trj_slices -- list of TrajectorySlice() objects
    mol -- molecule, which clusters are analyzed, should be CHL or PL
    thresh -- threshold for clusterization (A)
    max_clsize -- maximum cluster sizes to show
    '''
    all_counts = pd.read_csv(PATH / 'notebooks' / 'gclust' /
                             f'{mol}_clusters_{trj_slices[0].b}-{trj_slices[0].e}-'
                             f'{trj_slices[0].dt}_thresh_{thresh}.csv')
    print('counting cluster sizes...')
    df2 = all_counts.groupby(
        ['timepoint', 'system', 'CHL amount, %', 'monolayer']).agg(
        cluster_size=('label', 'value_counts')).reset_index()
    palette = sns.color_palette('RdYlGn_r', 4)
    print('plotting...')
    for exp, _ in EXPERIMENTS.items():
        fig, axs = plt.subplots(1, 3, figsize=(
            24, 7), sharex=True, sharey=True)
        for syst, ax in zip(EXPERIMENTS[exp], axs):
            hists = {}
            for chol_amount in df2['CHL amount, %'].unique():
                hists[chol_amount] = np.histogram(
                    df2[(df2['system'] == syst) &
                        (df2['CHL amount, %'] == chol_amount)]['cluster_size'],
                    bins=np.arange(1, max_clsize, 4), density=True)
            width = 1
            ax.bar(hists[0][1][:-1] - width, hists[0][0], color=palette[0],
                   width=width, ec='k', label='0% of CHL')
            ax.bar(hists[10][1][:-1], hists[10][0], color=palette[1],
                   width=width, ec='k', label='10% of CHL')
            ax.bar(hists[30][1][:-1] + width, hists[30][0], color=palette[2],
                   width=width, ec='k', label='30% of CHL')
            ax.bar(hists[50][1][:-1] + 2 * width, hists[50][0], color=palette[3],
                   width=width, ec='k', label='50% of CHL')
            # ax.set_xlim(left=0)
            ax.set_xlabel('Cluster size (n of molecules)')
            ax.set_title(syst)
            # ax.set_yscale('log')
        axs[0].set_ylabel('Density')
        axs[1].legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -.32))
        fig.suptitle(f'{exp}, threshold={thresh} Å')
        fig.savefig(PATH / 'notebooks' / 'gclust' /
                    str(f'{mol}_clusters_{trj_slices[0].b}-{trj_slices[0].e}-'
                        f'{trj_slices[0].dt}_thresh_{thresh}'
                        + '_'.join(exp.split()) + '.png'),
                    bbox_inches='tight')
    print('done.')


def plot_cluster_sizes_with_components(trj_slices: list, mol: str, thresh=6, max_clsize=40) -> None:
    '''
    trj_slices -- list of TrajectorySlice() objects
    mol -- molecule, which clusters are analyzed, should be CHL or PL
    thresh -- threshold for clusterization (A)
    max_clsize -- maximum cluster sizes to show
    '''
    all_counts = pd.read_csv(PATH / 'notebooks' / 'gclust' /
                             f'{mol}_clusters_{trj_slices[0].b}-{trj_slices[0].e}-'
                             f'{trj_slices[0].dt}_thresh_{thresh}.csv')

    print('counting cluster sizes...')

    all_counts['component'] = all_counts.apply(
        lambda x: 'vertical'
        if x['1'] == 1
        else ('horizontal' if x['2'] == 1 else np.nan), axis=1)

    df2 = all_counts.groupby(
        ['timepoint', 'system', 'CHL amount, %', 'monolayer', 'component']).agg(
        cluster_size=('label', 'value_counts')).reset_index()

    palette = sns.color_palette('Paired')

    print('plotting...')
    for exp, _ in EXPERIMENTS.items():
        fig, axs = plt.subplots(1, 3, figsize=(
            24, 7), sharex=True, sharey=True)
        for syst, ax in zip(EXPERIMENTS[exp], axs):
            hists = {}
            for chol_amount in df2['CHL amount, %'].unique():
                for comp in df2['component'].unique():
                    hists[(chol_amount, comp)] = np.histogram(
                        df2[(df2['system'] == syst) &
                            (df2['CHL amount, %'] == chol_amount) &
                            (df2['component'] == comp)]['cluster_size'],
                        bins=np.arange(
                            1, max_clsize, 3),
                        density=True)

            width = 1

            ax.bar(hists[(10, 'vertical')][1][:-1] - width, hists[(10, 'vertical')][0],
                   width,
                   ec='k',
                   color=palette[1], label='vertical component, 10% of CHL')
            ax.bar(hists[(10, 'horizontal')][1][:-1] - width, hists[(10, 'horizontal')][0],
                   width,
                   ec='k',
                   color=palette[0], label='horizontal component, 10% of CHL',
                   bottom=hists[(10, 'vertical')][0])
            ax.bar(hists[(30, 'vertical')][1][:-1], hists[(30, 'vertical')][0],
                   width,
                   ec='k',
                   color=palette[3], label='vertical component, 30% of CHL')
            ax.bar(hists[(30, 'horizontal')][1][:-1], hists[(30, 'horizontal')][0],
                   width,
                   ec='k',
                   color=palette[2], label='horizontal component, 30% of CHL',
                   bottom=hists[(30, 'vertical')][0])
            ax.bar(hists[(50, 'vertical')][1][:-1] + width, hists[(50, 'vertical')][0],
                   width,
                   ec='k',
                   color=palette[5], label='vertical component, 50% of CHL')
            ax.bar(hists[(50, 'horizontal')][1][:-1] + width, hists[(50, 'horizontal')][0],
                   width,
                   ec='k',
                   color=palette[4], label='horizontal component, 50% of CHL',
                   bottom=hists[(50, 'vertical')][0])
            # ax.set_xlim(left=0)
            ax.set_xlabel('Cluster size (n of molecules)')
            ax.set_title(syst)
            # ax.set_yscale('log')
        axs[0].set_ylabel('Density')
        axs[1].legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -.32))
        fig.suptitle(f'{exp}, threshold={thresh} Å')
        fig.savefig(PATH / 'notebooks' / 'gclust' /
                    str(f'{mol}_clusters_{trj_slices[0].b}-{trj_slices[0].e}-'
                        f'{trj_slices[0].dt}_thresh_{thresh}'
                        + '_'.join(exp.split()) + '.png'),
                    bbox_inches='tight')
    print('done.')


def upd_dict_with_stat_tests_res_by_chl(p, df2: pd.DataFrame, df_by_chl: dict,
                                        mol: str, thresh: int) -> None:
    '''
    helper function to perform stat tests on clsizes different by chl amount
    stats: kruskal _ posthoc Dunn with Holm p-adjust
    '''
    for syst in p.track(df2['system'].unique(), description='tests by chl amount'):
        data = [df2[(df2['system'] == syst) & (df2['CHL amount, %'] == 10)]['cluster_size'],
                df2[(df2['system'] == syst) & (df2['CHL amount, %'] == 30)]['cluster_size'],
                df2[(df2['system'] == syst) & (df2['CHL amount, %'] == 50)]['cluster_size']]

        kruskal_p = stats.kruskal(*data).pvalue
        if kruskal_p < 0.01:
            p_values = scikit_posthocs.posthoc_dunn(data, p_adjust='holm')
        else:
            p_values = pd.DataFrame({1: [1, kruskal_p, kruskal_p],
                                     2: [kruskal_p, 1, kruskal_p],
                                     3: [kruskal_p, kruskal_p, 1]})
            p_values.index += 1

        df_by_chl['lipid'].append(mol)
        df_by_chl['threshold'].append(thresh)
        df_by_chl['system'].append(syst)
        df_by_chl['10 vs 30 % CHL'].append(p_values.loc[1, 2])
        df_by_chl['30 vs 50 % CHL'].append(p_values.loc[2, 3])
        df_by_chl['10 vs 50 % CHL'].append(p_values.loc[1, 3])


def upd_dict_with_stat_tests_res_by_exp(p, df2: pd.DataFrame, df_by_exp: dict,
                                        mol: str, thresh: int) -> None:
    '''
    helper function to perform stat tests on clsizes different by experiment
    stats: kruskal _ posthoc Dunn with Holm p-adjust
    '''
    for exp in p.track(EXPERIMENTS, description='tests by experiment'):
        for chl_amount in (0, 10, 30, 50):
            systs = EXPERIMENTS[exp]
            data = [df2[(df2['system'] == systs[0]) & (
                df2['CHL amount, %'] == chl_amount)]['cluster_size'],
                df2[(df2['system'] == systs[1]) & (
                    df2['CHL amount, %'] == chl_amount)]['cluster_size'],
                df2[(df2['system'] == systs[2]) & (
                    df2['CHL amount, %'] == chl_amount)]['cluster_size']]

            kruskal_p = stats.kruskal(*data).pvalue
            if kruskal_p < 0.01:
                p_values = scikit_posthocs.posthoc_dunn(data, p_adjust='holm')
            else:
                p_values = pd.DataFrame({1: [1, kruskal_p, kruskal_p],
                                         2: [kruskal_p, 1, kruskal_p],
                                         3: [kruskal_p, kruskal_p, 1]})
                p_values.index += 1

            df_by_exp['lipid'].append(mol)
            df_by_exp['threshold'].append(thresh)
            df_by_exp['CHL amount, %'].append(chl_amount)
            df_by_exp['experiment'].append(exp)
            df_by_exp['1 vs 2'].append(p_values.loc[1, 2])
            df_by_exp['2 vs 3'].append(p_values.loc[2, 3])
            df_by_exp['1 vs 3'].append(p_values.loc[1, 3])
    p.remove_task(p.tasks[-2].id)
    p.remove_task(p.tasks[-1].id)


def perform_stat_tests_by_chl_by_exp(trj_slices: list) -> None:
    '''
    perform Kruskal-Wallis test followed by post-hoc Dunn test with Holm p-adjust
    on cluster sizes of CHL and PL for thresholds 3, 5, 7

    for all systems: are cluster sizes different with changing CHL amount?
    for each experiment: are cluster sizes different depending on experiment
    (chain length, saturation etc.)?
    '''
    df_by_chl = {
        'lipid': [],
        'threshold': [],
        'system': [],
        '10 vs 30 % CHL': [],
        '30 vs 50 % CHL': [],
        '10 vs 50 % CHL': [],
    }

    df_by_exp = {
        'lipid': [],
        'threshold': [],
        'CHL amount, %': [],
        'experiment': [],
        '1 vs 2': [],
        '2 vs 3': [],
        '1 vs 3': [],
    }

    with progress_bar as p:
        for mol in p.track(('CHL', 'PL'), description='mol'):
            for thresh in p.track((3, 5, 7), description='threshold'):
                all_counts = pd.read_csv(PATH / 'notebooks' / 'gclust' /
                                         f'{mol}_clusters_{trj_slices[0].b}-{trj_slices[0].e}-'
                                         f'{trj_slices[0].dt}_thresh_{thresh}.csv')
                if mol == 'CHL':
                    all_counts['component'] = all_counts.apply(
                        lambda x: 'vertical'
                        if x['1'] == 1
                        else ('horizontal' if x['2'] == 1 else np.nan), axis=1)
                    df2 = all_counts.groupby(
                        ['timepoint', 'system', 'CHL amount, %', 'monolayer', 'component']).agg(
                        cluster_size=('label', 'value_counts')).reset_index()
                else:
                    df2 = all_counts.groupby(
                        ['timepoint', 'system', 'CHL amount, %', 'monolayer']).agg(
                        cluster_size=('label', 'value_counts')).reset_index()

                upd_dict_with_stat_tests_res_by_chl(p, df2, df_by_chl, mol, thresh)
                upd_dict_with_stat_tests_res_by_exp(p, df2, df_by_exp, mol, thresh)
            p.remove_task(p.tasks[-1].id)

    by_chl = pd.DataFrame(df_by_chl)
    by_exp = pd.DataFrame(df_by_exp)

    by_chl.to_csv(PATH / 'notebooks' / 'gclust' / 'stats' /
                  f'cl_sizes_by_chl_amount_{trj_slices[0].b}-{trj_slices[0].e}-'
                  f'{trj_slices[0].dt}.csv', index=False)
    by_exp.to_csv(PATH / 'notebooks' / 'gclust' / 'stats' /
                  f'cl_sizes_by_exp_{trj_slices[0].b}-{trj_slices[0].e}-'
                  f'{trj_slices[0].dt}.csv', index=False)


def perform_stat_tests_by_comp(trj_slices: list) -> None:
    '''
    for each threshold (3, 5, 7)
    for each system and each chl amount (10, 30, 50)
    perform Mann-Whitneyu U-test comapring cluster sizes of CHL consisting of
    vertical or horizontal components (of CHL tilt angle)
    '''

    mw_comps_dict = {
        'threshold': [],
        'system': [],
        'CHL amount, %': [],
        'statistic': [],
        'p-value': []
    }
    mol = 'CHL'

    with progress_bar as p:
        for thresh in p.track((3, 5, 7), description='threshold'):
            all_counts = pd.read_csv(PATH / 'notebooks' / 'gclust' /
                                     f'{mol}_clusters_{trj_slices[0].b}-{trj_slices[0].e}-'
                                     f'{trj_slices[0].dt}_thresh_{thresh}.csv')
            all_counts['component'] = all_counts.apply(
                lambda x: 'vertical'
                if x['1'] == 1
                else ('horizontal' if x['2'] == 1 else np.nan), axis=1)
            df2 = all_counts.groupby(
                ['timepoint', 'system', 'CHL amount, %', 'monolayer', 'component']).agg(
                cluster_size=('label', 'value_counts')).reset_index()
            for syst in p.track(df2['system'].unique(), description='performing MW test'):
                for chl_amount in (10, 30, 50):
                    hor = df2[(df2['system'] == syst) & (df2['CHL amount, %'] == chl_amount) &
                              (df2['component'] == 'horizontal')]['cluster_size']
                    ver = df2[(df2['system'] == syst) & (df2['CHL amount, %'] == chl_amount) &
                              (df2['component'] == 'vertical')]['cluster_size']

                    stat, pval = stats.mannwhitneyu(hor, ver)
                    mw_comps_dict['threshold'].append(thresh)
                    mw_comps_dict['system'].append(syst)
                    mw_comps_dict['CHL amount, %'].append(chl_amount)
                    mw_comps_dict['statistic'].append(stat)
                    mw_comps_dict['p-value'].append(pval)
            p.remove_task(p.tasks[-1].id)

    mv_comps = pd.DataFrame(mw_comps_dict)
    mv_comps.to_csv(PATH / 'notebooks' / 'gclust' / 'stats' /
                    f'cl_sizes_by_comp_{trj_slices[0].b}-{trj_slices[0].e}-'
                    f'{trj_slices[0].dt}.csv', index=False)


def get_clsizes_mean_std(trj_slices: list, components: bool = False) -> None:
    '''
    get mean cluster size and std of cluster sizes for each system
    '''
    if components:
        df = pd.DataFrame({'lipid': [], 'threshold': [], 'system': [],
                           'CHL amount, %': [], 'component': [], 'mean_clsize': [],
                           'std_clsize': []})
    else:
        df = pd.DataFrame({'lipid': [], 'threshold': [], 'system': [],
                           'CHL amount, %': [], 'mean_clsize': [], 'std_clsize': []})

    mols = ('CHL', 'PL') if not components else ('CHL',)

    with progress_bar as p:
        for mol in p.track(mols, description='lipid'):
            for thresh in p.track((3, 5, 7), description='threshold'):
                all_counts = pd.read_csv(PATH / 'notebooks' / 'gclust' /
                                         f'{mol}_clusters_{trj_slices[0].b}-{trj_slices[0].e}-'
                                         f'{trj_slices[0].dt}_thresh_{thresh}.csv')
                if mol == 'CHL' or components:
                    all_counts['component'] = all_counts.apply(
                        lambda x: 'vertical'
                        if x['1'] == 1
                        else ('horizontal' if x['2'] == 1 else np.nan), axis=1)
                    df2 = all_counts.groupby(
                        ['timepoint', 'system', 'CHL amount, %', 'monolayer', 'component']).agg(
                        cluster_size=('label', 'value_counts')).reset_index()
                else:
                    df2 = all_counts.groupby(
                        ['timepoint', 'system', 'CHL amount, %', 'monolayer']).agg(
                        cluster_size=('label', 'value_counts')).reset_index()

                groupby = (['system', 'CHL amount, %', 'component']
                           if components else ['system', 'CHL amount, %'])
                mean_std_df = df2.groupby(groupby).agg(
                    mean_clsize=('cluster_size', 'mean'),
                    std_clsize=('cluster_size', 'std')
                ).reset_index()

                mean_std_df.insert(0, 'threshold', [thresh for _ in range(len(mean_std_df))])
                mean_std_df.insert(0, 'lipid', [mol for _ in range(len(mean_std_df))])
                df = pd.concat((df, mean_std_df))
            p.remove_task(p.tasks[-1].id)

    if components:
        df.to_csv(PATH / 'notebooks' / 'gclust' / 'stats' /
                  f'cl_sizes_components_mean_std_{trj_slices[0].b}-{trj_slices[0].e}-'
                  f'{trj_slices[0].dt}.csv', index=False)
    else:
        df.to_csv(PATH / 'notebooks' / 'gclust' / 'stats' /
                  f'cl_sizes_mean_std_{trj_slices[0].b}-{trj_slices[0].e}-'
                  f'{trj_slices[0].dt}.csv', index=False)


@ sparkles
@ duration
def main():
    '''
    parse arguments and execute all or some of the functions:
    chl_tilt_angle, angle_components_density, angle_components_3d
    '''
    parser = argparse.ArgumentParser(
        description='Script to perform lateral clusterization in each monolayer and '
        'plot cluster sizes in each experiment depending on CHL amount.')
    parser.add_argument('--obtain_cluster_labels',
                        action='store_true',
                        help='obtain df with molecule coords and labels')
    parser.add_argument('--plot_cluster_sizes_with_components',
                        action='store_true',
                        help='plot sizes of clusters with lateral components')
    parser.add_argument('--plot_cluster_sizes',
                        action='store_true',
                        help='plot sizes of clusters')
    parser.add_argument('--stats',
                        action='store_true',
                        help='dump tables with stat tests')
    parser.add_argument('-mol', '--molecule', type=str, default='CHL',
                        help='molecules to clusterize, may be "CHL" or "PL", default="CHL"')
    parser.add_argument('-cls', '--max_clsize', type=int, default=40,
                        help='maximum cluster size')
    parser.add_argument('-x', '--x', type=str, default='x_o',
                        help='name of x coordinate')
    parser.add_argument('-y', '--y', type=str, default='y_o',
                        help='name of y coordinate')
    parser.add_argument('-t', '--thresh', type=int, default=6,
                        help='threshold of clusterization (A), default=6')
    parser.add_argument('-b', '--b', type=int, default=199,
                        help='beginning time in ns, default=199')
    parser.add_argument('-e', '--e', type=int, default=200,
                        help='ending time in ns, default=200')
    parser.add_argument('-dt', '--dt', type=int, default=10,
                        help='dt in ps (default=10)')
    parser.add_argument('--chl_tilt_b_e_dt',
                        nargs='+',
                        default='100 200 100',
                        help='b e dt (3 numbers) for calculation of tilt components, '
                        'chl_tilt_angle will be calculated with this values. '
                        '(100 200 100 by default)')
    if len(sys.argv) < 2:
        parser.print_usage()
    args = parser.parse_args()

    sns.set(style='ticks', context='talk', palette='muted')

    systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in flatten(EXPERIMENTS.values())])
    trj_slices = [TrajectorySlice(
        System(PATH, s), args.b, args.e, args.dt) for s in systems]

    b_comp, e_comp, dt_comp = [
        int(i) for i in args.chl_tilt_b_e_dt.split()]

    if args.molecule == 'CHL':
        trj_slices = [s for s in trj_slices if 'chol' in s.system.name]

    if args.obtain_cluster_labels:
        obtain_cluster_labels(trj_slices, args.molecule,
                              args.thresh, args.x, args.y,
                              b_comp, e_comp, dt_comp)

    if args.plot_cluster_sizes_with_components:
        plot_cluster_sizes_with_components(
            trj_slices, args.molecule, args.thresh, args.max_clsize)

    if args.plot_cluster_sizes:
        plot_cluster_sizes(trj_slices, args.molecule,
                           args.thresh, args.max_clsize)

    if args.stats:
        # perform_stat_tests_by_chl_by_exp(trj_slices)
        # perform_stat_tests_by_comp(trj_slices)
        get_clsizes_mean_std(trj_slices)
        get_clsizes_mean_std(trj_slices, components=True)


# %%
if __name__ == '__main__':
    main()
