'''
Script to obtain chl angle components and draw CHL angle component+density plots, as well as
CHL angle component plots in 2d and 3d.
CHL ids and angle components are connected here.
'''

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
from integral_parameters_script import plot_violins
from MDAnalysis.analysis import leaflet
from modules import ndtest
from modules.angles import chl_tilt_angle
from modules.constants import EXPERIMENTS, PATH
from modules.density import plot_density_profile
from modules.general import (chunker, duration, flatten, get_keys_by_value,
                             opener, print_1line, sparkles)
from modules.traj import System, TrajectorySlice


def plot_comps_angle_vals(csv: Path, b_comp: int, e_comp: int, dt_comp: int):
    '''
    plot components angle values
    '''
    df = pd.read_csv(csv)
    df['α, °'] = df['α, °'].abs()

    df['component'] = df['1'].apply(
        lambda x: 'vertical' if x == 1 else ('horizontal' if x == 0 else x))

    df['experiment'] = df['system'].apply(
        lambda x: get_keys_by_value(x, EXPERIMENTS))
    df = df.explode('experiment')
    order = {'dmpc': 0, 'dppc_325': 1, 'dspc': 2, 'popc': 3,
             'dopc': 4, 'dopc_dops50': 5, 'dops': 6}
    df = df.sort_values(by=['system'], key=lambda x: x.map(order))
    df = df.sort_values(
        by=['CHL amount, %', 'component'], ascending=[True, False])

    fig, axs = plt.subplots(1, 3, figsize=(23, 7), sharey=True)
    for exp, ax in zip(EXPERIMENTS, axs):
        df1 = df[df['experiment'] == exp]
        hue = df1['CHL amount, %'].astype(
            str) + ', ' + df1['component'].astype(str)
        g = sns.violinplot(data=df1, x='system', y='α, °', hue=hue,
                           cut=0, palette='RdYlGn_r', inner='quartile', ax=ax)
        ax.set_title(exp)
        if ax != axs[-1]:
            g.legend_.remove()
        if ax != axs[0]:
            ax.set_ylabel('')
    fig.savefig(PATH / 'notebooks' / 'integral_parameters' / 'components' /
                f'angle_components_{b_comp}-{e_comp}-{dt_comp}.png',
                bbox_inches='tight')


def plot_comps_ratios(b_comp: int, e_comp: int, dt_comp: int):
    '''
    plot ratios of components, also save means and stds
    '''
    df = pd.read_csv(PATH / 'notebooks' / 'integral_parameters' /
                     f'chl_tilt_{b_comp}-{e_comp}-{dt_comp}_with_comps.csv'
                     )
    df1 = df.groupby(['system', 'CHL amount, %', 'timepoint'],
                     as_index=False).agg(
        sum1=('1', 'sum'), count1=('1', 'count'),
        sum2=('2', 'sum'), count2=('2', 'count'))
    df1['% of vertical component'] = df1['sum1'] / df1['count1'] * 100
    df1['% of horizontal component'] = df1['sum2'] / df1['count2'] * 100
    df1.to_csv(PATH / 'notebooks' / 'integral_parameters' / 'components' /
               f'angle_components_ratio_{b_comp}-{e_comp}-{dt_comp}.csv', index=False)
    plot_violins(
        PATH / 'notebooks' / 'integral_parameters' / 'components' /
        f'angle_components_ratio_{b_comp}-{e_comp}-{dt_comp}.csv', '% of horizontal component')
    plot_violins(
        PATH / 'notebooks' / 'integral_parameters' / 'components' /
        f'angle_components_ratio_{b_comp}-{e_comp}-{dt_comp}.csv', '% of vertical component')
    df2 = df1.groupby(
        ['system', 'CHL amount, %'], as_index=False).agg(
        vertical_mean=('% of vertical component', 'mean'),
        vertical_std=('% of vertical component', 'std'),
        horizontal_mean=('% of horizontal component', 'mean'),
        horizontal_std=('% of horizontal component', 'std')
    )
    df2.to_csv(PATH / 'notebooks' / 'integral_parameters' / 'components' /
               f'angle_components_ratio_{b_comp}-{e_comp}-{dt_comp}_mean_std.csv', index=False)


def add_comps_to_chl_tilt(path_to_df, b, e, dt,
                          b_comp=None, e_comp=None, dt_comp=None,
                          plot_comps_ratio=False):
    '''
    add info which component is every angle in chl_tilt_{b}-{e}-{dt}.to_csv
    (file produced by chl_tilt_angle() function)
    df should have columns 'system', 'α, °', 'CHL amount, %'
    '''
    df = pd.read_csv(path_to_df)
    print('determining angle components...')
    if (b_comp, e_comp, dt_comp) == (None, None, None):
        b_comp, e_comp, dt_comp = b, e, dt
    df['comp'] = df.apply(
        lambda row: angle_refers_to_component(
            row['system'], row['α, °'], row['CHL amount, %'],
            b_comp, e_comp, dt_comp), axis=1)
    df = df.explode(['comp'])
    df['comp'] = df.comp.astype(str)
    df = df.join(pd.get_dummies(df['comp']))
    df = df.drop(columns=['comp']).drop_duplicates()
    df.to_csv(str(path_to_df).rsplit('.', 1)[0] + '_with_comps.csv',
              index=False)

    if plot_comps_ratio:
        print('plotting components angle values...')
        plot_comps_angle_vals(PATH / 'notebooks' / 'integral_parameters' /
                              f'chl_tilt_{b_comp}-{e_comp}-{dt_comp}_with_comps.csv',
                              b_comp, e_comp, dt_comp)
        print('poltting components ratio...')
        plot_comps_ratios(b_comp, e_comp, dt_comp)
    print('done.')


def obtain_df_of_density(trj_slices: list[TrajectorySlice]) -> pd.DataFrame:
    '''
    summarize data from different files of density to one df
    '''
    records = []
    for trj in trj_slices:
        for fr in range(trj_slices[0].b, trj_slices[0].e, int(trj_slices[0].dt / 1000)):
            lines = opener(Path(trj.system.dir) / 'density_profiles' /
                           f'chols_{fr}-{fr}-0_dp.xvg')
            z = np.array(
                list(map(float, [i.split()[0] for i in lines])))
            a = np.array(
                list(map(float, flatten([i.split()[1:] for i in lines]))))
            n_chols = int(a.shape[0] / z.shape[0])
            z_col = [i for i in z for _ in range(n_chols)]
            chl_indices = flatten([list(range(n_chols)) for _ in z])
            records.append(
                (trj.system.name, fr * 1000, chl_indices, z_col, a))
    df = pd.DataFrame.from_records(
        records, columns=['system', 'timepoint', 'chl_index', 'z', 'density'])
    df.sort_values('system', inplace=True, ignore_index=True)
    df['CHL amount, %'] = df['system'].str.split(
        '_chol', n=1, expand=True)[1]
    df['system'] = df['system'].str.split('_chol', n=1, expand=True)[0]
    df.replace(to_replace=[None], value=0, inplace=True)
    return df.explode(['chl_index', 'z', 'density'])


def angle_refers_to_component(syst, angle, chl, b, e, dt) -> tuple:
    '''
    depending on system and chl amount determine component:
    1 vertical, 2 horizontal
    returns tuple (1,), (2,) or (1,2)

    ctr +- 1/2 wid => angle in component
    components are located in notebooks/chol_tilt/{syst}_{b}-{e}-{dt}_4_comps.csv
    '''
    comps = pd.read_csv('/home/klim/Documents/chol_impact/'
                        f'notebooks/chol_tilt/{syst}_chol{chl}_{b}-{e}-{dt}_4_comps.csv')
    comps = comps.sort_values(['ctr']).reset_index(drop=True)
    res = []
    if angle >= 0:
        if comps.loc[3, 'ctr'] - comps.loc[3, 'wid'] / 2 <= angle:
            res.append(2)
        if angle <= comps.loc[2, 'ctr'] + comps.loc[2, 'wid'] / 2:
            res.append(1)
    if angle < 0:
        if angle <= comps.loc[0, 'ctr'] + comps.loc[0, 'wid'] / 2:
            res.append(2)
        if angle >= comps.loc[1, 'ctr'] - comps.loc[1, 'wid'] / 2:
            res.append(1)
    return tuple(res)


def create_tilt_density_table(trj_slices: list[TrajectorySlice],
                              comp_b: int, comp_e: int, comp_dt: int) -> None:
    '''
    calculate fractions of horizontal component depending on density and z coordinate
    comp_b, comp_e, comp_dt -- b, e, dt of component splitting

    formula is simple: for each system for each Z coordinta and each timepoint
    calculate (density sum by one of components) / (density sum of both components)
    '''

    df = pd.read_csv(trj_slices[0].system.path / 'notebooks' /
                     'integral_parameters' /
                     f'chl_tilt_{comp_b}-{comp_e}-{comp_dt}_with_comps.csv')
    df = df[df['timepoint'].astype(str).str.fullmatch('|'.join(
        [str(i) for i in range(trj_slices[0].b * 1000,
                               trj_slices[0].e * 1000, trj_slices[0].dt)]))].copy()
    df = df.reset_index().drop(columns=['index'])

    # print('determining angle components...')
    # df['comp'] = df.apply(
    #     lambda row: angle_refers_to_component(
    #         row['system'], row['α, °'], row['CHL amount, %'],
    #         trj_slices[0].b, trj_slices[0].e, trj_slices[0].dt), axis=1)

    print('reformatting density data...')
    dens = obtain_df_of_density(trj_slices).sort_values(
        ['system', 'CHL amount, %', 'timepoint',
         'chl_index', 'z'], ignore_index=True).drop_duplicates(ignore_index=True)
    dens['chl_index'] = dens['chl_index'].astype(int)
    dens['CHL amount, %'] = dens['CHL amount, %'].astype(int)
    common_ndx = ['system', 'CHL amount, %', 'timepoint', 'chl_index']

    print('analyzing components and densities...')
    merged = dens.merge(df, on=common_ndx, how='left').drop_duplicates(
        ignore_index=True).sort_values(
        ['system', 'CHL amount, %', 'timepoint', 'z'], ignore_index=True)
    merged['density'] = merged.density.astype(float)
    # merged = merged.explode(['comp'])
    # merged['comp'] = merged.comp.astype(str)
    # merged = merged.join(pd.get_dummies(merged['comp']))
    merged['comp_dens'] = merged['1'] * \
        merged['density'] + merged['2'] * merged['density']
    merged = merged.join(merged.groupby(
        ['system', 'CHL amount, %', 'timepoint', 'z'])['comp_dens'].sum(),
        on=['system', 'CHL amount, %', 'timepoint', 'z'], rsuffix='_sum')
    merged['% of horizontal component'] = merged['2'] * \
        merged['density'] / merged['comp_dens_sum'] * 100
    merged['% of vertical component'] = merged['1'] * \
        merged['density'] / merged['comp_dens_sum'] * 100
    merged.to_csv(trj_slices[0].system.path / 'notebooks' /
                  'integral_parameters' /
                  f'tilt_density_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv',
                  index=False)


def angl_dens_multiple_plot(experiments: dict,
                            trj_slices: list[TrajectorySlice],
                            df: pd.DataFrame, y: str, out: str, y2=None) -> None:
    '''
    plot all density_angles plots
    y -- name of variable in df which to plot
    y2 -- second variable to plot
    out -- outfile name suffix
    '''

    for exp, systs in experiments.items():
        fig, axs = plt.subplots(3, 3, figsize=(
            21, 21), sharex=True, sharey=True)
        systs = [(s + '_chol10', s + '_chol30', s + '_chol50')
                 for s in systs]
        for s_row, ax_row in zip(systs, axs):
            for s, ax in zip(s_row, ax_row):
                ax2 = ax.twinx()
                data = df[(df['system'] == s.split('_chol', 1)[0])
                          & (df['CHL amount, %'] == int(s.split('_chol', 1)[1]))]
                sns.histplot(data=data,
                             x='z', y=y, ax=ax2,
                             bins=50, alpha=0.5,
                             stat='density')
                if y2 is not None:
                    sns.histplot(data=data,
                                 x='z', y=y2, ax=ax2,
                                 bins=50, alpha=0.5,
                                 color='C1', stat='density')

                try:
                    plot_density_profile(ax,
                                         TrajectorySlice(
                                             System(
                                                 trj_slices[0].system.path, s),
                                             trj_slices[0].b, trj_slices[0].e, trj_slices[0].dt),
                                         groups=['chols'], color='k')
                    plot_density_profile(ax,
                                         TrajectorySlice(
                                             System(
                                                 trj_slices[0].system.path, s),
                                             trj_slices[0].b, trj_slices[0].e, trj_slices[0].dt),
                                         groups=['phosphates'], color='C3')
                except FileNotFoundError as e:
                    print(e)

                if ax != ax_row[-1]:
                    ax2.set_ylabel('')
                    ax2.yaxis.set_ticklabels([])
                if ax != ax_row[0]:
                    ax.set_ylabel('')
                ax.set_xlabel('')
                ax.set_title(
                    f"{s.split('_chol', 1)[0]}, {s.split('_chol', 1)[1]} % CHL")
        for ax in axs[-1, :]:
            ax.set_xlabel('Z, nm')
        labels, handles = axs[0, 0].get_legend_handles_labels()
        fig.legend(labels, handles)
        fig.suptitle(exp)
        plt.savefig(trj_slices[0].system.path / 'notebooks' / 'integral_parameters' /
                    f'{"_".join(exp.split())}_'
                    f'density_{out}_{trj_slices[0].b}_{trj_slices[0].e}_{trj_slices[0].dt}.png',
                    bbox_inches='tight')


def plot_angles_density(experiments: dict, trj_slices: list[TrajectorySlice],
                        comp_b: int, comp_e: int, comp_dt: int) -> None:
    '''
    calculate percentage of chl tilt horizontal component in systems
    and plot it together with density plots;
    9 plots per experiment
    '''
    trj_slices = [s for s in trj_slices if 'chol' in s.system.name]
    if not (trj_slices[0].system.path / 'notebooks' /
            'integral_parameters' /
            f'tilt_density_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv'
            ).is_file():
        create_tilt_density_table(trj_slices, comp_b, comp_e, comp_dt)
    df = pd.read_csv(trj_slices[0].system.path / 'notebooks' /
                     'integral_parameters' /
                     f'tilt_density_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')
    print('plotting horizontal and vertival component distribution of angles with '
          'CHL density profiles...')
    angl_dens_multiple_plot(experiments, trj_slices, df,
                            '% of horizontal component', 'both_comps',
                            '% of vertical component')
    print('done.')


def components_z_2d_ks_statistics(trj_slices: list[TrajectorySlice],
                                  n_points: int = 50, max_n: int = 5000):
    trj_slices = [s for s in trj_slices if 'chol' in s.system.name]
    # split list in chunks of three
    trj_slices_chunked = [trj_slices[i:i + 3]
                          for i in range(0, len(trj_slices), 3)]
    df = pd.read_csv(PATH / 'notebooks' /
                     'integral_parameters' /
                     f'tilt_density_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')

    for i in trj_slices_chunked:
        fig, axs = plt.subplots(3, 1, figsize=(10, 16))
        for ax, j in zip(axs, i):
            s = str(j.system)
            print(s)
            data = df[(df['system'] == s.split('_chol', 1)[0])
                      & (df['CHL amount, %'] == int(s.split('_chol', 1)[1]))]

            p_values_test = []
            d_values_test = []
            p_values_rand = []
            d_values_rand = []

            sample_sizes = np.linspace(10, max_n, n_points, dtype=int)

            for sample_size, _ in zip(sample_sizes, range(1, n_points + 1)):

                data_sampled = data.sample(
                    n=sample_size).dropna().reset_index(drop=True)
                p_val, d_val = ndtest.ks2d2s(data_sampled['z'],
                                             data_sampled['% of horizontal component'],
                                             data_sampled['z'],
                                             data_sampled['% of vertical component'], extra=True)
                d_values_test.append(d_val)
                p_values_test.append(p_val)

                distr1 = np.random.normal(size=[sample_size, 2])
                distr2 = np.random.normal(size=[sample_size, 2])
                p_val, d_val = ndtest.ks2d2s(distr1[:, 0],
                                             distr1[:, 1],
                                             distr2[:, 0],
                                             distr2[:, 1], extra=True)
                d_values_rand.append(d_val)
                p_values_rand.append(p_val)
            print('done, plotting...')

            ax2 = ax.twinx()
            l1 = ax.plot(sample_sizes, p_values_rand, c='k', marker='s', ms=3, linestyle='-',
                         label='randomly generated 2D normal distributions (p-value)')
            l2 = ax2.plot(sample_sizes, d_values_rand, c='k', marker='o', ms=2, linestyle=':',
                          label='randomly generated 2D normal distributions (KS statistic)')
            l3 = ax.plot(sample_sizes, p_values_test, c='r', marker='s', ms=3,
                         label='Z - % of horizontal vs vertical component (p-value)')
            l4 = ax2.plot(sample_sizes, d_values_test, c='r', marker='o', ms=2, linestyle=':',
                          label='Z - % of horizontal vs vertical component (KS statistic)')
            ax.plot(sample_sizes, [0.01 for _ in sample_sizes],
                    c='grey', linestyle='--')
            # plt.yscale('log')
            ax.set_title(s)
            ax.set_ylabel('p-value')
            ax2.set_ylabel('KS statistic')
            ax.set_xlabel(f'n of samples (total: {len(data)})')
            if ax == axs[0]:
                lns = l1 + l2 + l3 + l4
                labs = [l.get_label() for l in lns]
                ax.legend(lns, labs, loc='upper right')
            ax.text(max_n * 0.9, 0.05, 'p=0.01')

        outname = str(i[0].system).rsplit('_', 1)[0]
        fig.savefig(
            PATH / 'notebooks' / 'chol_tilt' / 'components_z_ks_statistics' /
            f'components_z_2d_ks_statistics_{outname}_'
            f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.png',
            bbox_inches='tight')


def get_no_and_both_components_percentage(infile: Path, outfile: Path):
    '''
    generate and save table with % of molecules without angle components
    and % of molecules with both components
    infile should be _coords_with_comps.csv
    '''
    df = pd.read_csv(infile)
    both_comps = df.drop(columns=['1', '2'])
    no_comps = (both_comps.drop_duplicates()
                .groupby(['system', 'CHL amount, %'])
                .agg(nan_sum=('nan', 'sum'), count=('nan', 'count')))
    no_comps['nan, %'] = no_comps['nan_sum'] / no_comps['count'] * 100
    dups = both_comps[both_comps.duplicated()]
    dups = dups.groupby(['system', 'CHL amount, %']).agg(
        dup_count=('nan', 'count'))
    non_dup_comps = pd.concat((no_comps, dups), axis=1)
    non_dup_comps['both comps, %'] = non_dup_comps['dup_count'] / \
        non_dup_comps['count'] * 100
    non_dup_comps.to_csv(outfile)


def get_n_of_chl_in_monolayers(infile: Path, outfile: Path):
    '''
    generate and save table with n of CHL molecules in monolayers
    infile should be _coords_with_comps.csv
    '''
    df = pd.read_csv(infile)
    df_wo_comps = df.drop(columns=['1', '2']).drop_duplicates()
    df_wo_comps = df_wo_comps.groupby(
        ['timepoint', 'system', 'CHL amount, %']
    ).agg(monolayer_count=('monolayer', 'value_counts'))
    monolayer = df_wo_comps.reset_index().pivot(
        columns='monolayer', values='monolayer_count')
    monolayer['upper'] = monolayer['upper'].shift(-1)
    uneven_monolayers = pd.concat(
        (df_wo_comps.reset_index().drop(
            columns=['monolayer', 'monolayer_count']).drop_duplicates(),
            monolayer.dropna().astype(int)), join='inner',
        axis=1)
    uneven_monolayers.sort_values(
        ['system', 'CHL amount, %', 'timepoint']
    ).drop(columns=['timepoint']).drop_duplicates().to_csv(outfile, index=False)


def generate_coords_comps_table(trj_slices: list[TrajectorySlice],
                                chl_tilt_b: int, chl_tilt_e: int, chl_tilt_dt: int):
    '''
    obtaining angles for trajectory slices selected if they are not calculated
    collecting coordinates of CHL cOMs and O3 positions and adding them to dataframe of angles
    then calculating components for this DataFrame
    chl_tilt_b, chl_tilt_e, chl_tilt_dt -- for components
    '''
    trj_slices = [s for s in trj_slices if 'chol' in s.system.name]

    if not (PATH / 'notebooks' / 'integral_parameters' /
            f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv').is_file():
        chl_tilt_angle(trj_slices, no_comps=True)
    df = pd.read_csv(PATH / 'notebooks' /
                     'integral_parameters' /
                     f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')
    df = df.drop_duplicates(ignore_index=True)

    system = []
    timepoint = []
    chl_amount = []
    chl_index = []
    x_com = []
    y_com = []
    z_com = []
    x_o = []
    y_o = []
    z_o = []
    monolayer = []
    zmem = []
    x_box = []
    y_box = []
    z_box = []

    print('collecting coords...')

    for trj in list(dict.fromkeys(trj_slices)):
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
            zmem_tmp = 0.5 * (leaflet_1.centroid() + leaflet_0.centroid())[2]
            chols = u.select_atoms("resname CHL")

            if (np.sum(np.isin(chols.residues.resids, leaflet_1.resids)) == 0 or
               np.sum(np.isin(chols.residues.resids, leaflet_0.resids)) == 0):
                print('CHL not presented in monolayer!')

            chols_coms = {c: i.atoms.center_of_mass()
                          for c, i in enumerate(chols.residues)}
            o3_positions = {c: i.atoms.select_atoms("name O3").positions[0]
                            for c, i in enumerate(chols.residues)}

            for c, mol in enumerate(chols_coms):
                system.append(trj.system.name.split('_chol', 1)[0])
                timepoint.append(int(ts.time))
                chl_amount.append(int(trj.system.name.split('_chol', 1)[1]))
                chl_index.append(c)
                x_com.append(chols_coms[mol][0])
                y_com.append(chols_coms[mol][1])
                z_com.append(chols_coms[mol][2])
                x_o.append(o3_positions[mol][0])
                y_o.append(o3_positions[mol][1])
                z_o.append(o3_positions[mol][2])
                if chols.residues.resids[mol] in leaflet_0.resids:
                    monolayer.append('upper')
                elif chols.residues.resids[mol] in leaflet_1.resids:
                    monolayer.append('lower')
                else:
                    monolayer.append('na')
                zmem.append(zmem_tmp)
                x_box.append(ts.dimensions[0])
                y_box.append(ts.dimensions[1])
                z_box.append(ts.dimensions[2])

    print('assigning lists...')

    coords_df = pd.DataFrame(
        {'system': system,
         'timepoint': timepoint,
         'CHL amount, %': chl_amount,
         'chl_index': chl_index,
         'x_com': x_com,
         'y_com': y_com,
         'z_com': z_com,
         'x_o': x_o,
         'y_o': y_o,
         'z_o': z_o,
         'monolayer': monolayer,
         'zmem': zmem,
         'x_box': x_box,
         'y_box': y_box,
         'z_box': z_box},
    )

    coords_df.to_csv(PATH / 'notebooks' / 'integral_parameters' /
                     f'coords_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv',
                     index=False)

    outmerge = pd.merge(df, coords_df,
                        on=['system', 'timepoint', 'chl_index', 'CHL amount, %'],
                        indicator=True, how='outer')
    outmerge.to_csv(
        PATH / 'notebooks' / 'integral_parameters' /
        f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}_coords_debug.csv',
        index=False)

    df = pd.merge(df, coords_df, on=['system', 'timepoint', 'chl_index', 'CHL amount, %'])

    df.to_csv(PATH / 'notebooks' / 'integral_parameters' /
                     f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}_coords.csv',
                     index=False)

    add_comps_to_chl_tilt(
        PATH / 'notebooks' / 'integral_parameters' /
        f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}_coords.csv',
        chl_tilt_b, chl_tilt_e, chl_tilt_dt)

    get_no_and_both_components_percentage(
        infile=PATH / 'notebooks' / 'integral_parameters' /
        f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-'
        f'{trj_slices[0].dt}_coords_with_comps.csv',
        outfile=PATH / 'notebooks' / 'integral_parameters' / 'components' /
        f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-'
        f'{trj_slices[0].dt}_no_and_both_comps.csv')
    get_n_of_chl_in_monolayers(
        infile=PATH / 'notebooks' / 'integral_parameters' /
        f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-'
        f'{trj_slices[0].dt}_coords_with_comps.csv',
        outfile=PATH / 'notebooks' / 'integral_parameters' /
        f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-'
        f'{trj_slices[0].dt}_n_of_chl_in_monolayers.csv')


def angle_components_3d(trj_slices: list[TrajectorySlice],
                        comp_b: int, comp_e: int, comp_dt: int):
    '''
    plotting 3d scatter plots of CHL COMs and O3 positions
    with colorcoding the amount of horizontal component (red - 2 comp, green - 1)
    '''
    trj_slices = [s for s in trj_slices if 'chol' in s.system.name]

    if not (
        PATH / 'notebooks' / 'integral_parameters' /
        f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}_coords_with_comps.csv'
    ).is_file():
        generate_coords_comps_table(trj_slices, comp_b, comp_e, comp_dt)

    df = pd.read_csv(
        PATH / 'notebooks' / 'integral_parameters' /
        f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}_coords_with_comps.csv'
    )
    print('averaging data...')
    averaged_df = df.groupby(
        ['system', 'CHL amount, %', 'chl_index'], as_index=False).mean()
    for trj in trj_slices:
        print(f'plotting {str(trj.system)}...')
        data = averaged_df[
            (averaged_df['system'] == str(trj.system).rsplit('_chol', 1)[0]) &
            (averaged_df['CHL amount, %'] == int(str(trj.system).rsplit('_chol', 1)[1]))]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection='3d')
        x = np.dstack((data['x_com'].to_numpy(), data['x_o'].to_numpy()))
        y = np.dstack((data['y_com'].to_numpy(), data['y_o'].to_numpy()))
        z = np.dstack((data['z_com'].to_numpy(), data['z_o'].to_numpy()))
        im = ax.scatter(data['x_com'], data['y_com'], data['z_com'], c=data['2'],
                        cmap='RdYlGn_r', ec='k', s=40)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.06,
                     label='share of horizontal component')
        for i in range(len(x[0])):
            ax.plot(x[0][i], y[0][i], z[0][i], color='gray', lw=1)
        ax.set_title(str(trj.system) +
                     f' ({trj_slices[0].b}-{trj_slices[0].e} ns, dt={trj_slices[0].dt} ps)')
        ax.set_xlabel('X, Å')
        ax.set_ylabel('Y, Å')
        ax.set_zlabel('Z, Å')
        fig.savefig(
            PATH / 'notebooks' / 'chol_tilt' / 'components_3d_plots' /
            f'{str(trj.system)}_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}'
            '_tilt_comps_3d.png', bbox_inches='tight')
        print('done.')


def angle_components_2d(trj_slices: list[TrajectorySlice],
                        comp_b: int, comp_e: int, comp_dt: int,
                        draw_o_pos=False):
    '''
    plotting 2d (angle - distance to bilayer center) scatter plots of CHL COMs positions
    (averaged data)
    with colorcoding the amount of horizontal component (red - 2 comp, green - 1)
    '''
    trj_slices = [s for s in trj_slices if 'chol' in s.system.name]
    if not (PATH / 'notebooks' / 'integral_parameters' /
            f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}_coords_with_comps.csv'
            ).is_file():
        generate_coords_comps_table(trj_slices, comp_b, comp_e, comp_dt)

    df = pd.read_csv(
        PATH / 'notebooks' / 'integral_parameters' /
        f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}_coords_with_comps.csv')

    print('averaging data...')
    averaged_df = df.groupby(
        ['system', 'CHL amount, %', 'chl_index'], as_index=False).mean(numeric_only=True)

    c = 0
    for exp in chunker(trj_slices, 9):
        fig, axs = plt.subplots(3, 3, figsize=(
            10, 12), sharex=True, sharey=True)
        fig.suptitle(tuple(EXPERIMENTS.keys())[c])
        print(f'plotting {tuple(EXPERIMENTS.keys())[c]}...')
        for syst, ax_row in zip(chunker(exp, 3), axs):
            for trj, ax in zip(syst, ax_row):
                print(f'plotting {str(trj.system)}...')
                data = averaged_df[
                    (averaged_df['system'] == str(trj.system).rsplit('_chol', 1)[0]) &
                    (averaged_df['CHL amount, %'] == int(str(trj.system).rsplit('_chol', 1)[1]))]
                im = ax.scatter(np.abs(data['z_com'] - data['zmem']), np.abs(data['α, °']),
                                c=data['2'], cmap='RdYlGn_r', ec='k')

                if draw_o_pos:
                    z = np.dstack(
                        (data['z_com'].to_numpy(), data['z_o'].to_numpy()))
                    alpha = np.dstack(
                        (np.abs(data['α, °']), np.abs(data['α, °'])))
                    for i in range(len(z[0])):
                        ax.plot(z[0][i], alpha[0][i], color='gray', lw=1)

                ax.set_title(trj.system)
                if ax in axs[:, 0]:
                    ax.set_ylabel('α, °')
                if ax == axs[-1, 1]:
                    ax.set_xlabel('Distance to bilayer center, Å')
        cbar_ax = fig.add_axes([.92, 0.34, 0.01, 0.3])
        cbar = fig.colorbar(im, cax=cbar_ax, ticks=np.linspace(0, 1, 5))
        cbar.set_ticklabels((100 * np.linspace(0, 1, 5)).astype('int'))
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('% of horizontal component', rotation=270)
        outname = '_'.join(tuple(EXPERIMENTS.keys())[c].split())
        plt.savefig(
            PATH / 'notebooks' / 'chol_tilt' / 'components_2d_plots' /
            f'tilt_angle_{outname}_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.png',
            bbox_inches='tight')
        c += 1
    print('done.')


def angle_components_2d_hists(trj_slices: list[TrajectorySlice],
                              comp_b: int, comp_e: int, comp_dt: int):
    '''
    plotting 2d (angle - distance to bilayer center) hists of CHL COMs positions
    (data is not averaged)
    vertical component is blue, horizontal is orange
    '''
    trj_slices = [s for s in trj_slices if 'chol' in s.system.name]
    if not (
        PATH / 'notebooks' / 'integral_parameters' /
        f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}_coords_with_comps.csv'
    ).is_file():
        generate_coords_comps_table(trj_slices, comp_b, comp_e, comp_dt)

    df = pd.read_csv(
        PATH / 'notebooks' / 'integral_parameters' /
        f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}_coords_with_comps.csv'
    )
    c = 0
    for exp in chunker(trj_slices, 9):
        fig, axs = plt.subplots(3, 3, figsize=(
            10, 12), sharex=True, sharey=True)
        fig.suptitle(tuple(EXPERIMENTS.keys())[c])
        print(f'plotting {tuple(EXPERIMENTS.keys())[c]}...')
        for syst, ax_row in zip(chunker(exp, 3), axs):
            for trj, ax in zip(syst, ax_row):
                print(f'plotting {str(trj.system)}...')
                data = df[(df['system'] == str(trj.system).rsplit('_chol', 1)[0]) &
                          (df['CHL amount, %'] == int(str(trj.system).rsplit('_chol', 1)[1]))]
                comp1 = data[data['1'] == 1]
                comp2 = data[data['2'] == 1]

                sns.histplot(x=np.abs(comp1['z_com'] - comp1['zmem']),
                             y=np.abs(comp1['α, °']),
                             color='C0',
                             stat='density',
                             bins=30,
                             ax=ax)

                sns.histplot(x=np.abs(comp2['z_com'] - comp2['zmem']),
                             y=np.abs(comp2['α, °']),
                             color='C1',
                             stat='density',
                             bins=30,
                             ax=ax)

                ax.set_title(trj.system)
                if ax in axs[:, 0]:
                    ax.set_ylabel('α, °')
                if ax == axs[-1, 1]:
                    ax.set_xlabel('Distance to bilayer center, Å')

        plt.savefig(
            PATH / 'notebooks' / 'chol_tilt' / 'components_2d_plots' /
            f'tilt_angle_hist_{tuple(EXPERIMENTS.keys())[c]}_'
            f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.png',
            bbox_inches='tight')
        c += 1
    print('done.')


@ sparkles
@ duration
def main():
    '''
    parse arguments and execute all or some of the functions:
    chl_tilt_angle, angle_components_density, angle_components_3d
    '''
    parser = argparse.ArgumentParser(
        description='Script to obtain chl angle components and draw CHL angle '
        'component+density plots')
    parser.add_argument('--chl_tilt_angle',
                        action='store_true',
                        help='calculate angles between CHL c3-c17 vector and bilayer plane '
                        'in each frame of trajectory for each CHL molecule, '
                        'split to components')
    parser.add_argument('--angle_components_density',
                        action='store_true',
                        help='plot percentage of component related to Z axis '
                        'in combination with density plot')
    parser.add_argument('--angle_components_3d',
                        action='store_true',
                        help='draw 3d plots of distribution of components')
    parser.add_argument('--angle_components_2d',
                        action='store_true',
                        help='draw 2d (angle - Z) plots of distribution of components')
    parser.add_argument('--generate_coords_comps_table',
                        action='store_true',
                        help='generate coords comps distr for timerange')
    parser.add_argument('-b', '--b', type=int, default=150,
                        help='beginning time in ns, default=150')
    parser.add_argument('-e', '--e', type=int, default=200,
                        help='ending time in ns, default=200')
    parser.add_argument('-dt', '--dt', type=int, default=1000,
                        help='dt in ps (default=1000)')
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

    chl_tilt_b, chl_tilt_e, chl_tilt_dt = [
        int(i) for i in args.chl_tilt_b_e_dt.split()]

    if args.chl_tilt_angle:
        chl_tilt_trj_slices = [TrajectorySlice(
            System(PATH, s), chl_tilt_b, chl_tilt_e, chl_tilt_dt) for s in systems]

        chl_tilt_angle(chl_tilt_trj_slices)
        path_to_df = (PATH / 'notebooks' /
                      'integral_parameters' /
                      f'chl_tilt_{chl_tilt_b}-{chl_tilt_e}-{chl_tilt_dt}.csv')
        add_comps_to_chl_tilt(path_to_df, chl_tilt_b,
                              chl_tilt_e, chl_tilt_dt, plot_comps_ratio=True)

    if args.angle_components_density:
        plot_angles_density(EXPERIMENTS, trj_slices,
                            chl_tilt_b, chl_tilt_e, chl_tilt_dt)
        sns.set_context('paper')
        components_z_2d_ks_statistics(trj_slices)

    if args.angle_components_3d:
        angle_components_3d(trj_slices, chl_tilt_b, chl_tilt_e, chl_tilt_dt)

    if args.angle_components_2d:
        angle_components_2d(trj_slices, chl_tilt_b, chl_tilt_e, chl_tilt_dt)
        angle_components_2d_hists(
            trj_slices, chl_tilt_b, chl_tilt_e, chl_tilt_dt)

    if args.generate_coords_comps_table:
        generate_coords_comps_table(trj_slices, chl_tilt_b, chl_tilt_e, chl_tilt_dt)


if __name__ == '__main__':
    main()
