

# %% md
# This notebook is for testing and troubleshooting code
#
# # cholesterol_angle_components_density.py


# %% md
# ### Imports and define functions

# %%

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from modules import ndtest
from modules.general import flatten, opener, sparkles, duration
from modules.traj import System, TrajectorySlice
from modules.density import plot_density_profile
from modules.constants import PATH, TO_RUS, EXPERIMENTS
from integral_parameters_script import chl_tilt_angle


def add_comps_to_chl_tilt(b: int, e: int, dt: int):
    '''
    add info which component is every angle in chl_tilt_{b}-{e}-{dt}.to_csv
    (file produced by chl_tilt_angle() function)
    '''
    df = pd.read_csv(PATH / 'notebooks' /
                     'integral_parameters' / f'chl_tilt_{b}-{e}-{dt}.csv')
    print('determining angle components...')
    df['comp'] = df.apply(
        lambda row: angle_refers_to_component(
            row['system'], row['α, °'], row['CHL amount, %'],
            b, e, dt), axis=1)
    df = df.explode(['comp'])
    df['comp'] = df.comp.astype(str)
    df = df.join(pd.get_dummies(df['comp']))
    df = df.drop(columns=['comp']).drop_duplicates()
    df.to_csv(PATH / 'notebooks' /
              'integral_parameters' / f'chl_tilt_{b}-{e}-{dt}_with_comps.csv',
              index=False)
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
                sns.kdeplot(data=data,
                            x='z', y=y, ax=ax2,
                            bins=50, alpha=0.5,
                            stat='density')
                if y2 is not None:
                    sns.kdeplot(data=data,
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

# %% md
# ## 2D Kolmogorov-Smirnov test of Z-% of horizontal component vs Z-% of vertical component
# Creating plots of p-value depending on number of samples and comparing with KS test of
# two random 2D normal distributions.

# %%


systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                   for i in flatten(EXPERIMENTS.values())])


systems = list(dict.fromkeys([s for s in systems if 'dopc' in s]))[:4]
systems
# %%

trj_slices = [TrajectorySlice(
    System(PATH, s), 150, 200, 1000) for s in systems]

# %%


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
            print(f'done, plotting...')

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
        plt.savefig(PATH / 'notebooks' / 'chol_tilt' / 'components_z_ks_statistics' /
                    f'components_z_2d_ks_statistics_{outname}_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.png',
                    bbox_inches='tight')


# %%
components_z_2d_ks_statistics(trj_slices)

# %%
fig, ax = plt.subplots()
y = '% of horizontal component'
y2 = '% of vertical component'
s = 'dmpc_chol30'
ax2 = ax.twinx()
data = df[(df['system'] == s.split('_chol', 1)[0])
          & (df['CHL amount, %'] == int(s.split('_chol', 1)[1]))]
sns.histplot(data=data,
            x='z', y=y, ax=ax2,
            bins=50,
            # alpha=0.5,
            stat='density')
if y2 is not None:
    sns.histplot(data=data,
                x='z', y=y2, ax=ax2,
                bins=50,
                # alpha=0.5,
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
