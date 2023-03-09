'''
Script to obtain chl angle components and draw CHL angle component+density plots.
CHL ids and angle component are connected here.
'''

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from modules.general import flatten, opener, sparkles, duration
from modules.traj import System, TrajectorySlice
from modules.density import plot_density_profile
from modules.constants import PATH, TO_RUS, EXPERIMENTS
from integral_parameters_script import chl_tilt_angle


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


def create_tilt_density_table(trj_slices: list[TrajectorySlice]) -> None:
    '''
    calculate fractions of horizontal component depending on density and z coordinate
    '''

    df = pd.read_csv(trj_slices[0].system.path / 'notebooks' /
                     'integral_parameters' /
                     f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')
    df = df[df['timepoint'].astype(str).str.fullmatch('|'.join(
        [str(i) for i in range(trj_slices[0].b * 1000,
                               trj_slices[0].e * 1000, trj_slices[0].dt)]))].copy()
    df = df.reset_index().drop(columns=['index'])

    print('determining angle components...')
    df['comp'] = df.apply(
        lambda row: angle_refers_to_component(
            row['system'], row['α, °'], row['CHL amount, %'],
            trj_slices[0].b, trj_slices[0].e, trj_slices[0].dt), axis=1)

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
    merged = merged.explode(['comp'])
    merged['comp'] = merged.comp.astype(str)
    merged = merged.join(pd.get_dummies(merged['comp']))
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
                  'integral_parameters' / 'tilt_density.csv', index=False)


def angl_dens_multiple_plot(experiments: dict,
                            trj_slices: list[TrajectorySlice],
                            df: pd.DataFrame, y: str, out: str) -> None:
    '''
    plot all density_angles plots
    y -- name of variable in df which to plot
    out -- tilt or horiz_comp in outfile name
    '''

    for exp, systs in experiments.items():
        fig, axs = plt.subplots(3, 3, figsize=(
            21, 21), sharex=True, sharey=True)
        systs = [(s + '_chol10', s + '_chol30', s + '_chol50')
                 for s in systs]
        for s_row, ax_row in zip(systs, axs):
            for s, ax in zip(s_row, ax_row):
                ax2 = ax.twinx()
                sns.histplot(data=df[(df['system'] == s.split('_chol', 1)[0])
                                     & (df['CHL amount, %'] == int(s.split('_chol', 1)[1]))],
                             x='z', y=y, ax=ax2, alpha=0.5, bins=50,
                             stat='density')
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


def plot_angles_density(experiments: dict, trj_slices: list[TrajectorySlice]) -> None:
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
        create_tilt_density_table(trj_slices)
    df = pd.read_csv(trj_slices[0].system.path / 'notebooks' /
                     'integral_parameters' / 'tilt_density.csv')
    print('plotting horizontal component distribution of angles with '
          'CHL density profiles...')
    angl_dens_multiple_plot(experiments, trj_slices, df,
                            '% of horizontal component', 'hor_comp')
    print('done.')


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
    df.to_csv(PATH / 'notebooks' /
              'integral_parameters' / f'chl_tilt_{b}-{e}-{dt}_with_comps.csv',
              index=False)
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
    parser.add_argument('--rus',
                        action='store_true',
                        help='plot graphs with russian captions.')
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

    systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in flatten(EXPERIMENTS.values())])

    trj_slices = [TrajectorySlice(
        System(PATH, s), args.b, args.e, args.dt) for s in systems]

    if args.chl_tilt_b_e_dt is not None:
        chl_tilt_b, chl_tilt_e, chl_tilt_dt = [int(i)
                                               for i in args.chl_tilt_b_e_dt.split()]

        chl_tilt_trj_slices = [TrajectorySlice(
            System(PATH, s), chl_tilt_b, chl_tilt_e, chl_tilt_dt) for s in systems]

        chl_tilt_angle(chl_tilt_trj_slices, EXPERIMENTS, TO_RUS)
        add_comps_to_chl_tilt(chl_tilt_b, chl_tilt_e, chl_tilt_dt)


# %%
if __name__ == '__main__':
    main()

# %%
df = pd.read_csv(PATH / 'notebooks' /
                 'integral_parameters' / f'chl_tilt_100-200-100_with_comps.csv')
df
