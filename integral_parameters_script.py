#!/usr/bin/python3

'''
obtain and plot system parameters such as density profiles, area per lipid,
thickness, Scd and cholesterol tilt angle
'''

from pathlib import Path, PosixPath
from typing import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import axes
from scipy.interpolate import make_interp_spline
from scipy import integrate
from scipy.optimize import curve_fit
import MDAnalysis as mda

from modules.general import (
    opener, multiproc, flatten, sparkles,
    duration, calc_1d_com, get_keys_by_value
)
from modules.traj import System, TrajectorySlice
from modules.density import (
    get_densities, plot_density_profile, calculate_distances_between_density_groups,
    calculate_density_peak_widths
)
from modules.constants import PATH, TO_RUS, EXPERIMENTS


def get_chl_tilt(trj: TrajectorySlice) -> None:
    '''
    obtain cholesterol tilt angles
    '''

    print(
        f'🗄️ system:\t{trj.system}\n⌚️ time:\t{trj.b}-{trj.e} ns, dt={trj.dt} ps')

    if (trj.system.path / 'notebooks' / 'chol_tilt' /
            f'{trj.system}_{trj.b}-{trj.e}-{trj.dt}_tilt.xvg').is_file():
        print('already calculated, skipping...')
    else:
        print('obtaining 💁 system 🏙️ information...')
        u = mda.Universe(f'{trj.system.dir}/md/md.tpr',
                         f'{trj.system.dir}/md/md.gro',
                         refresh_offsets=True)
        chols = u.residues[u.residues.resnames == 'CHL'].atoms
        n_chol = len(u.residues[u.residues.resnames == 'CHL'])
        # adding 1 here because gromacs index files start atom numeration from 1
        # and in MDAnalysis atom numeration starts with 0
        c3 = ' '.join(
            list(map(str, (chols.select_atoms('name C3').indices + 1).tolist())))
        c17 = ' '.join(
            list(map(str, (chols.select_atoms('name C17').indices + 1).tolist())))

        with open(f'{trj.system.dir}/ch3_ch17.ndx', 'w', encoding='utf-8') as f:
            f.write(f'[C3]\n{c3}\n[C17]\n{c17}\n')

        print('calculating 👨‍💻 cholesterol 🫀 tilt 📐 ...')

        cmd = ['source `ls -t /usr/local/gromacs*/bin/GMXRC | head -n 1 ` && ',
               f'echo 0 1 | gmx bundle -s {trj.system.dir}/md/md.tpr',
               f'-f {trj.system.dir}/md/pbcmol.xtc',
               f'-na {n_chol} -z -n {trj.system.dir}/ch3_ch17.ndx',
               f'-b {trj.b*1000} -e {trj.e * 1000} -dt {trj.dt}',
               f'-ot {str(trj.system.path)}/notebooks/chol_tilt/'
               f'{trj.system}_{trj.b}-{trj.e}-{trj.dt}_tilt.xvg',
               '-xvg none']

        os.popen(' '.join(cmd)).read()
        print('done ✅\n')


def break_tilt_into_components(ax: axes._subplots.Axes, trj: TrajectorySlice) -> None:
    '''
    break and plot tilt components for one trj slice
    '''
    lines = opener(f'{trj.system.path}/notebooks/chol_tilt/'
                   f'{trj.system.name}_{trj.b}-{trj.e}-{trj.dt}_tilt.xvg')

    # a = np.array(
    #     list(map(float, flatten([i.split()[1:] for i in lines])))) - 90

    a = list(map(float, flatten([i.split()[1:] for i in lines])))
    a = np.array([i if i <= 90 else i - 180 for i in a])

    def func(x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            ctr = params[i]
            amp = params[i + 1]
            wid = params[i + 2]
            y = y + amp * np.exp(-((x - ctr) / wid)**2)
        return y

    x = np.arange(np.min(a), np.max(a),
                  (np.max(a) - np.min(a)) / 500)
    my_kde = sns.kdeplot(a,
                         ax=ax,
                         # fill=False,
                         lw=0,
                         color='black',
                         # element='poly',
                         # stat='density',
                         bw_adjust=0.4,
                         cut=0
                         )
    line = my_kde.lines[0]
    x, y = line.get_data()

    guess = [-30, 0.005, 20, -15, 0.02, 7, 15, 0.02, 7, 30, 0.005, 20]
    try:
        popt, _, _, _, _ = curve_fit(func, x, y, p0=guess, full_output=True)
        df = pd.DataFrame(popt.reshape(int(len(guess) / 3), 3),
                          columns=['ctr', 'amp', 'wid'])
        df['area'] = df.apply(lambda row: integrate.quad(func, np.min(
            x), np.max(x), args=(row['ctr'], row['amp'], row['wid']))[0], axis=1)

        fit = func(x, *popt)

        sns.histplot(a,
                     # fill=False,
                     ax=ax,
                     lw=0,
                     element='poly',
                     stat='density',
                     alpha=0.2,
                     edgecolor='black'
                     )

        ax.plot(x, fit, '-k', lw=2)
        ax.plot(x, func(x, *popt[:3]), '--')
        ax.plot(x, func(x, *popt[3:6]), '--')
        ax.plot(x, func(x, *popt[6:9]), '--')
        ax.plot(x, func(x, *popt[9:]), '--')

        df.to_csv(
            f'{trj.system.path}/notebooks/chol_tilt/'
            f'{trj.system.name}_{trj.b}-{trj.e}-{trj.dt}_4_comps.csv', index=False)

    except RuntimeError as e:
        print(f'couldn\'t curve_fit for {trj.system}: ', e)


def calculate_thickness(trj: TrajectorySlice) -> list[float]:
    '''
    calculates thickness in each step of trajectory part using phosphates densities
    '''

    trj_list = [TrajectorySlice(trj.system, fr, fr, 0)
                for fr in range(trj.b, trj.e, int(trj.dt / 1000))]
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(get_densities, trj_list)

    thickness_list = []
    for fr in range(trj.b, trj.e, int(trj.dt / 1000)):
        df = pd.read_csv(f'{trj.system.dir}/density_profiles/phosphates_{fr}-{fr}-0_dp.xvg',
                         header=None, delim_whitespace=True)
        x, y = df[0], df[1]
        x_y_spline = make_interp_spline(x, y)
        x_ = np.linspace(x.min(), x.max(), 500)
        thickness_list.append(
            calc_1d_com(x_[x_ > 0], x_y_spline(x_[x_ > 0]))
            - calc_1d_com(x_[x_ < 0], x_y_spline(x_[x_ < 0])))
    return thickness_list


def density_peak_widths_chols(trj: TrajectorySlice) -> list[float]:
    '''
    wrapper for calculate_density_width
    to get only 1 argument for using with multiproc()
    '''
    return calculate_density_peak_widths('chols', trj)


def calc_chols_p_dist(trj: TrajectorySlice) -> list[float]:
    '''
    wrapper for calculate_distances_between_density_groups
    to get only 1 argument for using with multiproc()
    '''
    return calculate_distances_between_density_groups('chols', 'phosphates', trj)


def calc_chols_o_p_dist(trj: TrajectorySlice) -> list[float]:
    '''
    wrapper for calculate_distances_between_density_groups
    to get only 1 argument for using with multiproc()
    '''
    return calculate_distances_between_density_groups('chols_o', 'phosphates', trj)


def calculate_area_per_lipid(trj: TrajectorySlice) -> list[float]:
    '''
    calculates area per lipid in each step of trajectory
    '''
    if not Path(f'{trj.system.dir}/md/box.xvg').is_file():
        cmd = ['source `ls -t /usr/local/gromacs*/bin/GMXRC | head -n 1 ` &&',
               f'echo 0 | gmx traj -f {trj.system.dir}/md/pbcmol.xtc',
               f'-s {trj.system.dir}/md/md.tpr -noz',
               f'-ob {trj.system.dir}/md/box.xvg -xvg none '
               f'-dt {trj.dt} -b {trj.b*1000} -e {trj.e*1000}']

        os.popen(' '.join(cmd)).read()

    box = pd.read_csv(f'{trj.system.dir}/md/box.xvg',
                      header=None, delim_whitespace=True, usecols=[1, 2], names=['x', 'y'])

    box['total area'] = box['x'] * box['y']
    box['area per lipid'] = box['total area'].apply(
        lambda x: x / 400 if '20x20' in trj.system.name else x / 100)

    return box['area per lipid'].tolist()


def run_scd_py(trj: TrajectorySlice) -> None:
    '''
    calculates per atom Scd values for the first PL in system name for whole trajectory part
    '''
    if not Path(f'{trj.system.path}/notebooks/scd/'
                f'{trj.system.name}_{trj.b}-{trj.e}-{trj.dt}_scd.csv').is_file():
        cmd = ['/nfs/belka2/soft/impulse/dev/inst/scd.py',
               f'-f {trj.system.dir}/md/pbcmol.xtc',
               f'-s {trj.system.dir}/md/md.tpr',
               f'-o {trj.system.path}/notebooks/scd/{trj.system.name}_{trj.b}-{trj.e}-{trj.dt}',
               f'-b {trj.b*1000} -e {trj.e*1000} --dt {trj.dt} --sel {trj.system.pl_selector()}']
        print(
            f'calculating Scd for {trj.system.name}, {trj.b}-{trj.e} ns, dt={trj.dt} ps...')
        os.popen(' '.join(cmd)).read()
    else:
        print(
            f'Scd file for {trj.system.name}, '
            f'{trj.b}-{trj.e} ns, dt={trj.dt} ps exists, skipping...')


def calculate_scd(trj: TrajectorySlice) -> None:
    '''
    calculates per atom Scd values for the first PL in system name for each step in trajectory part
    '''
    trj_list = [TrajectorySlice(trj.system, fr, fr, 0)
                for fr in range(trj.b, trj.e, int(trj.dt / 1000))]
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(run_scd_py, trj_list)


def scd_summary(trj_slices: list[TrajectorySlice]) -> None:
    '''
    aggregates data from individual _scd.csv files to single file
    '''
    scd_folder = Path('/home/klim/Documents/chol_impact/notebooks/scd')

    systems = []
    times = []
    chains = []
    atoms = []
    scds = []

    print('aggregating data...')

    for trj in trj_slices:
        scd_files = [str(scd_folder / f'{trj.system.name}_{i}-{i}-0_scd.csv')
                     for i in range(trj.b, trj.e, int(trj.dt / 1000))]
        for file in scd_files:
            lines = opener(file)
            systems.extend([trj.system.name for i in range(
                len(lines[1].split(',')) + len(lines[5].split(',')))])
            time = ' '.join([file.split('-')[1].split('_')[0], 'ns'])
            times.extend([time for i in range(
                len(lines[1].split(',')) + len(lines[5].split(',')))])
            chains.extend(['sn-2' for i in range(len(lines[1].split(',')))])
            chains.extend(['sn-1' for i in range(len(lines[5].split(',')))])
            atoms.extend(lines[1].split(','))
            atoms.extend(lines[5].split(','))
            scds.extend(lines[3].split(','))
            scds.extend(lines[7].split(','))

    print('creating summary table...')
    df = pd.DataFrame({'system': systems, 'timepoint': times,
                       'chain': chains, 'atom': atoms, 'scd': scds})
    df.sort_values(['system', 'chain', 'timepoint'],
                   inplace=True, ignore_index=True)
    df['CHL amount, %'] = df['system'].str.split('_chol', n=1, expand=True)[1]
    df['system'] = df['system'].str.split('_chol', n=1, expand=True)[0]
    df.replace(to_replace=[None], value=0, inplace=True)
    return df


def lists_of_values_to_df(func: Callable, trj_slices: list[TrajectorySlice]) -> pd.DataFrame:
    '''
    func: function such as calculate_thickness or calculate_arperlip,
          which takes trj_slices as input and gives list of floats as output

    this function calculates mean and std for each TrajectorySlice and
    creates dataframe with columns system, mean, std
    '''
    records = []
    for trj, i in multiproc(func, trj_slices, 8).items():
        records.append((trj.system.name, i))
    df = pd.DataFrame.from_records(
        records, columns=['system', 'data'])
    df.sort_values('system', inplace=True, ignore_index=True)
    df['CHL amount, %'] = df['system'].str.split('_chol', n=1, expand=True)[1]
    df['system'] = df['system'].str.split('_chol', n=1, expand=True)[0]
    df.replace(to_replace=[None], value=0, inplace=True)
    return df.explode('data')


def density(trj_slices: list[TrajectorySlice], experiments=None, to_rus=None) -> None:
    '''
    apply get_densities function to list of trajectories
    experiments and to_rus arguments are not used,
    but essential for activating from dict as implemented in main()
    '''
    print('obtain all densities...')
    with ProcessPoolExecutor(max_workers=8) as executor:
        executor.map(get_densities, trj_slices)
    print('done.')


def calculate_relative_changes(df: pd.DataFrame) -> pd.DataFrame:
    '''
    incoming dataframe should have columns 'experiment', 'system', 'CHL amount, %'
    and one column with data
    this function calculates relative changes inside each experiment for each bilayer
    with changing CHL amount
    '''
    df_relative = df.groupby(['experiment', 'system', 'CHL amount, %'],
                             as_index=False).agg(['mean', 'std']).reset_index()
    df_relative.columns = ['experiment',
                           'system', 'CHL amount, %', 'mean', 'std']

    def diff_mean(x: pd.Series) -> np.ndarray:
        '''
        calculates difference between n and n+1 row in Series in percents
        '''
        def rel_m(val1, val2):
            return (val2 - val1) / val1 * 100
        vals = [rel_m(x[n], x[n + 1]) for n in range(x.shape[0] - 1)]
        ret = list([np.nan])  # pad return vector however you need to
        ret = ret + vals
        return ret

    def diff_std(x: pd.Series, y: pd.Series) -> np.ndarray:
        '''
        calculates std between n and n+1 row in Series in percents
        '''
        def rel_s(s1, s2, m1, m2):
            return np.sqrt(np.abs((s2**2 * m1**2 - s1**2 * m2**2) / m1**4)) * 100
        vals = [rel_s(x[n], x[n + 1], y[n], y[n + 1])
                for n in range(x.shape[0] - 1)]
        ret = list([np.nan])  # pad return vector however you need to
        ret = ret + vals
        return ret

    df_relative['relative mean, %'] = df_relative.loc[:,
                                                      ['mean']].apply(diff_mean)
    df_relative.loc[df_relative[df_relative['CHL amount, %'] ==
                                df_relative['CHL amount, %'].min()]['relative mean, %'].index,
                    'relative mean, %'] = 0
    df_relative['relative std'] = df_relative.loc[:, [
        'std']].apply(diff_std, y=df_relative['mean'])
    df_relative.loc[df_relative[df_relative['CHL amount, %'] ==
                                df_relative['CHL amount, %'].min()]['relative std'].index,
                    'relative std'] = 0
    return df_relative.sort_values(['experiment', 'CHL amount, %'])


def plot_violins(csv: PosixPath, y: str, experiments: dict, to_rus: dict) -> None:
    '''
    plot violinplot for distribution of parameters by experiment
    and barplot for relative changes
    also save russian version
    '''
    def plots(df: pd.DataFrame, csv: PosixPath, y: str, to_rus: dict, rus: bool) -> None:
        '''
        translate everything to rus (optionally) and plot violins and relative changes plots
        '''
        df_relative = calculate_relative_changes(df)

        if rus:
            df['system'] = df['system'].apply(lambda x: to_rus[x])
            df['experiment'] = df['experiment'].apply(lambda x: to_rus[x])
            df.rename(columns={'system': 'Система',
                      y: to_rus[y]}, inplace=True)
            df_relative['system'] = df_relative['system'].apply(
                lambda x: to_rus[x])
            df_relative['experiment'] = df_relative['experiment'].apply(
                lambda x: to_rus[x])
            df_relative.rename(columns={'system': 'Система',
                                        'relative mean, %': 'Относительное изменение, %'},
                               inplace=True)

        x = 'Система' if rus else 'system'
        y = to_rus[y] if rus else y
        legend_title = 'Содержание ХС, %' if rus else 'CHL amount, %'
        y2 = 'Относительное изменение, %' if rus else 'relative mean, %'
        out = '_rus' if rus else ''

        g = sns.FacetGrid(df, col='experiment', height=7,
                          aspect=0.75, sharex=False, dropna=True)
        g.map_dataframe(sns.violinplot, x=x, y=y, hue='CHL amount, %',
                        cut=0, palette='RdYlGn_r', inner='quartile')
        g.axes[0][-1].legend(title=legend_title)
        # g.add_legend(title=legend_title)
        g.set_titles(col_template='{col_name}')

        plt.savefig(str(csv).split('.', 1)[0] + out + '.png',
                    bbox_inches='tight')
        plt.close()

        # plot relative changes
        g = sns.FacetGrid(df_relative, col='experiment', height=7,
                          aspect=0.75, sharex=False, dropna=True)
        g.map_dataframe(sns.barplot, x=x, y=y2, hue='CHL amount, %',
                        palette='RdYlGn_r', ec='k')
        g.axes[0][-1].legend(title=legend_title)
        # g.add_legend(title=legend_title)
        g.set_titles(col_template='{col_name}')
        for ax, exp in zip(g.axes[0], df_relative['experiment'].unique()):
            x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
            y_coords = [p.get_height() for p in ax.patches]
            yerr = []
            c = 0
            for i in y_coords:
                if np.isnan(i):
                    yerr.append(np.nan)
                else:
                    yerr.append(
                        df_relative[df_relative['experiment'] == exp]['relative std'].tolist()[c])
                    c += 1
            ax.errorbar(x=x_coords, y=y_coords, yerr=yerr, fmt='none', c='k')
        plt.savefig(str(csv).split('.', 1)[0] + '_relative' + out + '.png',
                    bbox_inches='tight')
        plt.close()

    df = pd.read_csv(csv)
    df = df[df.columns.intersection(
        ['system', 'experiment', 'CHL amount, %', y])]

    df['experiment'] = df['system'].apply(
        lambda x: get_keys_by_value(x, experiments))
    df = df.explode('experiment')
    order = {'dmpc': 0, 'dppc_325': 1, 'dspc': 2, 'popc': 3,
             'dopc': 4, 'dopc_dops50': 5, 'dops': 6}
    df = df.sort_values(by=['system'], key=lambda x: x.map(order))
    plots(df, csv, y, to_rus, False)
    plots(df, csv, y, to_rus, True)


def thickness(trj_slices: list[TrajectorySlice], experiments: dict, to_rus: dict) -> None:
    '''
    apply calculate_thickness function to list of trajectories and plot results as violinplot
    '''
    path, b, e, dt = trj_slices[0].system.path, trj_slices[0].b, trj_slices[0].e, trj_slices[0].dt

    print('obtaining thicknesses...')
    lists_of_values_to_df(calculate_thickness, trj_slices).rename(
        columns={'data': 'thickness, nm'}).to_csv(
        path / 'notebooks' / 'integral_parameters' / f'thickness_{b}-{e}-{dt}.csv', index=False)
    print('plotting results...')
    plot_violins(path / 'notebooks' / 'integral_parameters' / f'thickness_{b}-{e}-{dt}.csv',
                 'thickness, nm', experiments, to_rus)
    print('done.')


def arperlip(trj_slices: list[TrajectorySlice], experiments: dict, to_rus: dict) -> None:
    '''
    apply calculate_area_per_lipid function to list of trajectories and plot results as violinplot
    '''
    path, b, e, dt = trj_slices[0].system.path, trj_slices[0].b, trj_slices[0].e, trj_slices[0].dt
    print('obtaining area per lipid...')
    lists_of_values_to_df(calculate_area_per_lipid, trj_slices).rename(
        columns={'data': 'area per lipid, nm²'}).to_csv(
        path / 'notebooks' / 'integral_parameters' / f'arperlip_{b}-{e}-{dt}.csv', index=False)
    print('plotting results...')
    plot_violins(path / 'notebooks' / 'integral_parameters' / f'arperlip_{b}-{e}-{dt}.csv',
                 'area per lipid, nm²', experiments, to_rus)
    print('done.')


def scd(trj_slices: list[TrajectorySlice], experiments: dict, to_rus: dict) -> None:
    '''
    apply calculate_scd function to list of trajectories,
    unite data of several systems into one file
    and plot results (for chains) as violinplot
    '''
    path, b, e, dt = trj_slices[0].system.path, trj_slices[0].b, trj_slices[0].e, trj_slices[0].dt
    print('obtaining all scds...')
    with ProcessPoolExecutor(max_workers=8) as executor:
        executor.map(calculate_scd, trj_slices)
    print('saving scd...')
    scd_summary(trj_slices).to_csv(
        path / 'notebooks' / 'integral_parameters' / f'scd_{b}-{e}-{dt}.csv', index=False)
    print('plotting results...')
    plot_violins(path / 'notebooks' / 'integral_parameters' / f'scd_{b}-{e}-{dt}.csv',
                 'scd', experiments, to_rus)
    print('done.')


def chl_tilt_summary(trj_slices: list[TrajectorySlice]) -> None:
    '''
    aggregate chl tilt data of several systems into one file
    '''
    records = []
    for trj in trj_slices:
        lines = opener(trj_slices[0].system.path / 'notebooks' / 'chol_tilt' /
                       f'{trj.system.name}_{trj.b}-{trj.e}-{trj.dt}_tilt.xvg')
        timepoint = np.array(
            list(map(int, [i.split()[0] for i in lines])))
        a = list(map(float, flatten([i.split()[1:] for i in lines])))
        a = np.array([i if i <=90 else i-180 for i in a ])
        n_chols = int(a.shape[0] / timepoint.shape[0])
        timepoints = [i for i in timepoint for _ in range(n_chols)]
        chl_indices = flatten([list(range(n_chols)) for _ in timepoint])
        records.append((trj.system.name, timepoints, chl_indices, a))
    df = pd.DataFrame.from_records(
        records, columns=['system', 'timepoint', 'chl_index', 'α, °'])
    df.sort_values('system', inplace=True, ignore_index=True)
    df['CHL amount, %'] = df['system'].str.split('_chol', n=1, expand=True)[1]
    df['system'] = df['system'].str.split('_chol', n=1, expand=True)[0]
    df.replace(to_replace=[None], value=0, inplace=True)
    return df.explode(['timepoint', 'chl_index', 'α, °'])


def chl_tilt_angle(trj_slices: list[TrajectorySlice], experiments: dict = EXPERIMENTS, to_rus: dict = TO_RUS,
                   no_comps=False) -> None:
    '''
    apply get_chl_tilt function to list of trajectories,
    split each system into components (plot + save parameters),
    unite data of all systems into one file
    and plot results as violinplot
    '''
    trj_slices = [s for s in trj_slices if 'chol' in s.system.name]
    path, b, e, dt = trj_slices[0].system.path, trj_slices[0].b, trj_slices[0].e, trj_slices[0].dt
    print('obtaining cholesterol tilt...')
    with ProcessPoolExecutor(max_workers=8) as executor:
        executor.map(get_chl_tilt, trj_slices)
    print('saving chl tilt angles...')
    df = chl_tilt_summary(trj_slices)
    df.to_csv(
        path / 'notebooks' / 'integral_parameters' /
        f'chl_tilt_{b}-{e}-{dt}.csv',
        index=False)
    if not no_comps:
        df['α, °'] = df['α, °'].abs()
        df.to_csv(
            path / 'notebooks' / 'integral_parameters' / 'chl_tilt_to_plot.csv', index=False)
        print('plotting chol tilts and splitting into components...')
        for trj in trj_slices:
            if not Path(f'{trj.system.path}/notebooks/chol_tilt/'
                        f'{trj.system.name}_{trj.b}-{trj.e}-{trj.dt}_4_comps.csv').is_file():
                fig, ax = plt.subplots(figsize=(7, 7))
                break_tilt_into_components(ax, trj)
                ax.set_xlabel('Tilt (degree)')
                ax.set_ylabel('Density')
                fig.patch.set_facecolor('white')
                plt.savefig(f'{path}/notebooks/chol_tilt/'
                            f'{trj.system}_{trj.b}-{trj.e}-{trj.dt}_4_comps.png',
                            bbox_inches='tight', facecolor=fig.get_facecolor())
                plt.close()
        print('plotting results...')

        plot_violins(path / 'notebooks' / 'integral_parameters' / 'chl_tilt_to_plot.csv',
                     'α, °', experiments, to_rus)
    print('done.')


def chl_p_distance(trj_slices: list[TrajectorySlice], experiments: dict, to_rus: dict) -> None:
    '''
    apply calc_chols_p_dist and calc_chols_o_p_dist functions to list of trajectories,
    unite data of several systems into one file
    and plot results as violinplot
    '''
    trj_slices = [s for s in trj_slices if 'chol' in s.system.name]
    path, b, e, dt = trj_slices[0].system.path, trj_slices[0].b, trj_slices[0].e, trj_slices[0].dt
    print('obtaining chol-phosphates distances...')
    lists_of_values_to_df(calc_chols_p_dist, trj_slices).rename(
        columns={'data': 'distance, nm'}).to_csv(
        path / 'notebooks' / 'integral_parameters' /
        f'chols_phosphates_distances_{b}-{e}-{dt}.csv', index=False)
    print('obtaining chol_o-phosphates distances...')
    lists_of_values_to_df(calc_chols_o_p_dist, trj_slices).rename(
        columns={'data': 'distance, nm'}).to_csv(
        path / 'notebooks' / 'integral_parameters' /
        f'chols_o_phosphates_distances_{b}-{e}-{dt}.csv', index=False)
    print('plotting results...')
    plot_violins(path / 'notebooks' / 'integral_parameters' /
                 f'chols_phosphates_distances_{b}-{e}-{dt}.csv', 'distance, nm',
                 experiments, to_rus)
    plot_violins(path / 'notebooks' / 'integral_parameters' /
                 f'chols_o_phosphates_distances_{b}-{e}-{dt}.csv', 'distance, nm',
                 experiments, to_rus)
    print('done.')


def plot_dp_by_exp(experiments, trj_slices: list[TrajectorySlice], to_rus: dict, rus: bool) -> None:
    '''
    plots chol chol_o amd phosphates density profiles on same axis for
    systems with different amounts of CHL.
    plots systems from the same experiment on one figure
    '''
    chl = 'ХС' if rus else 'CHL'
    ylabel = 'Плотность, кг/м³' if rus else 'Density, kg/m³'
    xlabel = 'Z, нм' if rus else 'Z, nm'
    out = '_rus' if rus else ''

    for exp, systs in experiments.items():
        fig, axs = plt.subplots(1, 3, figsize=(
            21, 7), sharex=True, sharey=True)
        plotted = []
        for syst, ax in zip(systs, axs):
            c = 0  # for choosing colors
            for trj in [s for s in trj_slices if 'chol' in s.system.name]:  # trj_slices_chol
                if trj.system.name.rsplit('_', 1)[0] == syst and str(trj.system) not in plotted:
                    l = f'{trj.system.name.split("chol", 1)[1]} % ' + chl
                    plot_density_profile(
                        ax, trj, groups=['chols'],
                        color=sns.color_palette('Purples_r', 3)[c], label=chl + ', ' + l)
                    plot_density_profile(
                        ax, trj, groups=['chols_o'],
                        color=sns.color_palette('Blues_r', 3)[c], label=chl + ' O, ' + l)
                    plot_density_profile(
                        ax, trj, groups=['phosphates'],
                        color=sns.color_palette('Reds_r', 3)[c], label='PO4, ' + l)
                    plotted.append(str(trj.system))
                    if rus:
                        ax.set_title(to_rus[trj.system.name.rsplit('_', 1)[0]])
                    else:
                        ax.set_title(trj.system.name.rsplit('_', 1)[0])
                    c += 1
            ax.set_xlabel(xlabel)
        for i in axs[1:]:
            i.set_ylabel('')
        axs[0].set_ylabel(ylabel)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels)
        if rus:
            fig.suptitle(to_rus[exp], fontsize=20)
        else:
            fig.suptitle(exp, fontsize=20)
        plt.savefig(trj_slices[0].system.path / 'notebooks' / 'integral_parameters' /
                    f'{"_".join(exp.split())}_'
                    f'dp_{trj_slices[0].b}_{trj_slices[0].e}_{trj_slices[0].dt}{out}.png',
                    bbox_inches='tight')
        plt.close()


def density_profiles(experiments: dict, trj_slices: list[TrajectorySlice], to_rus: dict) -> None:
    '''
    plots density profile for each trajectory
    calculates chl peak widths and plots it
    plots chl and phosphates density profiles on single axes for trajectories
    '''
    path, b, e, dt = trj_slices[0].system.path, trj_slices[0].b, trj_slices[0].e, trj_slices[0].dt
    print('plotting density profiles...')
    for trj in trj_slices:
        if not Path(f'{path}/notebooks/dp/'
                    f'{trj.system}_{b}-{e}-{dt}_dp.png').is_file():
            fig, ax = plt.subplots(figsize=(7, 7))
            plot_density_profile(ax, trj)
            ax.set_title(trj.system.name)
            ax.legend()
            fig.patch.set_facecolor('white')
            plt.savefig(f'{path}/notebooks/dp/'
                        f'{trj.system}_{b}-{e}-{dt}_dp.png',
                        bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close()
    print('obtaining chols peak widths...')
    trj_slices_chol = [s for s in trj_slices if 'chol' in s.system.name]
    lists_of_values_to_df(density_peak_widths_chols, trj_slices_chol).rename(
        columns={'data': 'peak width, nm'}).to_csv(
        path / 'notebooks' / 'integral_parameters' /
        f'chols_peak_widths_{b}-{e}-{dt}.csv', index=False)
    print('plotting chols peak widths...')
    plot_violins(path / 'notebooks' / 'integral_parameters' /
                 f'chols_peak_widths_{b}-{e}-{dt}.csv',
                 'peak width, nm', experiments, to_rus)
    print('plotting dp by experiment...')
    plot_dp_by_exp(experiments, trj_slices, to_rus, False)
    plot_dp_by_exp(experiments, trj_slices, to_rus, True)
    print('done.')


def plot_scd_atoms(experiments: dict, trj_slices: list[TrajectorySlice], to_rus: dict) -> None:
    '''
    plot per atom scd data for each system (different amount of CHL),
    one file per experiment.
    '''
    def dfol(df: pd.DataFrame, system: str, chain: str, chl_amount: int) -> pd.DataFrame:
        '''
        extract data for one line in plot from df
        '''
        return df[(df['system'] == system)
                  & (df['chain'] == chain)
                  & (df['CHL amount, %'] == chl_amount)]

    def scd_plot(scd_ms: pd.DataFrame, exp: str, systs: tuple, to_rus: dict, rus: bool) -> None:
        '''
        plot scd data of one experiment
        '''
        out = '_rus' if rus else ''
        chls = 'ХС' if rus else 'CHL'
        scd_ms_part = scd_ms[scd_ms['system'].str.fullmatch('|'.join(systs))]
        fig, axs = plt.subplots(1, 3, figsize=(
            21, 7), sharex=True, sharey=True)
        for s,  ax in zip(systs, axs):
            for c, chl in enumerate((0, 10, 30, 50)):
                for sn, ls in zip(('sn-1', 'sn-2'),
                                  ('-', '--')):
                    ax.errorbar(x=dfol(scd_ms_part, s, sn, chl)['atom_n'],
                                y=dfol(scd_ms_part, s, sn, chl)['scd']['mean'],
                                yerr=dfol(scd_ms_part, s, sn, chl)[
                        'scd']['std'],
                        ls=ls, color=sns.color_palette('cubehelix')[c],
                        elinewidth=1, label=f'{chl} % {chls}, {sn}'
                    )
            s = to_rus[s] if rus else s
            ax.set_title(s)
            if rus:
                ax.set_xlabel('Номер атома углерода')
            else:
                ax.set_xlabel('C atom number')
        axs[0].set_ylabel('Scd')
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels)
        if rus:
            fig.suptitle(to_rus[exp])
        else:
            fig.suptitle(exp)
        plt.savefig(trj_slices[0].system.path / 'notebooks' / 'integral_parameters' /
                    f'{"_".join(exp.split())}_'
                    f'scd_{trj_slices[0].b}_{trj_slices[0].e}_{trj_slices[0].dt}{out}.png',
                    bbox_inches='tight')
        plt.close()

    path, b, e, dt = trj_slices[0].system.path, trj_slices[0].b, trj_slices[0].e, trj_slices[0].dt
    df = pd.read_csv(path / 'notebooks' /
                     'integral_parameters' / f'scd_{b}-{e}-{dt}.csv')
    # FIXME dropping dspc for now
    df = df[(df['system'] != 'dspc') | (
        (df['CHL amount, %'] != 10) & (df['CHL amount, %'] != 50))]
    df['atom_n'] = df['atom'].apply(lambda x: int(x[2:]))
    # df.sort_values(['system', 'CHL amount, %', 'chain', 'atom_n'], inplace=True)
    scd_ms = df.drop(columns=['timepoint', 'atom']).groupby(
        ['system', 'CHL amount, %', 'chain', 'atom_n'],
        as_index=False).agg(['mean', 'std'])
    scd_ms = scd_ms.reset_index(level=1).reset_index(
        level=1).reset_index(level=1).reset_index()

    print('plotting scd by experiment...')

    for exp, systs in experiments.items():
        scd_plot(scd_ms, exp, systs, to_rus, False)
        scd_plot(scd_ms, exp, systs, to_rus, True)
    print('done.')


def parse_args():
    '''
    arguments parser
    '''
    parser = argparse.ArgumentParser(
        description='Script to obtain integral parameters')
    parser.add_argument('--calculate',
                        nargs='+',
                        help='possible values:\n'
                        '"density" -- required for thickness and chl_p distances calculation,\n'
                        '"thickness" -- bilayer thickness,\n'
                        '"arperlip" -- area per lipid,\n'
                        '"scd" -- acyl chains order parameter,\n'
                        '"chl_tilt_angle" -- angle between CHL c3-c17 vector and bilayer plane,\n'
                        '"chl_p_distance" -- distances between CHL COM/O and phosphates.'
                        )
    parser.add_argument('--plot',
                        nargs='+',
                        help='possible values:\n'
                        '"dp" -- density profiles (for each system + CHL/phosphates comparison),\n'
                        '"scd_atoms" -- scd per atom for systems.')
    parser.add_argument('--all',
                        action='store_true',
                        help='start all "calculate" and "plot" tasks.')
    parser.add_argument('-b', '--b', type=int, default=150,
                        help='beginning time in ns')
    parser.add_argument('-e', '--e', type=int, default=200,
                        help='ending time in ns')
    parser.add_argument('-dt', '--dt', type=int, default=1000,
                        help='dt in ps')

    if len(sys.argv) < 2:
        parser.print_usage()

    return parser.parse_args()


@ sparkles
@ duration
def main():
    '''
    parse arguments and obtain and plot system parameters such as
    density profiles, area per lipid,thickness, Scd and cholesterol tilt angle
    '''
    sns.set(style='ticks', context='talk', palette='bright')
    args = parse_args()
    path = PATH
    experiments = EXPERIMENTS
    to_rus = TO_RUS

    systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in flatten(experiments.values())])

    # trj_slices = [TrajectorySlice(
    #     System(path, s), 150, 200, 1000) for s in systems]
    trj_slices = [TrajectorySlice(
        System(path, s), args.b, args.e, args.dt) for s in systems]

    to_calc = {'density': density,
               'thickness': thickness,
               'arperlip': arperlip,
               'scd': scd,
               # 'chl_tilt_angle': chl_tilt_angle,
               'chl_p_distance': chl_p_distance}

    to_plot = {'dp': density_profiles,
               'scd_atoms': plot_scd_atoms}

    if args.calculate is not None:
        for arg in args.calculate:
            to_calc.get(arg, lambda: 'Invalid')(
                trj_slices, experiments, to_rus)

    if args.plot is not None:
        for arg in args.plot:
            to_plot.get(arg, lambda: 'Invalid')(
                experiments, trj_slices, to_rus)

    if args.all:
        for i in to_calc:
            to_calc[i](trj_slices, experiments, to_rus)
        for i in to_plot:
            to_plot[i](experiments, trj_slices, to_rus)


# %%

if __name__ == '__main__':
    main()
