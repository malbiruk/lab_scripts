#!/usr/bin/python3

'''
obtain and plot system parameters such as density profiles, area per lipid,
thickness, Scd and cholesterol tilt angle
'''

from pathlib import Path, PosixPath
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable
import argparse
import os
import sys
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import axes
from scipy.interpolate import make_interp_spline
from scipy import integrate
from scipy.optimize import curve_fit
import MDAnalysis as mda
from MDAnalysis.selections.gromacs import SelectionWriter

from mhp_stats_script import flatten, sparkles, duration


def opener(inp: PosixPath) -> list[str]:
    '''
    open text file as list of lines
    '''
    with open(inp, 'r', encoding='utf-8') as f:
        lines = [i.strip() for i in f.read().strip().split('\n')]
    return lines


def multiproc(func: Callable, data: list, n_workers: int = 8) -> dict:
    '''
    wrapper for ProcessPoolExecutor,
    gets function, list of arguments (single argument) and max n of workers,
    gives dictionary {arguments: results}
    '''
    result = {}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(func, i): i for i in data}
        for f in as_completed(futures.keys()):
            result[futures[f]] = f.result()

    return result


@dataclass(frozen=True, unsafe_hash=True)
class System:
    '''
    stores system name and path
    '''
    path: PosixPath  # directory containing system directory
    name: str
    dir: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'dir', str(self.path / self.name))

    def __repr__(self):
        return f'System({self.name})'

    def __str__(self):
        return self.name

    def get_tpr_props(self) -> str:
        '''
        obtain properties of tpr of the system
        '''
        gmxutils = '/nfs/belka2/soft/impulse/dev/inst/gmx_utils.py'
        return os.popen(f'{gmxutils} {self.dir}/md/md.tpr').read()

    def get_n_chols(self) -> str:
        '''
        obtain number of cholesterol molecules
        '''
        return [i.strip() for i in
                [i for i in self.get_tpr_props().split('\n')
                 if i.startswith('CHOL')][0].split('|')][1]

    def resnames_from_systname(self) -> list[str]:
        '''
        obtain residue names of system (MDAnalysis format)
        '''
        no_numbers = ''.join([i for i in self.name if not i.isdigit()])
        return [i.upper() if not i == 'chol' else 'CHL' for i in no_numbers.split('_')]

    def pl_selector(self, n=0) -> str:
        '''
        obtain selector string of main phospholipid (impulse format)
        '''
        return f"'{self.name.split('_')[n].upper()}///'"


@dataclass(frozen=True, unsafe_hash=True)
class TrajectorySlice:
    '''
    stores beginnig (b), ending (e) timepoints (in ns)
    with dt (in ps) between them of system trajectory
    '''
    system: System
    b: int
    e: int
    dt: int

    def __repr__(self):
        return f'TrajectorySlice({self.system.name}, ' \
            f'b={self.b}, e={self.e}, dt={self.dt})'


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
        c3 = ' '.join(
            list(map(str, chols.select_atoms('name C3').indices.tolist())))
        c17 = ' '.join(
            list(map(str, chols.select_atoms('name C17').indices.tolist())))

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

    a = np.array(
        list(map(float, flatten([i.split()[1:] for i in lines])))) - 90

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

    guess = [-20, 0.005, 28, -6, 0.03, 4.5, 6, 0.03, 4.5, 24, 0.005, 28]
    try:
        popt, _, _, _, _ = curve_fit(func, x, y, p0=guess, full_output=True)
        df = pd.DataFrame(popt.reshape(int(len(guess) / 3), 3),
                          columns=['ctr', 'amp', 'wid'])
        df['area'] = df.apply(lambda row: integrate.quad(func, np.min(
            x), np.max(x), args=(row['ctr'], row['amp'], row['wid']))[0], axis=1)
        df.to_csv(
            f'{trj.system.path}/notebooks/chol_tilt/'
            f'{trj.system.name}_{trj.b}-{trj.e}-{trj.dt}_4_comps.csv', index=False)

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

    except RuntimeError as e:
        print(f'couldn\'t curve_fit for {trj.system}: ', e)


def get_densities(trj: TrajectorySlice) -> None:
    '''
    obtain densities of groups
    '''
    def create_index_files(system: System) -> None:
        '''
        create index files for groups
        '''
        def write_index_file(system: System,
                             ind: str,
                             group: mda.core.groups.AtomGroup,
                             lipids: mda.core.groups.AtomGroup) -> None:
            '''
            write index fie for current group
            '''
            with SelectionWriter(f'{system.dir}/density_profiles/{ind}.ndx') \
                    as write_ndx:
                write_ndx.write(lipids, name='LIPIDS')
                if ind == 'chols':
                    for i in group:
                        write_ndx.write(i)
                else:
                    write_ndx.write(group, name=ind)

        u = mda.Universe(f'{system.dir}/md/md.tpr',
                         f'{system.dir}/md/md.gro', refresh_offsets=True)

        mask = np.logical_or.reduce(
            ([u.residues.resnames == res for res in trj.system.resnames_from_systname()]))
        lipids = u.residues[mask].atoms

        chols = u.residues[u.residues.resnames == 'CHL']

        acyl_chains = sum((lip.atoms.select_atoms(
            'smarts [C;$(CCCCC)] or smarts C=C',
            rdkit_kwargs={'max_iter': 1000}).atoms
            for lip in (lipids - chols.atoms).residues))

        water = u.residues[u.residues.resnames == 'SOL'].atoms
        phosphates = (lipids - chols.atoms).select_atoms('smarts OP(O)(=O)O',
                                                         rdkit_kwargs={'max_iter': 1000}).atoms
        cholesterol_o = chols.atoms.select_atoms('element O').atoms
        individual_chols = [i.atoms for i in chols]

        if 'chol' in trj.system.name:
            write_index_file(system, 'chols', individual_chols, lipids)
            write_index_file(system, 'chols_o', cholesterol_o, lipids)
        write_index_file(system, 'acyl_chains', acyl_chains, lipids)
        write_index_file(system, 'phosphates', phosphates, lipids)
        write_index_file(system, 'water', water, lipids)

    def obt_dens(trj: TrajectorySlice, ind: str) -> None:
        '''
        obtain density for current group
        '''
        if (Path(trj.system.dir) / 'density_profiles' /
                f'{ind}_{trj.b}-{trj.e}-{trj.dt}_dp.xvg').is_file():
            print(f'{ind}_{trj.b}-{trj.e}-{trj.dt}_dp.xvg exists, skipping...')
        else:
            print(f'obtaining density of {ind}...')
            if 'chol' in trj.system.name:
                n_chol = trj.system.get_n_chols()
            cmd = ['source `ls -t /usr/local/gromacs*/bin/GMXRC | head -n 1 ` &&',
                   f'gmx density -s {trj.system.dir}/md/md.tpr',
                   f'-f {trj.system.dir}/md/pbcmol.xtc',
                   f'-n {trj.system.dir}/density_profiles/{ind}.ndx',
                   f'-b {trj.b*1000} -e {trj.e*1000} -dt {trj.dt} -sl 100',
                   f'-o {trj.system.dir}/density_profiles/{ind}_{trj.b}-{trj.e}-{trj.dt}_dp.xvg',
                   '-xvg none -center -symm']
            if ind == 'chols':
                numbers = ' '.join([str(i) for i in range(int(n_chol) + 1)])
                cmd.insert(1, f'echo {numbers} |')
                cmd.append(f'-ng {n_chol}')
            else:
                cmd.insert(1, 'echo 0 1 |')
            os.popen(' '.join(cmd)).read()

    groups = ['chols', 'chols_o', 'acyl_chains', 'phosphates', 'water']
    print(
        f'🗄️ system:\t{trj.system.name}\n⌚️ time:\t{trj.b}-{trj.e} ns, dt={trj.dt} ps')
    (Path(trj.system.dir) / 'density_profiles').mkdir(exist_ok=True)
    print('obtaining 💁 system 🏙️ information...')

    if not 'chol' in trj.system.name:
        for i in ('chols', 'chols_o'):
            if i in groups:
                groups.remove(i)

    if not np.all(
            [Path(f'{trj.system.dir}/density_profiles/{gr}.ndx').is_file()
             for gr in groups]):
        print('creating index files...')
        create_index_files(trj.system)

    else:
        print('skipping index files creation...')

    for gr in groups:
        obt_dens(trj, gr)

    print('done ✅\n')


def plot_density_profile(ax: axes._subplots.Axes,
                         trj: TrajectorySlice,
                         groups: list = None,
                         color: str = None,
                         label: str = None) -> None:
    '''
    plot density profile of system on single axes
    '''

    print('plotting dp for \n'
          f'🗄️ system:\t{trj.system}\n⌚️ time:\t{trj.b}-{trj.e} ns, dt={trj.dt} ps')
    if groups is None:
        groups = ['chols', 'chols_o', 'acyl_chains', 'phosphates', 'water'] \
            if 'chol' in trj.system.name else ['acyl_chains', 'phosphates', 'water']

    dfs = {gr: pd.read_csv(
        f'{trj.system.dir}/density_profiles/{gr}_{trj.b}-{trj.e}-{trj.dt}_dp.xvg',
        header=None, delim_whitespace=True) for gr in groups}

    # sum cholesterol profiles
    if 'chols' in groups:
        dfs['chols'][1] = dfs['chols'].iloc[:, 1:].sum(axis=1)
        dfs['chols'] = dfs['chols'].iloc[:, :2]

    for gr in groups[::-1]:
        x, y = dfs[gr][0], dfs[gr][1]
        x_y_spline = make_interp_spline(x, y)
        x_ = np.linspace(x.min(), x.max(), 500)
        y_ = x_y_spline(x_)
        if label is None:
            ax.plot(x_, y_, label=gr, color=color)
        else:
            ax.plot(x_, y_, label=f'{gr}, {label}', color=color)
        ax.set_xlabel('Z, nm')
        ax.set_ylabel('Density, kg/m³')


def calc_1d_com(x, m):
    '''
    calculate center of mass of 1D array
    '''
    return np.sum(x * m) / np.sum(m)


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


def calculate_distances_between_density_groups(
        grp1: str, grp2: str, trj: TrajectorySlice) -> list[float]:
    '''
    calculates distances between density groups in each step of trajectory part
    using densities of corresponding groups
    '''
    groups = ['chols', 'chols_o', 'acyl_chains', 'phosphates', 'water']
    if grp1 not in groups or grp2 not in groups:
        raise ValueError(f"invalid groups: '{grp1}' '{grp2}',"
                         " only 'chols', 'chols_o', 'acyl_chains', 'phosphates', 'water' "
                         'groups are supported')
    distances = []
    for fr in range(trj.b, trj.e, int(trj.dt / 1000)):
        df1 = pd.read_csv(f'{trj.system.dir}/density_profiles/{grp1}_{fr}-{fr}-0_dp.xvg',
                          header=None, delim_whitespace=True)
        df2 = pd.read_csv(f'{trj.system.dir}/density_profiles/{grp2}_{fr}-{fr}-0_dp.xvg',
                          header=None, delim_whitespace=True)
        x, y1, y2 = df1[0], df1[1], df2[1]
        x_y_spline1, x_y_spline2 = make_interp_spline(
            x, y1),  make_interp_spline(x, y2)
        x_ = np.linspace(x.min(), x.max(), 500)

        grps = dict(
            grp1_com_pos=calc_1d_com(x_[x_ > 0], x_y_spline1(x_[x_ > 0])),
            grp2_com_pos=calc_1d_com(x_[x_ > 0], x_y_spline2(x_[x_ > 0])),
            grp1_com_neg=calc_1d_com(x_[x_ < 0], x_y_spline1(x_[x_ < 0])),
            grp2_com_neg=calc_1d_com(x_[x_ < 0], x_y_spline2(x_[x_ < 0])))

        distances.append(grps['grp2_com_pos'] - grps['grp1_com_pos'])
        distances.append(grps['grp1_com_neg'] - grps['grp2_com_neg'])
    return distances


def calculate_density_peak_widths(grp: str, trj: TrajectorySlice) -> list[float]:
    '''
    peak width of density is calculated by
    obtaining x value of yCOM
    after that, minimum value ymin between x=0 and that x is found
    then we are finding y value in the center ymin and yCOM
    and finding peak width aka difference between x values where
    density profile is intersecting this y
    '''
    def calc_single_peak_width(x: np.ndarray, spline: Callable) -> float:
        '''
        calculate single peak width
        '''
        y = spline(x)
        xcom = calc_1d_com(x, y)
        ycom = spline(xcom)
        x, y = np.abs(x), np.abs(y)
        for i in (x, y, np.array(xcom), np.array(ycom)):
            np.abs(i, out=i)

        y_middle = np.mean([ycom, np.min(y[x < xcom])])
        res = x[np.isclose(np.array([y_middle for _ in y]), y, rtol=.07)]
        if len(res) >= 2:
            return res[np.argmax(np.diff(res)) + 1] - res[np.argmax(np.diff(res))]
        raise IndexError(
            f'length of array should be equal 2, got {len(res)}')

    if grp not in ['chols', 'chols_o', 'acyl_chains', 'phosphates', 'water']:
        raise ValueError(f"invalid group '{grp}',"
                         " only 'chols', 'chols_o', 'acyl_chains', 'phosphates', 'water' "
                         'groups are supported')

    peak_widths = []
    for fr in range(trj.b, trj.e, int(trj.dt / 1000)):
        df = pd.read_csv(f'{trj.system.dir}/density_profiles/{grp}_{fr}-{fr}-0_dp.xvg',
                         header=None, delim_whitespace=True)
        if grp == 'chols':
            df[1] = df.iloc[:, 1:].sum(axis=1)
            df = df.iloc[:, :2]
        x, y = df[0], df[1]
        x_y_spline = make_interp_spline(x, y)
        x_ = np.linspace(x.min(), x.max(), 500)
        try:
            peak_widths.append(calc_single_peak_width(x_[x_ > 0], x_y_spline))
            peak_widths.append(calc_single_peak_width(x_[x_ < 0], x_y_spline))

        # FIXME: errors are occuring :c
        except ValueError as e:
            print(trj.system.name, fr)
            print(e)
    return peak_widths


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
    # df.to_csv(scd_folder / 'scd_all_new.csv', index=False)
    #
    # print('summary table for atoms...')
    # df = pd.read_csv(scd_folder / 'scd_all_new.csv')
    # atom_summary = df.groupby(['system', 'atom'], as_index=False).agg(
    #     'mean').rename({'scd': 'mean'}, axis=1)
    # atom_summary = atom_summary.assign(std=df.groupby(
    #     ['system', 'atom']).agg(np.std)['scd'].values)
    # atom_summary.to_csv(scd_folder / 'scd_atoms.csv', index=False)
    #
    # print('summary table for chains...')
    # df = pd.read_csv(scd_folder / 'scd_all_new.csv')
    # chain_summary = df.groupby(['system', 'chain'], as_index=False).agg(
    #     'mean').rename({'scd': 'mean'}, axis=1)
    # chain_summary = chain_summary.assign(std=df.groupby(
    #     ['system', 'chain']).agg(np.std)['scd'].values)
    # chain_summary.to_csv(scd_folder / 'scd_chains.csv', index=False)


# def integral_summary(infile: PosixPath,
#                      outfile: PosixPath,
#                      index=None) -> None:
#     '''
#     move chol rows to new columns and calculate relative_changes
#     '''
#     def move_chol_rows_to_new_columns(
#             infile: PosixPath,
#             outfile: PosixPath,
#             index=None) -> None:
#         '''
#         system       | val          system | chol0 | chol10 | chol30 | chol50
#         popc            1           popc       1        2        3        4
#         popc_chol10     2      ->
#         popc_chol30     3
#         popc_chol50     4
#
#         concat dataframes by 'index' argument
#         '''
#
#         if index is None:
#             index = ['system']
#
#         df = pd.read_csv(infile)
#         df.sort_values('system', inplace=True, ignore_index=True)
#
#         df_parts = []
#         for i in [10, 30, 50]:
#             df_part = df[df['system'].str.contains(
#                 f'_chol{i}')].copy().reset_index(drop=True)
#             df_part['system'] = pd.Series(
#                 ['_'.join(i.split('_')[:-1]) for i in df_part.loc[:, 'system']])
#             df_part.columns = ['system'] + \
#                 [f'{name}_chol{i}' if not name in index else name for name in df_part.columns[1:]]
#             df_parts.append(df_part)
#
#         df.drop(df[df['system'].str.contains('chol')].index, inplace=True)
#         df.reset_index(drop=True, inplace=True)
#
#         df_concat = pd.concat([df.set_index(index)] +
#                               [i.set_index(index) for i in df_parts], axis=1)
#
#         df_concat.to_csv(outfile)
#
#     def calculate_relative_changes(infile: PosixPath, n_of_index_cols: int = 1) -> None:
#         '''
#         calculate relative changes for parameters and save them as new file in the same directory
#         '''
#         df = pd.read_csv(infile, index_col=list(range(n_of_index_cols)))
#         df2 = pd.DataFrame(index=df.index)
#         for i in [10, 30, 50]:
#             df2[f'mean_chol{i}'] = (
#                 df[f'mean_chol{i}'] - df['mean']) / df['mean'] * 100
#             df2[f'std_chol{i}'] = np.sqrt(np.abs(
#                 (df[f'std_chol{i}']**2 * df['mean']**2
#                  - df['std']**2 * df[f'mean_chol{i}']**2) / df['mean']**4)) * 100
#         df2.to_csv(str(infile).split('.', maxsplit=1)
#                    [0] + '_relative_changes.csv')
#     move_chol_rows_to_new_columns(infile, outfile, index)
#     if index is None:
#         calculate_relative_changes(outfile, 1)
#     else:
#         calculate_relative_changes(outfile, len(index))


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


def density(trj_slices: list[TrajectorySlice]) -> None:
    '''
    apply get_densities function to list of trajectories
    '''
    print('obtain all densities...')
    with ProcessPoolExecutor(max_workers=8) as executor:
        executor.map(get_densities, trj_slices)
    print('done.')


def plot_violins(csv: PosixPath, y: str, x: str = 'system',
                 hue: str = 'CHL amount, %', split=False) -> None:
    '''
    plot violinplot for distribution of parameters
    '''
    df = pd.read_csv(csv)
    _, _ = plt.subplots(figsize=(10, 7))
    sns.violinplot(data=df, x=x, y=y, hue=hue,
                   cut=0, palette='RdYlGn_r', split=split, inner='quartile')
    plt.savefig(str(csv).split('.', 1)[0] + '.png',
                bbox_inches='tight')
    plt.close()


def thickness(trj_slices: list[TrajectorySlice]) -> None:
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
                 'thickness, nm')
    print('done.')


def arperlip(trj_slices: list[TrajectorySlice]) -> None:
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
                 'area per lipid, nm²')
    print('done.')


def scd(trj_slices: list[TrajectorySlice]) -> None:
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
                 'scd')
    print('done.')


def chl_tilt_summary(trj_slices: list[TrajectorySlice]) -> None:
    '''
    aggregate chl tilt data of several systems into one file
    '''
    records = []
    for trj in trj_slices:
        lines = opener(trj_slices[0].system.path / 'notebooks' / 'chol_tilt' /
                       f'{trj.system.name}_{trj.b}-{trj.e}-{trj.dt}_tilt.xvg')
        a = np.array(
            list(map(float, flatten([i.split()[1:] for i in lines])))) - 90
        records.append((trj.system.name, a))
    df = pd.DataFrame.from_records(
        records, columns=['system', 'α, °'])
    df.sort_values('system', inplace=True, ignore_index=True)
    df['CHL amount, %'] = df['system'].str.split('_chol', n=1, expand=True)[1]
    df['system'] = df['system'].str.split('_chol', n=1, expand=True)[0]
    df.replace(to_replace=[None], value=0, inplace=True)
    return df.explode('α, °')


def chl_tilt_angle(trj_slices: list[TrajectorySlice]) -> None:
    '''
    apply get_chl_tilt function to list of trajectories,
    split each system into components (plot + save parameters),
    unite data of several systems into one file
    and plot results as violinplot
    '''
    trj_slices = [s for s in trj_slices if 'chol' in s.system.name]
    path, b, e, dt = trj_slices[0].system.path, trj_slices[0].b, trj_slices[0].e, trj_slices[0].dt
    print('obtaining cholesterol tilt...')
    with ProcessPoolExecutor(max_workers=8) as executor:
        executor.map(get_chl_tilt, trj_slices)
    print('saving chl tilt angles...')
    chl_tilt_summary(trj_slices).to_csv(
        path / 'notebooks' / 'integral_parameters' / f'chl_tilt_{b}-{e}-{dt}.csv', index=False)
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
    plot_violins(path / 'notebooks' / 'integral_parameters' / f'chl_tilt_{b}-{e}-{dt}.csv',
                 'α, °')
    print('done.')


def chl_p_distance(trj_slices: list[TrajectorySlice]) -> None:
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
                 f'chols_phosphates_distances_{b}-{e}-{dt}.csv', 'distance, nm')
    plot_violins(path / 'notebooks' / 'integral_parameters' /
                 f'chols_o_phosphates_distances_{b}-{e}-{dt}.csv', 'distance, nm')
    print('done.')


def plot_dp_by_exp(experiments, trj_slices: list[TrajectorySlice]) -> None:
    '''
    plots chol chol_o amd phosphates density profiles on same axis for
    systems with different amounts of CHL.
    plots systems from the same experiment on one figure
    '''
    trj_slices_chol = [s for s in trj_slices if 'chol' in s.system.name]
    path, b, e, dt = trj_slices[0].system.path, trj_slices[0].b, trj_slices[0].e, trj_slices[0].dt
    for exp, systs in experiments.items():
        fig, axs = plt.subplots(1, 3, figsize=(
            21, 7), sharex=True, sharey=True)
        reds = sns.color_palette('Reds_r', 3)
        purples = sns.color_palette('Purples_r', 3)
        blues = sns.color_palette('Blues_r', 3)
        plotted = []
        for syst, ax in zip(systs, axs):
            c = 0  # for choosing colors
            for trj in trj_slices_chol:
                if trj.system.name.rsplit('_', 1)[0] == syst and str(trj.system) not in plotted:
                    l = f'{trj.system.name.split("chol", 1)[1]} % CHL'
                    plot_density_profile(
                        ax, trj, groups=['chols'], color=purples[c], label=l)
                    plot_density_profile(
                        ax, trj, groups=['chols_o'], color=blues[c], label=l)
                    plot_density_profile(
                        ax, trj, groups=['phosphates'], color=reds[c], label=l)
                    plotted.append(str(trj.system))
                    ax.set_title(trj.system.name.rsplit('_', 1)[0])
                    c += 1
        for i in axs[1:]:
            i.set_ylabel('')
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels)
        fig.suptitle(exp, fontsize=16)
        plt.savefig(path / 'notebooks' / 'integral_parameters' /
                    f'{"_".join(exp.split())}_dp_{b}_{e}_{dt}.png',
                    bbox_inches='tight')
        plt.close()


def density_profiles(experiments: dict, trj_slices: list[TrajectorySlice]) -> None:
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
                 'peak width, nm')
    print('plotting dp by experiment...')
    plot_dp_by_exp(experiments, trj_slices)
    print('done.')


def scd_atoms():
    pass


def angles_density():
    pass

# %%

# TODO: plotting: scd
# angles, angles + densities (horizontal component percentage)
# reformat parse args and main:
# parse args to one list
# in help list of available values
# in main make dict = {key_from_list: function}
# then use dict.get()


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
                        '"scd_atoms" -- scd per atom for systems,\n'
                        '"angles_density" -- density profiles with percentage '
                        'of horizontal component.')
    parser.add_argument('--b', type=int, default=150,
                        help='beginning time in ns')
    parser.add_argument('--e', type=int, default=200,
                        help='ending time in ns')
    parser.add_argument('--dt', type=int, default=1000,
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
    plt.style.use('seaborn-talk')
    args = parse_args()
    path = Path('/home/klim/Documents/chol_impact/')
    experiments = {
        'chain length': ('dmpc', 'dppc_325', 'dspc'),
        'chain saturation': ('dppc_325', 'popc', 'dopc'),
        'head polarity': ('dopc', 'dopc_dops50', 'dops'),
    }
    systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in flatten(experiments.values())])
    systems.remove('dopc_dops50_chol50')
    systems.remove('dopc_dops50')

    trj_slices = [TrajectorySlice(
        System(path, s), args.b, args.e, args.dt) for s in systems]
    # FIXME: delete after dopc_dops50 will be calculated
    trj_slices.append(TrajectorySlice(
        System(path, 'dopc_dops50'), 4, 54, 1000))

    to_calc = {'density': density,
               'thickness': thickness,
               'arperlip': arperlip,
               'scd': scd,
               'chl_tilt_angle': chl_tilt_angle,
               'chl_p_distance': chl_p_distance}

    to_plot = {'dp': density_profiles,
               'scd_atoms': scd_atoms,
               'angles_density': angles_density}

    if args.calculate is not None:
        for arg in args.calculate:
            to_calc.get(arg, lambda: 'Invalid')(trj_slices)

    if args.plot is not None:
        for arg in args.plot:
            to_plot.get(arg, lambda: 'Invalid')(experiments, trj_slices)

    #
    # if args.chl_peak_width:
    #     print('obtaining chols peak widths...')
    #     lists_of_values_to_df(density_peak_widths_chols, trj_slices_chol).to_csv(
    #         path / 'notebooks' / 'chl_peak_widths' / 'chols_peak_widths.csv', index=False)
    #     print('done.')
    #
    # if args.plot_dps:
    #     print('plotting density profiles...')
    #     sns.set_palette('bright')
    #     for trj in trj_slices:
    #         fig, ax=plt.subplots(figsize=(7, 7))
    #         plot_density_profile(ax, trj)
    #         fig.patch.set_facecolor('white')
    #         plt.savefig(f'{path}/notebooks/dp/'
    #                     f'{trj.system}_{trj.b}-{trj.e}-{trj.dt}_dp.png',
    #                     bbox_inches='tight', facecolor=fig.get_facecolor())
    #         plt.close()


if __name__ == '__main__':
    main()
