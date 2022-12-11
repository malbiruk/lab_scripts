#!/usr/bin/python3

'''
obtain and plot system parameters such as density profiles, area per lipid,
thickness, Scd and cholesterol tilt angle
'''

from pathlib import Path, PosixPath
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable
import argparse
import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import axes
from scipy.interpolate import make_interp_spline
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


def multiproc(func: Callable, data: list, n_workers: int) -> dict:
    '''
    wrapper for ProcessPoolExecutor,
    gets function, list of arguments and max n of workers,
    gives dictionary {arguments: results}
    '''
    result = {}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_addr = {executor.submit(func, i): i
                          for i in data}
        for f in as_completed(future_to_addr.keys()):
            result[future_to_addr[f]] = f.result()

    return result


@dataclass(frozen=True, unsafe_hash=True)
class System:
    '''
    stores system name and path
    '''
    path: PosixPath  # directory containing system directory
    name: str

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
        obtain residue names of system
        '''
        no_numbers = ''.join([i for i in self.name if not i.isdigit()])
        return [i.upper() if not i == 'chol' else 'CHL' for i in no_numbers.split('_')]


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
    print('obtaining 💁 system 🏙️ information...')
    u = mda.Universe(f'{trj.system.dir}/md/md.tpr',
                     f'{trj.system.dir}/md/pbcmol.xtc',
                     refresh_offsets=True)
    chols = u.residues[u.residues.resnames == 'CHL'].atoms
    n_chol = len(u.residues[u.residues.resnames == 'CHL'])
    c3 = chols.select_atoms('name C3')
    c17 = chols.select_atoms('name C17')

    with SelectionWriter(f'{trj.system.dir}/ch3_ch17.ndx') as write_ndx:
        write_ndx.write(c3, name='C3')
        write_ndx.write(c17, name='C17')

    print('calculating 👨‍💻 cholesterol 🫀 tilt 📐 ...')

    cmd = ['source `ls -t /usr/local/gromacs*/bin/GMXRC | head -n 1 `',
           f'gmx bundle -s {trj.system.dir}/md/md.tpr',
           f'-f {trj.system.dir}/md/pbcmol.xtc',
           f'-na {n_chol} -z -n {trj.system.dir}/ch3_ch17.ndx',
           f'-b {trj.b*1000} -e {trj.e * 1000} -dt {trj.dt}',
           f'-ot {str(trj.system.path)}/notebooks/chol_tilt/'
           f'{trj.system}_{trj.b}-{trj.e}-{trj.dt}_tilt.xvg',
           '-xvg none']

    os.popen(' '.join(cmd)).read()
    print('done ✅\n')


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
                         f'{system.dir}/md/pbcmol.xtc', refresh_offsets=True)

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


def plot_density_profile(ax: axes._subplots.Axes, trj: TrajectorySlice) -> None:
    '''
    plot density profile of system on single axes
    '''

    print('plotting dp for \n'
          f'🗄️ system:\t{trj.system}\n⌚️ time:\t{trj.b}-{trj.e} ns, dt={trj.dt} ps')
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
        ax.plot(x_, y_, label=gr)
        ax.legend()
        ax.set_title(trj.system.name)
        ax.set_xlabel('Z, nm')
        ax.set_ylabel('Density, kg/m³')


def calc_1d_com(x, m):
    '''
    calculate center of mass of 1D array
    '''
    return np.sum(x * m) / np.sum(m)


def calculate_thickness(trj: TrajectorySlice) -> list[float]:
    '''
    calculates thickness in each step of trajectory using phosphates densities
    '''
    thickness = []
    for fr in range(trj.b, trj.e, int(trj.dt / 1000)):
        get_densities(TrajectorySlice(trj.system, fr, fr, 0))
        df = pd.read_csv(f'{trj.system.dir}/density_profiles/phosphates_{fr}-{fr}-0_dp.xvg',
                         header=None, delim_whitespace=True)
        x, y = df[0], df[1]
        x_y_spline = make_interp_spline(x, y)
        x_ = np.linspace(x.min(), x.max(), 500)
        thickness.append(
            calc_1d_com(x_[x_ > 0], x_y_spline(x_[x_ > 0]))
            - calc_1d_com(x_[x_ < 0], x_y_spline(x_[x_ < 0])))
    return thickness


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


@sparkles
@duration
def main():
    '''
    parse arguments and obtain and plot system parameters such as
    density profiles, area per lipid,thickness, Scd and cholesterol tilt angle
    '''

    parser = argparse.ArgumentParser(
        description='Script to obtain integral parameters')
    parser.add_argument('--obtain_densities',
                        action='store_true',
                        help='obtain density data')
    parser.add_argument('--obtain_thickness',
                        action='store_true',
                        help='obtain thickness data')
    parser.add_argument('--obtain_arperlip',
                        action='store_true',
                        help='obtain area per lipid data')
    parser.add_argument('--dt', type=int, default=1000,
                        help='dt in ps')
    parser.add_argument('--b', type=int, default=150,
                        help='beginning time in ns')
    parser.add_argument('--e', type=int, default=200,
                        help='ending time in ns')

    if len(sys.argv) < 2:
        parser.print_usage()

    args = parser.parse_args()

    plt.style.use('seaborn-talk')

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

    if args.obtain_densities:
        print('obtain all densities')
    if args.obtain_thickness:
        thicknesses = []
    if args.obtain_arperlip:
        arperlips = []

    trj_slices = [TrajectorySlice(System(path, s), args.b, args.e, args.dt) for s in systems]

    if args.obtain_densities:
        with ProcessPoolExecutor(max_workers=8) as executor:
            executor.map(get_densities, trj_slices)

    if args.obtain_thickness:
        for trj, th in multiproc(calculate_thickness, trj_slices, 8).items():
            thicknesses.append((trj.system.name, np.mean(th), np.std(th)))

    if args.obtain_arperlip:
        for trj, arpl in multiproc(calculate_area_per_lipid, trj_slices, 8).items():
            arperlips.append((trj.system.name, np.mean(arpl), np.std(arpl)))

    if args.obtain_thickness:
        print('saving thickness...')
        thick_df = pd.DataFrame.from_records(
            thicknesses, columns=['system', 'mean', 'std'])
        thick_df.to_csv(path / 'notebooks' / 'thickness' /
                        'new_thickness.csv', index=False)
        print('done.')

    if args.obtain_arperlip:
        print('saving area per lipid...')
        arperlip_df = pd.DataFrame.from_records(
            arperlips, columns=['system', 'mean', 'std'])
        arperlip_df.to_csv(path / 'notebooks' /
                           'area_per_lipid' / 'new_arperlip.csv', index=False)
        print('done.')


if __name__ == '__main__':
    main()
