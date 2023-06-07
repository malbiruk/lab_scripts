'''
useful functions to work with density and density profiles
'''
from pathlib import Path
from typing import Callable
import os
import numpy as np
import pandas as pd
from matplotlib import axes
from scipy.interpolate import make_interp_spline
import MDAnalysis as mda
from MDAnalysis.selections.gromacs import SelectionWriter

from modules.general import calc_1d_com
from modules.traj import System, TrajectorySlice


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


def plot_density_profile(ax,
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
            ax.plot(x_, y_, label=f'{label}', color=color)
        ax.set_xlabel('Z, nm')
        ax.set_ylabel('Density, kg/m³')


def calculate_distances_between_density_groups(
        trj: TrajectorySlice, grp1: str, grp2: str) -> list[float]:
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


def calculate_density_peak_widths(trj: TrajectorySlice, grp: str) -> list[float]:
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
