#!/usr/bin/python3


from pathlib import Path
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import MDAnalysis as mda
from MDAnalysis.selections.gromacs import SelectionWriter

from MHP_stats_script import flatten, sparkles, duration


def get_tpr_props(path, syst):
    gmxutils = '/nfs/belka2/soft/impulse/dev/inst/gmx_utils.py'
    return os.popen(f'{gmxutils} {path}/{syst}/md/md.tpr').read()


def opener(inp):
    with open(inp, 'r') as f:
        lines = [i.strip() for i in f.read().strip().split('\n')]
    return lines


def get_chl_tilt(path, syst, b, e, dt):
    print(f'🗄️ system:\t{syst}\n⌚️ time:\t{b}-{e} ns, dt={dt} ps')
    print('obtaining 💁 system 🏙️ information...')
    tpr_props = get_tpr_props(path, syst)
    n_chol = [i.strip() for i in
              [i for i in tpr_props.split('\n') if i.startswith('CHOL')][0].split('|')][1]
    u = mda.Universe(f'{path}/{syst}/md/md.tpr',
                     f'{path}/{syst}/md/pbcmol.xtc', refresh_offsets=True)
    chols = u.residues[u.residues.resnames == 'CHL'].atoms

    # c3 = ' '.join(
    #     list(map(str, chols.select_atoms('name C3').indices.tolist())))
    # c17 = ' '.join(
    #     list(map(str, chols.select_atoms('name C17').indices.tolist())))

    c3 = chols.select_atoms('name C3')
    c17 = chols.select_atoms('name C17')

    with SelectionWriter(f'{path}/{syst}/ch3_ch17.ndx') as write_ndx:
        write_ndx.write(c3, name='C3')
        write_ndx.write(c17, name='C17')

    # with open(f'{path}/notebooks/chol_tilt/ch.ndx', 'w') as f:
    #     f.write(f'[C3]\n{c3}\n[C17]\n{c17}\n')

    print('calculating 👨‍💻 cholesterol 🫀 tilt 📐 ...')

    cmd = ['source `ls -t /usr/local/gromacs*/bin/GMXRC | head -n 1 `',
           f'gmx bundle -s {path}/{syst}/md/md.tpr',
           f'-f {path}/{syst}/md/pbcmol.xtc',
           f'-na {n_chol} -z -n {path}/{syst}/ch3_ch17.ndx',
           f'-b {b}000 -e {e}000 -dt {dt}',
           f'-ot {path}/notebooks/chol_tilt/{syst}_{b}-{e}-{dt}_tilt.xvg',
           '-xvg none']

    os.popen(' '.join(cmd)).read()
    lines = opener(f'{path}/notebooks/chol_tilt/{syst}_{b}-{e}-{dt}_tilt.xvg')
    arr = np.array(
        list(map(float, flatten([i.split()[1:] for i in lines]))))
    np.savetxt(f'{path}/notebooks/chol_tilt/{syst}_{b}-{e}-{dt}_tilt.txt', arr)
    print('done ✅\n')


def resnames_from_systname(syst):
    no_numbers = ''.join([i for i in syst if not i.isdigit()])
    return [i.upper() if not i == 'chol' else 'CHL' for i in no_numbers.split('_')]


def get_densities(path, syst, b, e, dt):
    groups = ['chols', 'chols_o', 'acyl_chains', 'phosphates', 'water']
    print(f'🗄️ system:\t{syst}\n⌚️ time:\t{b}-{e} ns, dt={dt} ps')
    (path / syst / 'density_profiles').mkdir(exist_ok=True)
    print('obtaining 💁 system 🏙️ information...')

    if 'chol' in syst:
        tpr_props = get_tpr_props(path, syst)
        n_chol = [i.strip() for i in
                  [i for i in tpr_props.split('\n')
                  if i.startswith('CHOL')][0].split('|')][1]
    else:
        for i in ('chols', 'chols_o'):
            if i in groups:
                groups.remove(i)

    if not np.all([Path(f'{path}/{syst}/density_profiles/{gr}.ndx').is_file() for gr in groups]):
        print('creating index files...')

        u = mda.Universe(f'{path}/{syst}/md/md.tpr',
                         f'{path}/{syst}/md/pbcmol.xtc', refresh_offsets=True)

        mask = np.logical_or.reduce(
            ([u.residues.resnames == res for res in resnames_from_systname(syst)]))
        lipids = u.residues[mask].atoms

        chols = u.residues[u.residues.resnames == 'CHL']

        acyl_chains = sum([lip.atoms.select_atoms(
            'smarts [C;$(CCCCC)] or smarts C=C',
            rdkit_kwargs={'max_iter': 1000}).atoms
            for lip in (lipids - chols.atoms).residues])

        water = u.residues[u.residues.resnames == 'SOL'].atoms
        phosphates = (lipids - chols.atoms).select_atoms('smarts OP(O)(=O)O',
                                                         rdkit_kwargs={'max_iter': 1000}).atoms
        cholesterol_o = chols.atoms.select_atoms('element O').atoms
        individual_chols = [i.atoms for i in chols]

        # create index files:
        def create_index_file(path, syst, ind, group, lipids):
            with SelectionWriter(f'{path}/{syst}/density_profiles/{ind}.ndx') \
                    as write_ndx:
                write_ndx.write(lipids, name='LIPIDS')
                if ind == 'chols':
                    for i in group:
                        write_ndx.write(i)
                else:
                    write_ndx.write(group, name=ind)

        if 'chol' in syst:
            create_index_file(path, syst, 'chols', individual_chols, lipids)
            create_index_file(path, syst, 'chols_o', cholesterol_o, lipids)
        create_index_file(path, syst, 'acyl_chains', acyl_chains, lipids)
        create_index_file(path, syst, 'phosphates', phosphates, lipids)
        create_index_file(path, syst, 'water', water, lipids)

    else:
        print('skipping index files creation...')

    def obt_dens(path, syst, b, e, dt, ind):
        if (path / syst / 'density_profiles' /
                f'{ind}_{b}-{e}-{dt}_dp.xvg').is_file():
            print(f'{ind}_{b}-{e}-{dt}_dp.xvg exists, skipping...')
        else:
            print(f'obtaining density of {ind}...')
            cmd = ['source `ls -t /usr/local/gromacs*/bin/GMXRC | head -n 1 ` &&',
                   f'gmx density -s {path}/{syst}/md/md.tpr',
                   f'-f {path}/{syst}/md/pbcmol.xtc',
                   f'-n {path}/{syst}/density_profiles/{ind}.ndx',
                   f'-b {b}000 -e {e}000 -dt {dt} -sl 100',
                   f'-o {path}/{syst}/density_profiles/{ind}_{b}-{e}-{dt}_dp.xvg',
                   '-xvg none -center -symm']
            if ind == 'chols':
                numbers = ' '.join([str(i) for i in range(int(n_chol) + 1)])
                cmd.insert(1, f'echo {numbers} |')
                cmd.append(f'-ng {n_chol}')
            else:
                cmd.insert(1, 'echo 0 1 |')
            os.popen(' '.join(cmd)).read()

    for gr in groups:
        obt_dens(path, syst, b, e, dt, gr)

    print('done ✅\n')


def plot_density_profile(ax, path, syst, b, e, dt):
    print('plotting dp for \n'
          f'🗄️ system:\t{syst}\n⌚️ time:\t{b}-{e} ns, dt={dt} ps')
    groups = ['chols', 'chols_o', 'acyl_chains', 'phosphates', 'water'] \
        if 'chol' in syst else ['acyl_chains', 'phosphates', 'water']

    dfs = {gr: pd.read_csv(f'{path}/{syst}/density_profiles/{gr}_{b}-{e}-{dt}_dp.xvg',
                           header=None, delim_whitespace=True) for gr in groups}

    # sum cholesterol profiles
    if 'chols' in groups:
        dfs['chols'][1] = dfs['chols'].iloc[:, 1:].sum(axis=1)
        dfs['chols'] = dfs['chols'].iloc[:, :2]

    for gr in groups[::-1]:
        x, y = dfs[gr][0], dfs[gr][1]
        x_y_Spline = make_interp_spline(x, y)
        x_ = np.linspace(x.min(), x.max(), 500)
        y_ = x_y_Spline(x_)
        ax.plot(x_, y_, label=gr)
        ax.legend()
        ax.set_title(f'{syst}')
        ax.set_xlabel('Z, nm')
        ax.set_ylabel('Density, kg/m³')


def calc_1d_com(x, m):
    return np.sum(x * m) / np.sum(m)


def calculate_thickness(path, syst, b, e, dt):
    thickness = []
    for fr in range(b, e, int(dt / 1000)):
        get_densities(path, syst, fr, fr, 0)
        df = pd.read_csv(f'{path}/{syst}/density_profiles/phosphates_{fr}-{fr}-0_dp.xvg',
                         header=None, delim_whitespace=True)
        x, y = df[0], df[1]
        x_y_Spline = make_interp_spline(x, y)
        x_ = np.linspace(x.min(), x.max(), 500)
        thickness.append(
            calc_1d_com(x_[x_ > 0], x_y_Spline(x_[x_ > 0]))
            - calc_1d_com(x_[x_ < 0], x_y_Spline(x_[x_ < 0])))
    return thickness


def calculate_area_per_lipid(path, syst, b, e, dt):
    if not Path(f'{path}/{syst}/md/box.xvg').is_file():
        cmd = ['source `ls -t /usr/local/gromacs*/bin/GMXRC | head -n 1 ` &&',
               f'echo 0 | gmx traj -f {path}/{syst}/md/pbcmol.xtc',
               f'-s {path}/{syst}/md/md.tpr -noz',
               f'-ob {path}/{syst}/md/box.xvg -xvg none -dt {dt} -b {b}000 -e {e}000']

        os.popen(' '.join(cmd)).read()

    box = pd.read_csv(f'{path}/{syst}/md/box.xvg',
                      header=None, delim_whitespace=True, usecols=[1, 2], names=['x', 'y'])

    box['total area'] = box['x'] * box['y']
    box['area per lipid'] = box['total area'].apply(
        lambda x: x / 400 if '20x20' in syst else x / 100)

    return box['area per lipid'].tolist()


@sparkles
@duration
def main():
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
                       for exp in experiments for i in experiments[exp]])

    systems.remove('dopc_dops50_chol50')
    systems.remove('dopc_dops50')

    b, e, dt = args.b, args.e, args.dt

    if args.obtain_densities:
        print('obtain all densities')
    if args.obtain_thickness:
        thicknesses = []
    if args.obtain_arperlip:
        arperlips = []

    for syst in systems:
        if args.obtain_densities:
            get_densities(path, syst, b, e, dt)
        if args.obtain_thickness:
            th = calculate_thickness(path, syst, b, e, dt)
            thicknesses.append((syst, np.mean(th), np.std(th)))
        if args.obtain_arperlip:
            arpl = calculate_area_per_lipid(path, syst, b, e, dt)
            arperlips.append((syst, np.mean(arpl), np.std(arpl)))

    if args.obtain_thickness:
        print('saving thickness...')
        thick_df = pd.DataFrame.from_records(
            thicknesses, columns=['system', 'mean', 'std'])
        thick_df.to_csv(path / 'notebooks' / 'thickness' / 'new_thickness.csv')
        print('done.')

    if args.obtain_arperlip:
        print('saving area per lipid...')
        arperlip_df = pd.DataFrame.from_records(
            arperlips, columns=['system', 'mean', 'std'])
        arperlip_df.to_csv(path / 'notebooks' /
                           'area_per_lipid' / 'new_arperlip.csv')
        print('done')


if __name__ == '__main__':
    main()
