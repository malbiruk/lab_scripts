'''
functions used to obtain angles of CHL and break them into components
'''
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import axes
from scipy import integrate
from scipy.optimize import curve_fit
import MDAnalysis as mda

from modules.general import opener, flatten
from modules.traj import TrajectorySlice
from modules.constants import PATH
from integral_parameters_script import plot_violins


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

    guess = [-20, 0.005, 10, -15, 0.02, 7, 10, 0.02, 7, 20, 0.005, 10]
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
        a = np.array([i if i <= 90 else i - 180 for i in a])
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


def chl_tilt_angle(trj_slices: list[TrajectorySlice], no_comps=False) -> None:
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
        records = []
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

            comps = pd.read_csv(PATH / Path('notebooks/chol_tilt') /
                                f'{trj.system.name}_{trj.b}-{trj.e}-{trj.dt}_4_comps.csv')
            comps = comps.sort_values(['ctr']).reset_index(drop=True)
            records.append((trj.system.name.split('_chol', 1)[0],
                            trj.system.name.split('_chol', 1)[1],
                            np.mean(np.abs(comps.loc[[0, 3], 'ctr'])),
                            np.mean(np.abs(comps.loc[[0, 3], 'wid'])),
                            np.sum(comps.loc[[0, 3], 'area']),
                            np.mean(np.abs(comps.loc[[1, 2], 'ctr'])),
                            np.mean(np.abs(comps.loc[[1, 2], 'wid'])),
                            np.sum(comps.loc[[1, 2], 'area'])
                            ))

        df = pd.DataFrame.from_records(
            records, columns=['system', '% of CHL',
                              'horizontal_ctr', 'horizontal_wid', 'horizontal_area',
                              'vertical_ctr', 'vertical_wid', 'vertical_area'])

        df.to_csv(PATH / 'notebooks' / 'integral_parameters' / 'components' /
                  'angle_components_parameters_'
                  f'{ trj_slices[0].b}-{ trj_slices[0].e}-{ trj_slices[0].dt}.csv', index=False)
        print('plotting results...')

        plot_violins(path / 'notebooks' / 'integral_parameters' / 'chl_tilt_to_plot.csv',
                     'α, °')
    print('done.')
