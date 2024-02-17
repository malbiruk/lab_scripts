'''
this script calculates scd of PL contacting with CHL and not contacting
'''
import logging
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
from contacts_new import get_n_pl_df
from MDAnalysis.analysis import leaflet
from modules.constants import EXPERIMENTS, PATH
from modules.general import flatten, initialize_logging, multiproc, opener
from modules.traj import System, TrajectorySlice


def run_scd_py(trj: TrajectorySlice, selection: str, postfix: str) -> None:
    '''
    calculates per atom Scd values for whole trajectory part
    for molecules in selection
    '''
    if (PATH / 'notebooks' / 'scd' / 'near_chl' /
            f'{trj.system.name}_{postfix}_{trj.b}-{trj.e}-0_scd.csv').is_file():
        return

    cmd = ['/nfs/belka2/soft/impulse/dev/inst/scd.py',
           f'-f {trj.system.dir}/md/{trj.system.xtc}',
           f'-s {trj.system.dir}/md/{trj.system.tpr}',
           f'-o {trj.system.path}/notebooks/scd/near_chl/'
           f'{trj.system.name}_{postfix}_{trj.b}-{trj.e}-{trj.dt}',
           f'-b {trj.b} -e {trj.e} --dt {trj.dt} '
           f'--sel {selection}']
    # logging.info(
    #     'calculating Scd for %s, '
    #     '%s-%s ns, dt=%s ps...',
    #     trj.system.name, trj.b, trj.e, trj.dt)

    if len(selection[1:-2].split(',')) < 2:
        return

    try:
        subprocess.run(' '.join(cmd), shell=True, check=True)
    except subprocess.CalledProcessError:
        logging.error('SubprocessError: %s',
                      f'{trj.system.name}_{postfix}_{trj.b}-{trj.e}-{trj.dt}')


def calculate_scd(trj: TrajectorySlice,
                  selections: list, postfix: str) -> None:
    '''
    calculates per atom Scd values for each step in trajectory part
    for each timestep corresponding selection
    '''
    trj_list = [TrajectorySlice(trj.system, fr, fr, 0)
                for fr in range(int(trj.b * 1000), int(trj.e * 1000), trj.dt)]
    multiproc(run_scd_py, trj_list, selections,
              (postfix for _ in trj_list),
              n_workers=20)


def scd_summary(trj_slices: list, postfix: str) -> pd.DataFrame:
    '''
    aggregates data from individual _scd.csv files to single file
    '''
    scd_folder = PATH / 'notebooks' / 'scd'

    systems = []
    times = []
    chains = []
    atoms = []
    scds = []

    print('aggregating data...')

    for trj in trj_slices:
        scd_files = [
            str(scd_folder / 'near_chl' /
                f'{trj.system.name}_{postfix}_{i}-{i}-0_scd.csv')
            for i in range(int(trj.b * 1000), int(trj.e * 1000), trj.dt)]
        for file in scd_files:
            try:
                lines = opener(file)
            except FileNotFoundError:
                logging.error('FileNotFound: %s', Path(file).name)
                continue
            systems.extend([trj.system.name for i in range(
                len(lines[1].split(',')) + len(lines[5].split(',')))])
            time = file.split('-')[1].split('_')[0]
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


def generate_selections(trj, lip_dc, n_pl_df):
    '''
    for trj one tuple,
    each element -- list of selections for each frame
    (strings with ,)
    '''
    data = lip_dc[(lip_dc['index'] == trj.system.name) &
                  (lip_dc['dmn'] == 'CHL') & (lip_dc['amn'] != 'CHL')].copy()
    data['ami'] = data['ami'].astype(str)
    prefixes = data.groupby(
        'timepoint').agg({'ami': lambda x: ','.join(x)})['ami'].to_list()

    prefixes_near_chol = [
        '/' + ','.join(set(i.split(','))) + '//' for i in prefixes]
    all_pls = n_pl_df[(n_pl_df['index'] == trj.system.name)]['n_pl'].to_list()
    prefixes_not_near_chol = [
        '/' + ','.join(set([str(s) for s in range(n)])
                       - set(i.split(','))) + '//'
        for i, n in zip(prefixes, all_pls)]
    return prefixes_near_chol, prefixes_not_near_chol


def plot_scd_near_chl(trj_slices):
    df_near_chol = pd.read_csv(PATH / 'notebooks' / 'scd' / 'near_chl' /
                               f'scd_near_chl_dt{trj_slices[0].dt}.csv')
    df_not_near_chol = pd.read_csv(PATH / 'notebooks' / 'scd' / 'near_chl' /
                                   f'scd_not_near_chl_dt{trj_slices[0].dt}.csv')
    df_near_chol['near_chol'] = 'yes'
    df_not_near_chol['near_chol'] = 'no'
    df = pd.concat([df_near_chol, df_not_near_chol], ignore_index=True)

    def dfol(
            df: pd.DataFrame, system: str,
            chain: str, chl_amount: int) -> pd.DataFrame:
        '''
        extract data for one line in plot from df
        '''
        return df[(df['system'] == system)
                  & (df['chain'] == chain)
                  & (df['CHL amount, %'] == chl_amount)]

    df['atom_n'] = df['atom'].apply(lambda x: int(x[2:]))
    scd_ms = df.drop(columns=['timepoint', 'atom']).groupby(
        ['system', 'CHL amount, %', 'chain', 'atom_n', 'near_chol'],
        as_index=False).agg(['mean', 'std'])
    scd_ms = scd_ms.reset_index(level=1).reset_index(
        level=1).reset_index(level=1).reset_index()

    for syst in scd_ms['system'].unique():
        fig, axs = plt.subplots(1, 3, figsize=(20, 7),
                                sharex=True, sharey=True)
        for ax, chl_amount in zip(axs, scd_ms['CHL amount, %'].unique()):
            for c, near_chol in enumerate(scd_ms['near_chol'].unique()):
                scd_ms_part = scd_ms[scd_ms['near_chol'] == near_chol]
                for sn, ls in zip(('sn-1', 'sn-2'),
                                  ('-', '--')):
                    subdf = dfol(scd_ms_part, syst, sn, chl_amount)
                    ax.errorbar(
                        x=subdf['atom_n'],
                        y=subdf['scd']['mean'],
                        yerr=subdf['scd']['std'],
                        ls=ls, color=f'C{c}',
                        elinewidth=1, label=f'{sn}, '
                        f'contacts with CHL: {near_chol}'
                    )
            ax.set_title(f'{chl_amount} % CHL')
            ax.set_xlabel('C atom number')
        fig.suptitle(syst)
        axs[0].set_ylabel('Scd')
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels,
                   loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2)
        fig.savefig(PATH / 'notebooks' / 'scd' / 'res' /
                    'scd_near_chl_'
                    f'{syst}_{trj_slices[0].b}-{trj_slices[0].e}-'
                    f'{trj_slices[0].dt}.png',
                    bbox_inches='tight', dpi=300)


def main():
    '''
    analysis of SOL inside bilayer: n and lt of hbonds with CHL and PL
    '''
    sns.set(style='ticks', context='talk', palette='muted')
    initialize_logging('scd_near_chl.log', False)
    systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in ['popc', 'dmpc','dopc','dops']])
    systems = list(dict.fromkeys(systems))
    trj_slices = [TrajectorySlice(System(
        PATH, s), 200.0, 201.0, 1) for s in systems]

    logging.info('loading data...')
    n_pl_df = get_n_pl_df(trj_slices)
    lip_dc = pd.read_csv(
        PATH / 'notebooks' / 'contacts' /
        f'lip_dc_dc_{trj_slices[0].b}-{trj_slices[0].e}-'
        f'{trj_slices[0].dt}_rchist_full.csv')

    logging.info('calculating scd...')
    for trj in trj_slices:
        near_chl, not_near_chl = generate_selections(trj, lip_dc, n_pl_df)
        calculate_scd(trj, near_chl, 'near_chl')
        calculate_scd(trj, not_near_chl, 'not_near_chl')

    scd_summary(trj_slices, 'near_chl').to_csv(
        PATH / 'notebooks' / 'scd' / 'near_chl' /
        f'scd_near_chl_dt{trj_slices[0].dt}.csv',
        index=False
    )
    scd_summary(trj_slices, 'not_near_chl').to_csv(
        PATH / 'notebooks' / 'scd' / 'near_chl' /
        f'scd_not_near_chl_dt{trj_slices[0].dt}.csv',
        index=False
    )
    logging.info('plotting...')
    plot_scd_near_chl(trj_slices)
    logging.info('done.')


if __name__ == '__main__':
    main()

# TODO: все поломалось, как-то обновить алгоритм
