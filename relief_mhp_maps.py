'''
This script is created to create relief maps and draw
mhp maps with relief and mark CHL fractions on surface
'''
import logging
import os
import subprocess
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from mhp import obtain_mhpmap
from modules.constants import PATH
from modules.general import initialize_logging, progress_bar
from modules.tg_bot import run_or_send_error
from modules.traj import System, TrajectorySlice
from rich import traceback
from scipy.ndimage import gaussian_filter
from skimage.measure import find_contours

app = typer.Typer(rich_markup_mode='rich', add_completion=False)


def obtain_relief(trj: TrajectorySlice, force: bool = False) -> bool:
    '''
    obtain relief for specified trajectory slice
    '''
    desired_datafile = (PATH / trj.system.name /
                        f'relief_{trj.b}-{trj.e}-{trj.dt}' / '1_udfr.npy')

    if desired_datafile.is_file() and force is False:
        logging.info(
            'relief data for %s already calculated, skipping...',
            trj.system.name)
        return True

    (PATH / trj.system.name / f'relief_{trj.b}-{trj.e}-{trj.dt}').mkdir(
        parents=True, exist_ok=True)
    os.chdir(PATH / trj.system.name / f'relief_{trj.b}-{trj.e}-{trj.dt}')
    tpr = str(PATH / trj.system.name / 'md' / trj.system.tpr)
    xtc = str(PATH / trj.system.name / 'md' / trj.system.xtc)

    args = f'SYSTEM={str(tpr)}\nTRJ={str(xtc)}' \
        f'\nBEG={int(trj.b*1000)}\nEND={int(trj.e*1000)}\nDT={trj.dt}' \
        '\nNC=75\nOUT="1"' \
        '\nDUMP=1\nFDUMP=1\nLEAF="up"\nLIPSEL="lip///"'

    with open('args', 'w', encoding='utf-8') as f:
        f.write(args)

    impulse = Path('/nfs/belka2/soft/impulse/dev/inst/runtask.py')
    task_ = Path(
        '/home/krylov/Progs/IBX/AMMP/test/postpro/maps/'
        'plane/relief_apl/bl_relief.mk')
    cmd = f'{impulse} -f args -t {task_}'
    msg = f'couldn\'t obtain relief data for `{trj.system.name}`'

    if run_or_send_error(cmd, msg,
                         stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT):
        logging.info('sucessfully calculated relief data for %s',
                     trj.system.name)
        return True

    logging.error('couldn\'t calculate relief data for %s', trj.system.name)
    return False


def plot_relief(data, ax):
    '''
    plot standardized relief data (values from -4 to 2), single ax
    '''
    data = (data - np.mean(data)) / np.std(data)
    cmap = sns.color_palette('viridis', as_cmap=True)
    x, y = data.shape
    x = np.arange(x)
    y = np.arange(y)

    levels = np.linspace(-4, 2, 7)
    h = ax.contour(x, y, data, levels=levels, extend='both',
                   cmap=cmap
                   )
    return h


def plot_mhpmap(data, ax,
                lev=2, bside=75, title='', resolution=150,
                cbar=False):
    '''
    plot smoothed mhp data (values from -lev to lev), single ax
    '''
    # pylint: disable = too-many-arguments
    data_s = gaussian_filter(data, sigma=1)
    x, y = data_s.shape
    x = np.arange(x)
    y = np.arange(y)

    levels = np.linspace(-lev, lev, 16)
    cmap = sns.color_palette('BrBG_r', as_cmap=True)

    h = ax.contourf(x, y, data_s,
                    cmap=cmap,
                    levels=levels,
                    extend='both',
                    )

    positions = np.linspace(0, resolution - 1, 5)
    ax.xaxis.set_ticks(positions)
    ax.set_xticklabels([int(round(i * (bside / resolution)))
                        for i in positions])
    ax.yaxis.set_ticks(positions)
    ax.set_yticklabels([int(round(i * (bside / resolution)))
                        for i in positions])

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, resolution - 1)
    ax.set_ylim(0, resolution - 1)
    ax.set_xlabel('X, A')
    ax.set_ylabel('Y, A')
    ax.set_title(title)
    if cbar:
        plt.colorbar(h, location='right',
                     ticks=[-lev, -(lev / 2), 0, lev / 2, lev],
                     shrink=0.5)
    return h


def get_where_chols_in_trj(trj):
    '''
    this function uses '1_pa.nmp' files and mdanalysis
    to create array with 1 (where CHL) and 0 (where other atoms on surface)
    '''
    # pylint: disable = unsubscriptable-object
    at_info = np.load(PATH / trj.system.name /
                      f'mhp_{trj.b}-{trj.e}-{trj.dt}' / '1_pa.nmp')['data']

    u = mda.Universe(
        str(PATH / trj.system.name / 'md' / trj.system.tpr),
        str(PATH / trj.system.name / 'md' / trj.system.xtc),
        refresh_offsets=True)

    return np.vectorize(
        lambda x: u.atoms[x].resname == 'CHL')(at_info).astype(int)


def plot_chols_surface(data, ax):
    '''
    plot contours of where_chols (arrays with 0 and 1), single ax
    '''
    contours = find_contours(data, 0.5)
    for c, contour in enumerate(contours):
        label = 'CHL' if c == 0 else None
        ax.plot(contour[:, 1], contour[:, 0], c='k', ls='--', lw=2,
                label=label)


def plot_mhp_relief_chol(trj_slices):
    '''
    for each ts in trj_slices print figure with all trjs
    (needs to be 4: 0, 10, 30, 50 % CHL)

    mhpmaps with relief and where_chols will be presented in each plot
    '''
    logging.info('loading data...')
    data = {trj: {
        'relief': np.load(
            PATH / trj.system.name /
            f'relief_{trj.b}-{trj.e}-{trj.dt}' / '1_udfr.npy')[0],
        'mhp': np.load(
            PATH / trj.system.name /
            f'mhp_{trj.b}-{trj.e}-{trj.dt}' / '1_data.nmp')['data'],
        'where_chols': get_where_chols_in_trj(trj),
        'bside': int(round(pd.read_csv(
            PATH / trj.system.name /
            f'relief_{trj.b}-{trj.e}-{trj.dt}' / '1_xy.csv',
            header=None).loc[0, 0]))}
            for trj in trj_slices}

    logging.info('drawing plots...')
    with progress_bar as p:
        for ts in p.track(range(int(
                (trj_slices[0].e * 1000 - trj_slices[0].b * 1000)
                / trj_slices[0].dt)),
                description='drawing plots'):

            fig, axs = plt.subplots(2, 2, figsize=(20, 20))
            axs = axs.flatten()

            for ax, trj in zip(axs, trj_slices):
                mh = plot_mhpmap(data[trj]['mhp'][ts], ax,
                                 bside=data[trj]['bside'],
                                 resolution=75,
                                 title=trj.system.name)

                h = plot_relief(data[trj]['relief'][ts], ax)
                # if 'chol' in trj.system.name:
                plot_chols_surface(data[trj]['where_chols'][ts], ax)

            cax1 = fig.add_axes([0.3, 0.05, 0.4, 0.015])
            cax2 = fig.add_axes([0.95, 0.3, 0.015, 0.4])
            clb1 = fig.colorbar(h, cax=cax1, orientation='horizontal',
                                shrink=0.5)
            clb2 = fig.colorbar(mh, cax=cax2,
                                ticks=[-2, -(2 / 2), 0, 2 / 2, 2],
                                shrink=0.7)

            clb1.ax.set_title('Elevation')
            clb2.ax.set_ylabel('MHP, log P', rotation=270)
            axs[1].legend(loc='upper right', bbox_to_anchor=(1.3, 1.02),
                          frameon=False)
            fig.suptitle(
                f'{int(trj_slices[0].b*1000 + trj_slices[0].dt * ts)} ps')
            fig.savefig(
                PATH / 'notebooks' / 'mhpmaps' / 'imgs' / 'relief' /
                f'{trj_slices[0].system.name}_relief_mhp_chol_{ts}.png',
                bbox_inches='tight', dpi=300)
            plt.close()
        logging.info('done.')


@ app.command()
def main(
    systems: List[str] = typer.Argument(
        ..., help='systems for which to run the script '
        '(chol 10, 30, 50 % is hard-coded)'),
    xtc: str = typer.Option(
        'pbcmol_201.xtc', '-xtc', help='name of trajectory files',
        rich_help_panel='Trajectory parameters'),
    tpr: str = typer.Option(
        '201_ns.tpr', '-tpr', help='name of topology files',
        rich_help_panel='Trajectory parameters'),
    b: int = typer.Option(
        200, '-b', help='beginning of trajectories (in ns)',
        rich_help_panel='Trajectory parameters'),
    e: int = typer.Option(
        201, '-e', help='end of trajectories (in ns)',
        rich_help_panel='Trajectory parameters'),
    dt: int = typer.Option(
        50, '-dt', help='timestep of trajectories (in ps)',
        rich_help_panel='Trajectory parameters'),
):
    '''
    calculate and create mhp_relief figures for single system with all
    CHL concentrations
    '''
    # pylint: disable = too-many-arguments
    sns.set(style='ticks', context='talk', palette='muted')
    initialize_logging('mhp_relief.log')

    for system in systems:
        trj_slices = [TrajectorySlice(
            System(PATH, s, xtc, tpr), b, e, dt)
            for s in [system] + [f'{system}_chol{i}' for i in [10, 30, 50]]]

        with progress_bar as p:
            for trj in p.track(trj_slices):
                obtain_mhpmap(trj, False , 75)
                obtain_relief(trj, False)

        plot_mhp_relief_chol(trj_slices)


# %%
if __name__ == '__main__':
    traceback.install()
    app()


# # %%
# trj = trj_slices[2]
#
#
# # %%
# palette = sns.color_palette('crest_r', 3)
# fig, ax = plt.subplots(figsize=(5, 7))
#
# for c, trj in enumerate(trj_slices[1:]):
#
# ax.axvline(-0.5, ls=':', c='k')
# ax.axvline(0.5, ls=':', c='k')
# ax.set_xlabel('mhp')
# ax.legend(loc='upper right', title='CHL amount, %')
# ax.set_title('popc, only CHL')
# fig.savefig(
#     PATH / 'notebooks' / 'mhpmaps' / 'imgs' /
#     f'popc_chol_mhp_hist_{ts}.png',
#     bbox_inches='tight', dpi=300)
