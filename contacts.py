'''
set of utilities to obtain and plot contacts data
'''
# pylint: disable = too-many-arguments

import logging
import os
import subprocess
from pathlib import Path
from typing import Tuple

# import pandas as pd
import typer
from modules.constants import EXPERIMENTS, PATH
from modules.general import duration, flatten, initialize_logging, multiproc
from modules.tg_bot import run_or_send_error
from modules.traj import System, TrajectorySlice

app = typer.Typer(rich_markup_mode='rich', add_completion=False)


def calculate_contacts(trj: TrajectorySlice, grp1: str, grp2: str, contact_type: str,
                       per_atom: int = 1) -> bool:
    '''
    wrapper for impulse to calculate specified contacts

    trj -- trajectory slice for which calculate contacts
    grp1, grp2 -- names of groups between which calculate contacts
    contact_type -- "hb" for hbonds, "dc" for contacts by distance
    per_atom -- if 1, calculate per atom, if 0 -- per molecule
    '''
    (Path(trj.system.dir) / 'contacts').mkdir(parents=True, exist_ok=True)
    os.chdir(Path(trj.system.dir) / 'contacts')

    grp2_name = ''
    if grp2 is not None:
        grp2_name = grp2 if '/' not in grp2 else grp2.replace('/', '')
        grp2_name = grp2 if '/' not in grp2 else grp2_name.replace('|', '')
        grp2_name = grp2_name + '_'

    args = f'''
TOP={trj.system.dir}/md/{trj.system.tpr}
TRJ={trj.system.dir}/md/{trj.system.xtc}
BEG={int(trj.b*1000)}
END={int(trj.e*1000)}
DT={trj.dt}
SUBSET="///"
GRP="{grp1+'///' if grp2 is None else ''}"
GRP1="{grp1+'///' if grp2 is not None else ''}"
GRP2="{grp2+'///' if grp2 is not None else ''}"
CLIST={contact_type}
PERATOM={per_atom}
FMT=csv
OUT="{grp1}_{grp2_name + contact_type}"'''

    with open('args', 'w', encoding='utf-8') as f:
        f.write(args)

    msg = f'couldn\'t obtain contacts for `{trj.system.name}`'
    impulse = '/nfs/belka2/soft/impulse/dev/inst/runtask.py'
    cont_stat = '/home/krylov/Progs/IBX/AMMP/test/postpro/contacts/cont_stat.js'
    cmd = f'{impulse} -f args -t {cont_stat}'

    if run_or_send_error(cmd, msg, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT):
        logging.info('sucessfully calculated contacts for %s', trj.system.name)
        return True

    logging.error('couldn\'t calculate contacts for %s', trj.system.name)
    return False


@duration
@app.command()
def get(ctx: typer.Context,
    contact_types: Tuple[bool, bool, bool, bool] = typer.Option(
        (True, True, True, True),
        help='tasks to run [dim](pick from [bold]hb_lip, hb_sol, dc_lip, dc_sol[/])[/]'),
):
    '''
    run [bold]impulse cont_stat[/] on all systems to obtain contacts info
    '''
    trj, tpr, b, e, dt, n_workers, verbose, messages = ctx.obj

    initialize_logging('get_contacts.log', verbose)
    systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in flatten(EXPERIMENTS.values())])
    systems = list(dict.fromkeys(systems))

    trajectory_slices = [TrajectorySlice(System(PATH, s, trj, tpr), b, e, dt) for s in systems]

    trajectory_slices_no_chl = [trj for trj in trajectory_slices if 'chol' in trj.system.name]

    grp2_selectors = [trj.system.pl_selector()[1:-4] if 'dopc_dops50' not in trj.system.name
                      else trj.system.pl_selector(0)[1:-4] + '///|' +
                      trj.system.pl_selector(1)[1:-6] for trj in trajectory_slices_no_chl]

    if contact_types[0]:
        logging.info('started intrabilayer hbonds calculation')
        multiproc(calculate_contacts,
                  trajectory_slices,
                  ('lip' for _ in trajectory_slices),
                  (None for _ in trajectory_slices),
                  ('hb' for _ in trajectory_slices),
                  (1 for _ in trajectory_slices),
                  n_workers=n_workers,
                  messages=messages,
                  descr='intrabilayer hbonds'
                  )
        logging.info('intrabilayer hbonds calculation done')
        logging.info('')

    if contact_types[1]:
        logging.info('started hbonds with water calculation')
        multiproc(calculate_contacts,
                  trajectory_slices,
                  ('lip' for _ in trajectory_slices),
                  ('SOL' for _ in trajectory_slices),
                  ('hb' for _ in trajectory_slices),
                  (1 for _ in trajectory_slices),
                  n_workers=n_workers,
                  messages=messages,
                  descr='hbonds with water'
                  )
        logging.info('hbonds with water calculation done')
        logging.info('')

    if contact_types[2]:
        logging.info('started intrabilayer contacts calculation')
        multiproc(calculate_contacts,
                  trajectory_slices_no_chl,
                  ('CHOL' for _ in trajectory_slices_no_chl),
                  (grp2_selectors),
                  ('dc' for _ in trajectory_slices_no_chl),
                  (1 for _ in trajectory_slices_no_chl),
                  n_workers=n_workers,
                  messages=messages,
                  descr='intrabilayer contacts'
                  )
        logging.info('intrabilayer contacts calculation done')
        logging.info('')

    if contact_types[3]:
        logging.info('started contacts with water calculation')
        multiproc(calculate_contacts,
                  trajectory_slices_no_chl,
                  ('CHOL' for _ in trajectory_slices_no_chl),
                  ('SOL' for _ in trajectory_slices_no_chl),
                  ('dc' for _ in trajectory_slices_no_chl),
                  (1 for _ in trajectory_slices_no_chl),
                  n_workers=n_workers,
                  messages=messages,
                  descr='contacts with water'
                  )
        logging.info('contacts with water calculation done')
        logging.info('')
    logging.info('done.')

@app.callback()
def callback(ctx: typer.Context,
           trj: str = typer.Option(
               'pbcmol_201.xtc', '-trj', help='name of trajectory files',
               rich_help_panel='Trajectory parameters'),
           tpr: str = typer.Option(
               '201_ns.tpr', '-tpr', help='name of topology files',
               rich_help_panel='Trajectory parameters'),
           b: float = typer.Option(
               200, '-b', help='beginning of trajectories (in ns)',
               rich_help_panel='Trajectory parameters'),
           e: float = typer.Option(
               201, '-e', help='end of trajectories (in ns)',
               rich_help_panel='Trajectory parameters'),
           dt: int = typer.Option(
               1, '-dt', help='timestep of trajectories (in ps)',
               rich_help_panel='Trajectory parameters'),
           n_workers: int = typer.Option(
               8, help='n of processes to start for each task',
               rich_help_panel='Script config'),
           verbose: bool = typer.Option(
               False, '--verbose', '-v', help='print debug log',
               rich_help_panel='Script config'),
           messages: bool = typer.Option(
               True, help='send updates info in telegram',
               rich_help_panel='Script config')):
    '''set of utilities to obtain and plot contacts data'''
    ctx.obj = (trj, tpr, b, e, dt, n_workers, verbose, messages) # store command arguments


if __name__ == '__main__':
    app()


# %%
# trj.system.pl_selector()[1:-4],
# trj = TrajectorySlice(System(PATH, 'dopc_dops50_chol30'), 200, 201, 1)
# calculate_contacts(trj, 'lip', None, 'hb', 1)

# %% calculate contacts
# calculate_contacts CHL+PL CHL+PL hb 1
# calculate_contacts CHL+PL SOL hb 1
# calculate_contacts CHL+PL CHL+PL cd 1

# %% calculate areas
# exposed_area CHL SOL
# exposed_area PL SOL
# exposed_area CHL PL

# %% plots
# hbonds (prob, lt): CHL-PL, PL-PL, PL-SOL, CHL-SOL (per CHL/PL averaged by time)
# hbonds (prob, lt): CHL - dif groups of PL, PL-PL (dif groups), PL-SOL (dif groups)
# contacts: CHL-PL / LIP-LIP
