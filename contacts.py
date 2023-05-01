import logging
import os
import subprocess
from pathlib import Path

import pandas as pd
import rich
import typer
from modules.constants import EXPERIMENTS, PATH
from modules.general import (duration, flatten, initialize_logging, multiproc,
                             progress_bar)
from modules.tg_bot import run_or_send_error
from modules.traj import System, TrajectorySlice

app = typer.Typer(rich_markup_mode='rich', add_completion=False)


def calculate_contacts(trj: TrajectorySlice, grp1: str, grp2: str, contact_type: str,
                       per_atom: int = 1):
    '''
    wrapper for impulse to calculate specified contacts

    trj -- trajectory slice for which calculate contacts
    grp1, grp2 -- names of groups between which calculate contacts
    contact_type -- "hb" for hbonds, "dc" for contacts by distance
    per_atom -- if 1, calculate per atom, if 0 -- per molecule
    '''
    (Path(trj.system.dir) / 'contacts').mkdir(parents=True, exist_ok=True)
    os.chdir(Path(trj.system.dir) / 'contacts')

    grp2_name = grp2 if '/' not in grp2 else grp2.replace('/', '')
    grp2_name = grp2 if '/' not in grp2 else grp2_name.replace('|', '')

    args = f'''
TOP={trj.system.dir}/md/md.tpr
TRJ={trj.system.dir}/md/pbcmol_201.xtc
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
OUT="{grp1}_{grp2_name}_{contact_type}"'''

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
def get():
    initialize_logging('contacts.log')
    systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in flatten(EXPERIMENTS.values())])
    systems = list(dict.fromkeys(systems))

    trajectory_slices = [TrajectorySlice(System(PATH, s), 200, 201, 1) for s in systems]

    # multiproc(calculate_contacts,
    #           trajectory_slices,
    #           ('lip' for _ in trajectory_slices),
    #           (None for _ in trajectory_slices),
    #           ('hb' for _ in trajectory_slices),
    #           (1 for _ in trajectory_slices),
    #           n_workers=8,
    #           messages=True,
    #           descr='intrabilayer hbonds'
    #           )
    #
    # multiproc(calculate_contacts,
    #           trajectory_slices,
    #           ('lip' for _ in trajectory_slices),
    #           ('SOL' for _ in trajectory_slices),
    #           ('hb' for _ in trajectory_slices),
    #           (1 for _ in trajectory_slices),
    #           n_workers=8,
    #           messages=True,
    #           descr='hbonds with water'
    #           )
    #
    trajectory_slices_no_chl = [trj for trj in trajectory_slices if 'chol' in trj.system.name]
    # grp2_selectors = [trj.system.pl_selector()[1:-4] if 'dopc_dops50' not in trj.system.name
    #                   else trj.system.pl_selector(0)[1:-4] + '///|' +
    #                   trj.system.pl_selector(1)[1:-6] for trj in trajectory_slices_no_chl]
    #
    # multiproc(calculate_contacts,
    #           trajectory_slices_no_chl,
    #           ('CHOL' for _ in trajectory_slices_no_chl),
    #           (grp2_selectors),
    #           ('dc' for _ in trajectory_slices_no_chl),
    #           (1 for _ in trajectory_slices_no_chl),
    #           n_workers=8,
    #           messages=True,
    #           descr='intrabilayer contacts'
    #           )

    multiproc(calculate_contacts,
              trajectory_slices_no_chl,
              ('CHOL' for _ in trajectory_slices_no_chl),
              ('SOL' for _ in trajectory_slices_no_chl),
              ('dc' for _ in trajectory_slices_no_chl),
              (1 for _ in trajectory_slices_no_chl),
              n_workers=8,
              messages=True,
              descr='contacts with water'
              )
    # trj = TrajectorySlice(System(PATH, 'dopc_dops50_chol30'), 200, 201, 1)
    # calculate_contacts(trj, 'lip', None, 'hb', 1)


if __name__ == '__main__':
    app()

# trj.system.pl_selector()[1:-4],

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
