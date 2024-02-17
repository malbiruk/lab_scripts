import logging
import os
import subprocess

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from contacts import import_ft_table
from modules.constants import EXPERIMENTS, PATH
from modules.general import chunker, flatten, initialize_logging, multiproc
from modules.traj import System, TrajectorySlice
from z_slices_angles_contacts import (contacts_to_single_tables, get_n_chl_df,
                                      update_contacts_tables_single_trj)


def create_args(trj, atom, outname):
    '''
    atom: PH1, NH7
    '''
    (PATH / trj.system.name / 'a').mkdir(
        parents=True, exist_ok=True)
    os.chdir(PATH / trj.system.name / 'a')

    args = f'''SYSTEM = "../md/{trj.system.tpr}"
TRJ = "../md/{trj.system.xtc}"
BEG = {trj.b*1000}
END = {trj.e*1000}
OUT = "{atom}"
DT = {trj.dt}
S1 = "CHOL///O3"
S2 = "{trj.system.name.split("_",1)[0].upper()}///{atom}"'''

    with open(outname, 'w', encoding='utf-8') as f:
        f.write(args)


def calc_rdfs(trj):
    if 'dopc_dops' not in trj.system.name:
        create_args(trj, 'PH1', 'rdf_P.args')
        os.chdir(PATH / trj.system.name / 'a')
        cmd = '/nfs/belka2/soft/impulse/dev/inst/runtask.py '\
            '-t /nfs/belka2/soft/impulse/tasks/post/rdf/rdf.mk '\
            '-f rdf_P.args'
        subprocess.run(cmd, shell=True, check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT)

        create_args(trj, 'NH7', 'rdf_N.args')
        os.chdir(PATH / trj.system.name / 'a')
        cmd = '/nfs/belka2/soft/impulse/dev/inst/runtask.py '\
        '-t /nfs/belka2/soft/impulse/tasks/post/rdf/rdf.mk '\
        '-f rdf_N.args'
        subprocess.run(cmd, shell=True, check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT)

        if 'dops' in trj.system.name:
            create_args(trj, 'OH9', 'rdf_O9.args')
            os.chdir(PATH / trj.system.name / 'a')
            cmd = '/nfs/belka2/soft/impulse/dev/inst/runtask.py '\
            '-t /nfs/belka2/soft/impulse/tasks/post/rdf/rdf.mk '\
            '-f rdf_O9.args'
            subprocess.run(cmd, shell=True, check=True,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT)

            create_args(trj, 'OH10', 'rdf_O10.args')
            os.chdir(PATH / trj.system.name / 'a')
            cmd = '/nfs/belka2/soft/impulse/dev/inst/runtask.py '\
            '-t /nfs/belka2/soft/impulse/tasks/post/rdf/rdf.mk '\
            '-f rdf_O10.args'
            subprocess.run(cmd, shell=True, check=True,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT)


def plot_rdfs(trj_slices):
    for systs in chunker(trj_slices, 3):
        fig, axs = plt.subplots(1, 3, figsize=(20, 7),
                                sharex=True, sharey=True)
        for ax, trj in zip(axs, systs):
            df_n = pd.read_csv(
                PATH / trj.system.name / 'a' / 'NH7_rdf.csv')
            df_p = pd.read_csv(
                PATH / trj.system.name / 'a' / 'PH1_rdf.csv')
            ax.plot(df_n['# r'], df_n['g'], label='N')
            ax.plot(df_p['# r'], df_p['g'], label='P')
            if 'dops' in systs[0].system.name:
                df_oh9 = pd.read_csv(
                    PATH / trj.system.name / 'a' / 'OH9_rdf.csv')
                df_oh10 = pd.read_csv(
                    PATH / trj.system.name / 'a' / 'OH10_rdf.csv')
                ax.plot(df_n['# r'], df_oh9['g'], label='OH9')
                ax.plot(df_p['# r'], df_oh10['g'], label='OH10')
            ax.set_xlabel('distance to CHL O, Å')
            sname, chl = trj.system.name.split('_chol', 1)
            ax.set_title(f'{sname}, {chl}% CHL')
            ax.set_xlim(0, 20)
        axs[0].set_ylabel('g(r)')
        axs[1].legend(loc='upper center',
                      bbox_to_anchor=(0.5, -0.15), ncol=4)

        fig.savefig(PATH / 'notebooks' / 'rdf' /
                    f'{systs[0].system.name.split("_chol",1)[0]}_P_N_chl_O.png',
                    bbox_inches='tight', dpi=300)


def main():
    sns.set(style='ticks', context='talk', palette='muted')
    initialize_logging('sol_inside_bilayer.log', False)
    # systems = flatten([(i + '_chol10', i + '_chol30', i + '_chol50')
    #                    for i in flatten(EXPERIMENTS.values())])
    systems = flatten([(i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in ['dopc', 'dops']])
    systems = list(dict.fromkeys(systems))
    trj_slices = [TrajectorySlice(System(
        PATH, s),
        150, 200, 20) for s in systems]

    # multiproc(calc_rdfs, trj_slices, n_workers=6)
    plot_rdfs(trj_slices)


# %%
if __name__ == '__main__':
    main()


# %%
