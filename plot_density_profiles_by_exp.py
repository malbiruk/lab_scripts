'''
draw figures of density profiles per experiment (vertical layout)
'''

import subprocess

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
from modules.constants import EXPERIMENTS, PATH
from modules.density import plot_density_profile
from modules.general import chunker, flatten, multiproc
from modules.traj import System, TrajectorySlice


def plot_dp(trj_slices, detailed=False):
    for exp, systs in EXPERIMENTS.items():
        print(exp)
        print(f'plotting {exp}...')
        fig, axs = plt.subplots(4, 3, figsize=(20, 25),
                                gridspec_kw={'hspace': 0.3},
                                sharex=True, sharey=True)

        axs = axs.reshape(-1, order='F')
        for c, trj in enumerate(
            [trj for trj in trj_slices
             if trj.system.name
             in flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                        for i in systs])]):
            plot_density_profile(axs[c], trj)
            axs[c].set_title(trj.system.name)
            if detailed:
                axs[c].set_xlim(0, 3)

            if c not in list(range(4)):
                axs[c].set_ylabel('')
            axs[c].xaxis.set_tick_params(which='both', labelbottom=True)

        axs[7].legend(loc='upper center',
                      bbox_to_anchor=(0.5, -0.2), ncol=6)
        fig.savefig(
            PATH / 'notebooks' / 'dp' / 'full_by_exp' /
            f'{"_".join(exp.split())}_'
            f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}'
            '_dp.png',
            bbox_inches='tight', dpi=300)

def fv_figure_single(ax, trj, label=None, color='C0'):
    df = pd.read_csv(
        PATH / 'notebooks' / 'intrabilayer_space' /
        f'{trj.system.name}_{trj.b}-{trj.e}-{trj.dt}_fvoz.csv',
        header=None, skiprows=1, names=['Z, Å', 'Free Volume, Å³'])

    df['Z, Å'] = df['Z, Å'] + \
        (df['Z, Å'].max() - df['Z, Å'].min()) / 2 - df['Z, Å'].max()

    ax.plot(df['Z, Å'], df['Free Volume, Å³'], color=color, label=label)


def plot_intrabilayer_volume(trj_slices):
    palette = sns.color_palette('crest_r', 4)
    chl_amounts = [0, 10, 30, 50]
    fig, axs = plt.subplots(3, 3, figsize=(20, 20),
                            gridspec_kw={'wspace': 0.3},
                            sharex=True, sharey=True)
    axs = axs.reshape(-1, order='F').flatten()
    c = 0
    for _, systs in EXPERIMENTS.items():
        trjs = [trj for trj in trj_slices
                if trj.system.name
                in flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                            for i in systs])]
        for chunk in chunker(trjs, 4):
            for it, trj in enumerate(chunk):
                fv_figure_single(axs[c], trj, chl_amounts[it], palette[it])
            axs[c].set_title(chunk[0].system.name)
            if c < 3:
                axs[c].set_ylabel('Free Volume, Å³')
            if c in [2, 5, 8]:
                axs[c].set_xlabel('Z, Å')
            c += 1
    axs[5].legend(title='CHL amount, %', loc='upper center',
                  bbox_to_anchor=(0.5, -0.2), ncol=6)
    fig.savefig(
        PATH / 'notebooks' / 'intrabilayer_space' / 'imgs' /
        f'free_volume_dt{trj_slices[0].dt}.png',
        bbox_inches='tight', dpi=300)


def calculate_intrabilayer_space(trj):
    args = f'SYSTEM={trj.system.dir}/md/{trj.system.tpr}\n' \
        f'TRJ={trj.system.dir}/md/{trj.system.xtc}\n' \
        f'BEG={trj.b*1000}\nEND={trj.e*1000}\nDT={trj.dt}\n' \
        f'OUT="{trj.system.path}/notebooks/intrabilayer_space/' \
        f'{trj.system.name}_{trj.b}-{trj.e}-{trj.dt}"\n' \
        'STP=0.5\nRE=1.4\nBH=50'

    with open(f'{trj.system.dir}/intrabilayer_space.args',
              'w', encoding='utf-8') as f:
        f.write(args)

    cmd = ' '.join([
        '/nfs/belka2/soft/impulse/dev/inst/runtask.py',
        '-t /home/krylov/Progs/IBX/AMMP/test/postpro/fv/fv_stat.mk',
        f'-f {trj.system.dir}/intrabilayer_space.args',
    ])

    subprocess.run(cmd, shell=True, check=True)


def main(detailed: bool = typer.Option(False)):
    '''
    draw figures of density profiles per experiment (vertical layout)
    '''
    sns.set(style='ticks', context='talk', palette='muted')
    systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in flatten(EXPERIMENTS.values())])
    systems = list(dict.fromkeys(systems))
    trj_slices = [TrajectorySlice(System(
        PATH, s), 150, 200, 1000) for s in systems]

    plot_dp(trj_slices)
    # multiproc(calculate_intrabilayer_space, trj_slices)
    plot_intrabilayer_volume(trj_slices)

    print('done.')


if __name__ == '__main__':
    typer.run(main)
