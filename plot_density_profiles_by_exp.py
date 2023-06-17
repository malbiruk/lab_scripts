'''
draw figures of density profiles per experiment (vertical layout)
'''

import matplotlib.pyplot as plt
import seaborn as sns
import typer
from modules.constants import EXPERIMENTS, PATH
from modules.density import plot_density_profile
from modules.general import flatten
from modules.traj import System, TrajectorySlice


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
            f'{"_".join(exp.split())}_details_'
            f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}'
            '_dp.png',
            bbox_inches='tight', dpi=300)

    print('done.')


if __name__ == '__main__':
    typer.run(main)
