import logging

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
from modules.constants import EXPERIMENTS, PATH, TO_RUS
from modules.general import flatten, initialize_logging, multiproc
from modules.traj import System, TrajectorySlice
from scipy.ndimage import label
from skimage import measure


def get_ratio_of_contour_chl_with_bordering_phob(
        progress: dict, task_id: int, trj: TrajectorySlice):
    '''
    in each mhpmap calculates contours of chl position
    saves ratio of n of dots in contour neighboring with phob mhp dots
    '''
    # pylint: disable = not-an-iterable, too-many-nested-blocks, too-many-locals

    mapp = np.load(
        PATH / trj.system.name /
        f'mhp_{trj.b}-{trj.e}-{trj.dt}' / '1_data.nmp')['data']
    at_info = np.load(PATH / trj.system.name /
                      f'mhp_{trj.b}-{trj.e}-{trj.dt}' / '1_pa.nmp')['data']

    u = mda.Universe(
        str(PATH / trj.system.name / 'md' / trj.system.tpr),
        str(PATH / trj.system.name / 'md' / trj.system.xtc),
        refresh_offsets=True)

    atom_resname_dict = {}
    for atom in u.atoms:
        atom_resname_dict[atom.id] = atom.resname
    atom_resname_dict[-1] = np.nan

    where_chols = np.vectorize(
        lambda x: atom_resname_dict[x] == 'CHL')(at_info).astype(int)

    all_contours = []
    for ts in where_chols:
        all_contours.append(measure.find_contours(ts, 0.5))

    hydroph_neighbor_ratios = []
    len_of_task = len(mapp)
    c = 0
    for contours, data, mask in zip(all_contours, mapp, where_chols):
        labeled_mask = measure.label(mask)
        for contour in contours:
            contour_length = len(contour)
            hydroph_neighbor_dots = []

            for dot in contour:
                x, y = dot
                x, y = int(x), int(y)
                if x in [0, 150] or y in [0, 150]:
                    continue
                for i in range(max(x - 1, 0), min(x + 2, 150)):
                    for j in range(max(y - 1, 0), min(y + 2, 150)):
                        if labeled_mask[i, j] == 0:
                            if data[i, j] >= 0.5:
                                hydroph_neighbor_dots.append(dot)

            ratio_of_phob_neighbor_dots = len(
                {tuple(arr) for arr in hydroph_neighbor_dots}) / contour_length
            hydroph_neighbor_ratios.append(ratio_of_phob_neighbor_dots)
        c += 1
        progress[task_id] = {'progress': c + 1, 'total': len_of_task}

    np.save(PATH / 'tmp' / f'{trj.system.name}_chl_hydroph_neighbors.npy',
            np.array(hydroph_neighbor_ratios))


def get_chl_phob_neighbors_df(trj_slices):
    fname = (PATH / 'notebooks' / 'mhpmaps' /
             f'chl_hydroph_neighbors_dt{trj_slices[0].dt}.csv')
    if fname.is_file():
        return pd.read_csv(fname)

    multiproc(get_ratio_of_contour_chl_with_bordering_phob,
              trj_slices,
              n_workers=len(trj_slices),
              show_progress='multiple')
    to_df = {
        'index': [],
        'system': [],
        'CHL amount, %': [],
        '% of CHL near phob': []
    }
    for trj in trj_slices:
        try:
            a = np.load(PATH / 'tmp' /
                        f'{trj.system.name}_chl_hydroph_neighbors.npy')
        except FileNotFoundError:
            print(trj.system.name)

        to_df['index'].extend([trj.system.name] * len(a))
        to_df['system'].extend([trj.system.name.split('_chol', 1)[0]] * len(a))
        to_df['CHL amount, %'].extend(
            [trj.system.name.split('_chol', 1)[1]] * len(a))
        to_df['% of CHL near phob'].extend(a * 100)
    df = pd.DataFrame(to_df)
    df.to_csv(fname, index=False)
    return df


def plot_chl_phob_neighbors_df(df, trj_slices):
    fig, axs = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    for ax, exp in zip(axs, EXPERIMENTS):
        data = df[df['system'].isin(EXPERIMENTS[exp])]
        sns.violinplot(data=data,
                       x='system', y='% of CHL near phob',
                       hue='CHL amount, %', ax=ax, inner='quartile',
                       edgecolor='k', palette='RdYlGn_r'
                       )
        ax.set_title(exp)
        ax.set_ylim(0)
        if ax != axs[1]:
            ax.legend([], [], frameon=False)
        if ax != axs[0]:
            ax.set_ylabel('')

    sns.move_legend(axs[1], loc='upper center',
                    bbox_to_anchor=(0.5, -0.2), ncol=6)

    fig.patch.set_facecolor('white')
    fig.savefig(
        PATH / 'notebooks' / 'mhpmaps' / 'imgs' /
        'chl_hydroph_neighbors_'
        f'dt{trj_slices[0].dt}.png',
        bbox_inches='tight', dpi=300)


def main():
    sns.set(style='ticks', context='talk', palette='muted')
    initialize_logging('chl_mhp_phob_nb.log', )
    systems = flatten([(i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in flatten(EXPERIMENTS.values())])
    systems = list(dict.fromkeys(systems))
    trajectory_slices = [TrajectorySlice(
        System(PATH, s,
               'pbcmol_201.xtc', '201_ns.tpr'), 200.0, 201.0, 1)
        for s in systems]
    df = get_chl_phob_neighbors_df(trajectory_slices)
    plot_chl_phob_neighbors_df(df, trajectory_slices)


# %%
if __name__ == '__main__':
    main()


# %%
# sns.set(style='ticks', context='talk', palette='muted')
#
# systems = flatten([(i + '_chol10', i + '_chol30', i + '_chol50')
#                    for i in ['dopc', 'dops']])
# systems = list(dict.fromkeys(systems))
# trajectory_slices = [TrajectorySlice(
#     System(PATH, s,
#            'pbcmol_201.xtc', '201_ns.tpr'), 200.0, 201.0, 1)
#     for s in systems]
#
#
# # %%
# all_counts = {}
# trj_slices = trajectory_slices
#
# for trj in trj_slices:
#     mapp = np.load(
#         PATH / trj.system.name /
#         f'mhp_{trj.b}-{trj.e}-{trj.dt}' / '1_data.nmp')['data']
#     at_info = np.load(PATH / trj.system.name /
#                       f'mhp_{trj.b}-{trj.e}-{trj.dt}' / '1_pa.nmp')['data']
#
#     u = mda.Universe(
#         str(PATH / trj.system.name / 'md' / trj.system.tpr),
#         str(PATH / trj.system.name / 'md' / trj.system.xtc),
#         refresh_offsets=True)
#
#     atom_resname_dict = {}
#     for atom in u.atoms:
#         atom_resname_dict[atom.id] = atom.resname
#     atom_resname_dict[-1] = np.nan
#
#     where_chols = np.vectorize(
#         lambda x: atom_resname_dict[x] == 'CHL')(at_info).astype(int)
#
#     for mask in where_chols:
#         s = [[1, 1, 1],
#              [1, 1, 1],
#              [1, 1, 1]]
#         clusters, _ = label(mask, structure=s)
#
#         # periodic boundary conditions
#         for y in range(clusters.shape[0]):
#             if clusters[y, 0] > 0 and clusters[y, -1] > 0:
#                 clusters[clusters == clusters[y, -1]] = clusters[y, 0]
#         for x in range(clusters.shape[1]):
#             if clusters[0, x] > 0 and clusters[-1, x] > 0:
#                 clusters[clusters == clusters[-1, x]] = clusters[0, x]
#         labels, counts = np.unique(
#             clusters[clusters > 0], return_counts=True)
#
#         try:
#             all_counts[trj.system.name].extend(list(counts))
#         except KeyError:
#             all_counts[trj.system.name] = list(counts)
#
# # %%
# df = pd.DataFrame([(k, v) for k, vals in all_counts.items() for v in vals],
#                   columns=['index', 'clsize'])
#
# # %%
# df[['system', 'CHL amount, %']] = df['index'].str.split('_chol', n=1, expand=True)
# df
#
# df['% поверхности'] = df['clsize'] / (150 * 150) * 100
# # %%
#
# fig, axs = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
# for ax, syst in zip(axs, ['dopc', 'dops']):
#     data = df[df['system'] == syst]
#     sns.histplot(data=data, x='% поверхности',
#                  ax=ax, hue='CHL amount, %',
#                  palette='crest_r', common_norm=False,
#                  stat='density',
#                  kde=True,
#                  # edgecolor='k',
#                  log_scale=True,
#                  legend=ax == axs[1])
#     ax.set_xlim(1)
#     ax.set_ylim(0, 1.5)
#     # ax.set_xlabel('% поверхности')
#     ax.set_title(TO_RUS[syst])
# axs[0].set_ylabel('Плотность вероятности')
#
# legend = axs[1].get_legend()
# handles = legend.legend_handles
# labels = df['CHL amount, %'].unique()
# legend.remove()
# fig.legend(handles, labels, title='Концентрация ХС, %', loc='upper center',
#            bbox_to_anchor=(0.5, 0), ncol=6)
# fig.patch.set_facecolor('white')
# fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'surface' /
#             f'chol_surface_sizes.png',
#             bbox_inches='tight', dpi=300)


# %%
u = mda.Universe(str(PATH / 'dopc_chol30' / 'md' / 'md.tpr'),
                 str(PATH / 'dopc_chol30' / 'md' / 'md.gro'),
                 )


# %%
chl = u.select_atoms('resname CHL').residues[0]
chl.atoms.names
