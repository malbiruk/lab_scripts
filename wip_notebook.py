import argparse
import sys
import MDAnalysis as mda
from MDAnalysis.analysis import leaflet
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering

from chl_angle_components_density import generate_coords_comps_table
from modules.general import flatten, duration, sparkles, print_1line
from modules.traj import System, TrajectorySlice
from modules.constants import PATH, EXPERIMENTS


# %%
systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                   for i in flatten(EXPERIMENTS.values())])
trj_slices = [TrajectorySlice(
    System(PATH, s), 199, 200, 10) for s in systems]


# %%

records = []
print('collecting coords...')

for trj in trj_slices:
    print(trj)

    trj.generate_slice_with_gmx()
    u = mda.Universe(f'{trj.system.dir}/md/md.tpr',
                     f'{trj.system.dir}/md/pbcmol_{trj.b}-{trj.e}-{trj.dt}.xtc')

    for ts in u.trajectory:
        cutoff, n = leaflet.optimize_cutoff(
            u, 'name P* or name O3', dmin=7, dmax=17)
        print_1line(f'cutoff {cutoff} A, {n} groups')
        leaflet_ = leaflet.LeafletFinder(
            u, 'name P* or name O3', pbc=True, cutoff=cutoff)
        if len(leaflet_.groups()) != 2:
            print(f'{len(leaflet_.groups())} groups found...')
        leaflet_0 = leaflet_.group(0)
        leaflet_1 = leaflet_.group(1)
        zmem = 0.5 * (leaflet_1.centroid() + leaflet_0.centroid())[2]
        phosphors = u.select_atoms("name P*")
        for i in phosphors:
            records.append((
                trj.system.name.split(
                    '_chol', 1)[0] if '_chol' in trj.system.name else trj.system.name,
                int(trj.system.name.split(
                    '_chol', 1)[1]) if '_chol' in trj.system.name else 0,
                ts.time,
                i.resname,
                i.position[0],
                i.position[1],
                i.position[2],
                'upper' if i in leaflet_0 else (
                    'lower' if i in leaflet_1 else 'na'),
                zmem,
                ts.dimensions[0],
                ts.dimensions[1],
                ts.dimensions[2]))

df = pd.DataFrame.from_records(records, columns=[
    'system',
    'CHL amount, %',
    'timepoint',
    'lipid',
    'x_p',
    'y_p',
    'z_p',
    'monolayer',
    'zmem',
    'x_box',
    'y_box',
    'z_box'
])

df.to_csv(PATH / 'notebooks' / 'gclust' /
          f'PL_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}_coords.csv')



# %%
thresh=3
mol='PL'
# %%
all_counts = pd.read_csv(PATH / 'notebooks' / 'gclust' /
                         f'{mol}_clusters_{trj_slices[0].b}-{trj_slices[0].e}-'
                         f'{trj_slices[0].dt}_thresh_{thresh}.csv')

df2 = all_counts.groupby(
    ['timepoint', 'system', 'CHL amount, %', 'monolayer']).agg(
    cluster_size=('label', 'value_counts')).reset_index()

max_clsize = 70
# %%
sns.set(style='ticks', context='talk', palette='muted')

# %%
palette = sns.color_palette('RdYlGn_r',4)

for exp in EXPERIMENTS:
    fig, axs = plt.subplots(1, 3, figsize=(
        24, 7), sharex=True, sharey=True)
    for syst, ax in zip(EXPERIMENTS[exp], axs):
        hists = {}
        for chol_amount in df2['CHL amount, %'].unique():
            hists[chol_amount] = np.histogram(
                df2[(df2['system'] == syst) &
                    (df2['CHL amount, %'] == chol_amount)]['cluster_size'],
                bins=np.arange(1, max_clsize, 4), density=True)

        width = 1

        ax.bar(hists[0][1][:-1] - width, hists[0][0], color=palette[0],
               width=width, ec='k', label='0% of CHL')
        ax.bar(hists[10][1][:-1], hists[10][0], color=palette[1],
               width=width, ec='k', label='10% of CHL')
        ax.bar(hists[30][1][:-1] + width, hists[30][0], color=palette[2],
               width=width, ec='k', label='30% of CHL')
        ax.bar(hists[50][1][:-1] + 2 * width, hists[50][0], color=palette[3],
               width=width, ec='k', label='50% of CHL')
        # ax.set_xlim(left=0)
        ax.set_xlabel('Cluster size (n of molecules)')
        ax.set_title(syst)
        # ax.set_yscale('log')
    axs[0].set_ylabel('Density')
    axs[1].legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -.32))
    fig.suptitle(f'{exp}, threshold={thresh} Å')
print('done.')
