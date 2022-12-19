#!/usr/bin/python3


import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.ndimage import zoom, label, gaussian_filter, uniform_filter
import matplotlib.pyplot as plt
import seaborn as sns
import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder


# %%

def mhpmap_obtainer(path, syst, b, e, dt, force=False):
    desired_datafile = path / 'notebooks' / 'mhpmaps' / \
        'data' / f'{syst}_{b}-{e}-{dt}_data.nmp'
    if desired_datafile.is_file() and force is False:
        print(
            f' data for {syst} {b}-{e} ns, dt={dt} ps already calculated, skipping...')
    else:
        print(f'🗄️ system:\t{syst}\n⌚️ time:\t{b}-{e} ns, dt={dt} ps')
        os.chdir(path / 'tmp')
        tpr = path / 'tmp' / 'md.tpr'
        xtc = path / 'tmp' / 'md.xtc'

        tpr.unlink(missing_ok=True)
        xtc.unlink(missing_ok=True)
        os.symlink(path / syst / 'md' / 'md.tpr', tpr)
        os.symlink(path / syst / 'md' / 'pbcmol.xtc', xtc)

        args = f'TOP={str(tpr)}\nTRJ={str(xtc)}' \
            f'\nBEG={b}000\nEND={e}000\nDT={dt}\nNX=150\nNY=150' \
            f'\nMAPVAL="M"\nMHPTBL="98"\nPRJ="P"\nUPLAYER=1\nMOL="lip///"' \
            '\nSURFSEL=$MOL\nPOTSEL=$MOL\nDUMPDATA=1\nNOIMG=1'

        with open('args', 'w') as f:
            f.write(args)

        print('calculating 👨‍💻 maps 🗾 ...')

        impulse = Path('/nfs/belka2/soft/impulse/dev/inst/runtask.py')
        prj = Path(
            '/home/krylov/Progs/IBX/AMMP/test/postpro/maps/galaxy/new/prj.json')
        subprocess.run([impulse, '-f', Path.cwd() / 'args', '-t', prj])

        (Path.cwd() / '1_data.nmp').replace(
            path / 'notebooks' / 'mhpmaps' / 'data' /
            f'{syst}_{b}-{e}-{dt}_data.nmp'
        )
        (Path.cwd() / '1_pa.nmp').replace(
            path / 'notebooks' / 'mhpmaps' /
            'data' / f'{syst}_{b}-{e}-{dt}_pa.nmp'
        )
        for i in Path.cwd().glob('1_*'):
            i.unlink()
        print('done ✅\n')


def calc_fractions(path, systems_list, times, timedelta, dt):
    sys_names = []
    timesteps = []
    phob_frs = []
    phil_frs = []
    neutr_frs = []
    for syst in systems_list:
        print(syst)
        for t in times:
            data = np.load(path / 'notebooks' / 'mhpmaps' /
                           'data' / f'{syst}_{t}-{t+timedelta}-{dt}_data.nmp')['data']
            print(t)
            for i in data:
                sys_names.append(syst)
                timesteps.append(f'{t}-{t+timedelta}')
                i = i.flatten()
                phob = i[i >= 0.5].shape[0]
                phil = i[i <= -0.5].shape[0]
                neutr = i.shape[0] - phil - phob
                phob_frs.append(phob / i.shape[0])
                phil_frs.append(phil / i.shape[0])
                neutr_frs.append(neutr / i.shape[0])
        print()

        columns = ['system_name',
                   'timestep (ns)', 'phob_fr', 'phil_fr', 'neutr_fr']
        df = pd.DataFrame(list(zip(sys_names, timesteps, phob_frs, phil_frs, neutr_frs)),
                          columns=columns)
        df.to_csv(path / 'notebooks' / 'mhpmaps' /
                  'for_hists_fractions.csv', index=False)

        df_stats = (df.groupby(['system_name'], as_index=False).agg('mean')
                    .assign(phob_std=df.groupby(['system_name']).agg(np.std)['phob_fr'].values)
                    .assign(phil_std=df.groupby(['system_name']).agg(np.std)['phil_fr'].values)
                    .assign(neutr_std=df.groupby(['system_name']).agg(np.std)['neutr_fr'].values))

        df_stats.to_csv(path / 'notebooks' / 'mhpmaps' /
                        'for_hists_fractions_stats.csv', index=False)

    print('done ✅\n')


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def draw_bars(ax, df, systs, x, width, pos_definitions, pos, alpha, chl_label, show_label=False):
    if show_label:
        ax.bar(pos_definitions[pos][0], df.loc[systs, 'phob_fr'], width,
               yerr=df.loc[systs, 'phob_std'], capsize=5, label='phob ' + chl_label, color='C0', alpha=alpha)
        ax.bar(pos_definitions[pos][1], df.loc[systs, 'neutr_fr'], width,
               yerr=df.loc[systs, 'neutr_std'], capsize=5, label='neutr ' + chl_label, color='C1', alpha=alpha)
        ax.bar(pos_definitions[pos][2], df.loc[systs, 'phil_fr'], width,
               yerr=df.loc[systs, 'phil_std'], capsize=5, label='phil ' + chl_label, color='C2', alpha=alpha)
    else:
        ax.bar(pos_definitions[pos][0], df.loc[systs, 'phob_fr'], width,
               yerr=df.loc[systs, 'phob_std'], capsize=5, color='C0', alpha=alpha)
        ax.bar(pos_definitions[pos][1], df.loc[systs, 'neutr_fr'], width,
               yerr=df.loc[systs, 'neutr_std'], capsize=5, color='C1', alpha=alpha)
        ax.bar(pos_definitions[pos][2], df.loc[systs, 'phil_fr'], width,
               yerr=df.loc[systs, 'phil_std'], capsize=5, color='C2', alpha=alpha)

    ax.bar(pos_definitions[pos][0], df.loc[systs, 'phob_fr'], width,
           yerr=df.loc[systs, 'phob_std'], capsize=5, ec='k', fill=False, lw=2)
    ax.bar(pos_definitions[pos][1], df.loc[systs, 'neutr_fr'], width,
           yerr=df.loc[systs, 'neutr_std'], capsize=5, ec='k', fill=False, lw=2)
    ax.bar(pos_definitions[pos][2], df.loc[systs, 'phil_fr'], width,
           yerr=df.loc[systs, 'phil_std'], capsize=5, ec='k', fill=False, lw=2)


def draw_area_exp(ax, df, experiments, exp, show_label):
    if exp in ['chain length', 'chain saturation']:
        x = np.arange(len(experiments[exp]))
        width = 0.07
        pos_definitions = {
            1: (x - 6 * width, x - 2 * width, x + 2 * width),
            2: (x - 5 * width, x - 1 * width, x + 3 * width),
            3: (x - 4 * width, x, x + 4 * width),
            4: (x - 3 * width, x + 1 * width, x + 5 * width),
        }
        draw_bars(ax, df, experiments[exp], x, width,
                  pos_definitions, 1, 1, '0% CHL', show_label)
        draw_bars(ax, df, [i + '_chol10' for i in experiments[exp]],
                  x, width, pos_definitions,
                  2, 0.5, '10% CHL', show_label)
        draw_bars(ax, df, [i + '_chol30' for i in experiments[exp]],
                  x, width, pos_definitions,
                  3, 0.3, '30% CHL', show_label)
        draw_bars(ax, df, [i + '_chol50' for i in experiments[exp]],
                  x, width, pos_definitions,
                  4, 0.1, '50% CHL', show_label)

    elif exp == 'head polarity':
        x = np.arange(len(experiments[exp]))
        width = 0.12
        pos_definitions = {
            1: (x - 2 * width, x, x + 2 * width),
            2: (x - width, x + width, x + 3 * width),
        }
        draw_bars(ax, df, experiments[exp], x, width,
                  pos_definitions, 1, 1, '0% CHL', show_label)
        draw_bars(ax, df, [i + '_chol30' for i in experiments[exp]],
                  x, width, pos_definitions,
                  2, 0.3, '30% CHL', show_label)

    else:
        raise ValueError('unknown exp value')

    x = np.arange(len(experiments[exp]))
    ax.set_title(exp)
    ax.xaxis.set_ticks(x)
    ax.set_xticklabels(experiments[exp])


def draw_chl_area_exp(ax, df, experiments, exp, show_label):
    if exp in ['chain length', 'chain saturation']:
        x = np.arange(len(experiments[exp]))
        width = 0.1
        pos_definitions = {
            1: (x - 4 * width, x - width, x + 2 * width),
            2: (x - 3 * width, x, x + 3 * width),
            3: (x - 2 * width, x + width, x + 4 * width),
        }
        draw_bars(ax, df, [i + '_chol10' for i in experiments[exp]], x, width,
                  pos_definitions, 1, 1, '10% CHL', show_label)
        draw_bars(ax, df, [i + '_chol30' for i in experiments[exp]],
                  x, width, pos_definitions,
                  2, 0.5, '30% CHL', show_label)
        draw_bars(ax, df, [i + '_chol50' for i in experiments[exp]],
                  x, width, pos_definitions,
                  3, 0.3, '50% CHL', show_label)

    elif exp == 'head polarity':
        x = np.arange(len(experiments[exp]))
        width = 0.2
        pos_definitions = {
            1: (x - width, x, x + width),
        }
        draw_bars(ax, df, [i + '_chol30' for i in experiments[exp]],
                  x, width, pos_definitions,
                  1, 0.5, '30% CHL', show_label)

    else:
        raise ValueError('unknown exp value')

    x = np.arange(len(experiments[exp]))
    ax.set_title(exp)
    ax.xaxis.set_ticks(x)
    ax.set_xticklabels(experiments[exp])


def calc_chl_fractions(path, systems_list, times, timedelta, dt):

    def obtain_chol_atom_ids(path, syst):
        print('obtaining chl ids...')
        u = mda.Universe(f'{path}/{syst}/md/md.tpr',
                         f'{path}/tmp/tmp.xtc', refresh_offsets=True)
        L = LeafletFinder(u, 'name P* or name O3', pbc=True)

        leaflet0 = L.groups(0)
        leaflet1 = L.groups(1)

        upper = leaflet0.residues.atoms
        lower = leaflet1.residues.atoms

        return upper.residues[upper.residues.resnames == 'CHL'].atoms.ix

    def chl_percentage(mask, at_flat, map_flat, chol_atom_ids, result=[]):
        c = 0
        for i in at_flat[np.argwhere(mask)]:
            if i in chol_atom_ids:
                c += 1
        result.append(c * 100 / map_flat[mask].shape[0])

    systs = []
    phob_mean = []
    phob_std = []
    neutral_mean = []
    neutral_std = []
    phil_mean = []
    phil_std = []

    for syst in systems_list:
        print(f'🗄️ system:\t{syst}')

        phob_chol_perc = []
        neutral_chol_perc = []
        phil_chol_perc = []

        cmd = 'source `ls -t /usr/local/gromacs*/bin/GMXRC | head -n 1 ` && ' \
            f'echo 0 | gmx trjconv -f {path}/{syst}/md/pbcmol.xtc -s {path}/{syst}/md/md.tpr ' \
            f'-b 190000 -e 190001 -dt 1 -o {path}/tmp/tmp.xtc'
        os.popen(cmd).read()

        chol_atom_ids = obtain_chol_atom_ids(path, syst)

        for ts in times:
            print(f'⏱️ time:\t{ts}-{ts+timedelta} ns')
            fname = f'{path}/notebooks/mhpmaps/data/{syst}_{ts}-{ts+timedelta}-{dt}'
            mapp = np.load(f'{fname}_data.nmp')['data']
            at_info = np.load(f'{fname}_pa.nmp')['data']

            for s in range(mapp.shape[0]):
                map_flat = mapp[s].flatten()
                at_flat = at_info[s].flatten()

                hdrphb = map_flat >= 0.5
                hdrphl = map_flat <= -0.5
                ntrl = (map_flat > -0.5) & (map_flat < 0.5)

                chl_percentage(hdrphb, at_flat, map_flat,
                               chol_atom_ids, phob_chol_perc)
                chl_percentage(hdrphl, at_flat, map_flat,
                               chol_atom_ids, phil_chol_perc)
                chl_percentage(ntrl, at_flat, map_flat,
                               chol_atom_ids, neutral_chol_perc)

        print()

        systs.append(syst)

        phob_chol_perc = np.array(phob_chol_perc)
        phob_mean.append(np.mean(phob_chol_perc))
        phob_std.append(np.std(phob_chol_perc))

        phil_chol_perc = np.array(phil_chol_perc)
        phil_mean.append(np.mean(phil_chol_perc))
        phil_std.append(np.std(phil_chol_perc))

        neutral_chol_perc = np.array(neutral_chol_perc)
        neutral_mean.append(np.mean(neutral_chol_perc))
        neutral_std.append(np.std(neutral_chol_perc))

    print('\n🏁')

    df = pd.DataFrame(list(zip(systs, phob_mean, phob_std, neutral_mean, neutral_std, phil_mean, phil_std)),
                      columns=['system', 'phob_fr', 'phob_std', 'neutr_fr', 'neutr_std', 'phil_fr', 'phil_std'])

    df.to_csv(path / 'notebooks' / 'mhpmaps' /
              'chol_fractions.csv', index=False)


def collect_mhp_values_from_timepoints(path, syst, times, deltat, dt):
    data = np.array([])
    for t in times:
        data = np.append(data, np.load(
            f'{path}/notebooks/mhpmaps/data/{syst}_{t}-{t+deltat}-{dt}_data.nmp')['data'])
    return data


def single_kdeplot(ax, data, color, linestyle, label, cov_factor=.01):
    x = np.arange(np.min(data), np.max(data),
                  (np.max(data) - np.min(data)) / 500)
    density = gaussian_kde(data)
    density.covariance_factor = lambda: cov_factor
    density._compute_covariance()
    ax.plot(x, density(x), lw=2, color=color, ls=linestyle, label=label)


def single_histplot(ax, data, color, linestyle, label, lt=False):
    if lt:
        kwargs = {'edgecolor': color, 'linestyle': linestyle, 'linewidth': 2, }
        ax.hist(data, histtype='step', density=True, label=label, **kwargs)
    else:
        hist, bins = np.histogram(data, bins=10, density=True)
        ax.plot((bins[1:] + bins[:-1]) / 2, hist, lw=2,
                color=color, ls=linestyle, label=label)


def plot_mhp_kdeplots(path, experiments, exp, times, deltat, dt):
    if exp in ['chain length', 'chain saturation']:
        systs = experiments[exp]
        systs_chol10 = [i + '_chol10' for i in experiments[exp]]
        systs_chol30 = [i + '_chol30' for i in experiments[exp]]
        systs_chol50 = [i + '_chol50' for i in experiments[exp]]
    elif exp == 'head polarity':
        systs = experiments[exp]
        systs_chol30 = [i + '_chol30' for i in experiments[exp]]

    fig, axs = plt.subplots(1, len(experiments[exp]), figsize=(
        7 * len(experiments[exp]), 7), sharey=True, sharex=True)

    for syst, ax in zip(systs, axs):
        print(f'plotting {syst}...')
        data = collect_mhp_values_from_timepoints(
            path, syst, [times[-2]], deltat, dt)
        if ax == axs[0]:
            single_kdeplot(ax, data, 'C0', '-', '0% CHL')
        else:
            single_kdeplot(ax, data, 'C0', '-', None)
        ax.set_title(syst)
        ax.set_ylim(0)
        ax.set_xlabel('MHP, log P')

    if exp != 'head polarity':
        for syst, ax in zip(systs_chol10, axs):
            print(f'plotting {syst}...')
            data = collect_mhp_values_from_timepoints(
                path, syst, [times[-2]], deltat, dt)
            if ax == axs[0]:
                single_kdeplot(ax, data, 'C1', '-.', '10% CHL')
            else:
                single_kdeplot(ax, data, 'C1', '-.', None)

    for syst, ax in zip(systs_chol30, axs):
        print(f'plotting {syst}...')
        data = collect_mhp_values_from_timepoints(
            path, syst, [times[-2]], deltat, dt)
        if ax == axs[0]:
            single_kdeplot(ax, data, 'C2', '--', '30% CHL')
        else:
            single_kdeplot(ax, data, 'C2', '--', None)

    if exp != 'head polarity':
        for syst, ax in zip(systs_chol50, axs):
            print(f'plotting {syst}...')
            data = collect_mhp_values_from_timepoints(
                path, syst, [times[-2]], deltat, dt)
            if ax == axs[0]:
                single_kdeplot(ax, data, 'C3', ':', '50% CHL')
            else:
                single_kdeplot(ax, data, 'C3', ':', None)

    axs[0].set_ylabel('Density')
    fig.suptitle(exp)
    fig.legend(bbox_to_anchor=(1.04, 1))
    fname = '_'.join(exp.split()) + f'_mhp_hists_dt{dt}.png'
    plt.savefig(path / 'notebooks' / 'mhpmaps' / 'imgs' /
                fname, bbox_inches='tight', dpi=300)


def clust(path, syst, b, e, dt, option='hydrophobic', force=False, area_threshold=0.5):
    desired_datafiles = (
        path / 'notebooks' / 'mhpmaps' / 'clust' /
        f'{syst}_{b}-{e}-{dt}_{option}.txt',
        path / 'notebooks' / 'mhpmaps' / 'clust' /
        f'{syst}_{b}-{e}-{dt}_{option}_lt.txt'
    )

    if [i.is_file() for i in desired_datafiles] == [True, True] and force is False:
        print(
            f' data for {syst} {b}-{e} ns, dt={dt} ps, {option} already calculated, skipping...')

    else:

        def single_clust(mapp, bside, option='hydrophobic'):
            M = gaussian_filter(mapp, sigma=1)
            shrinkby = mapp.shape[0] / bside
            Mfilt = uniform_filter(input=M, size=shrinkby)
            M = zoom(input=Mfilt, zoom=1. / shrinkby, order=0)

            # MHP>=0.5 => hydrophobic, MHP<=-0.5 => hydrophilic
            if option == 'hydrophilic':
                threshold_indices = M > -0.5
            if option == 'hydrophobic':
                threshold_indices = M < 0.5
            M[threshold_indices] = 0
            s = [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]]
            clusters, num_features = label(M, structure=s)

            # periodic boundary conditions
            for y in range(clusters.shape[0]):
                if clusters[y, 0] > 0 and clusters[y, -1] > 0:
                    clusters[clusters == clusters[y, -1]] = clusters[y, 0]
            for x in range(clusters.shape[1]):
                if clusters[0, x] > 0 and clusters[-1, x] > 0:
                    clusters[clusters == clusters[-1, x]] = clusters[0, x]
            labels, counts = np.unique(
                clusters[clusters > 0], return_counts=True)
            return counts, labels, clusters

        def cluster_lifetimes(clusters_list, labels_list, dt, area_threshold=0.5):
            class Cluster:
                id = 1

                def __init__(self, clusters_list_slice, l):
                    self.id = Cluster.id
                    self.init_frame = clusters_list_slice
                    self.label = l
                    self.position = np.transpose(
                        np.nonzero(self.init_frame == self.label))
                    self.flat_position = np.flatnonzero(
                        self.init_frame == self.label)
                    self.size = len(self.position)
                    self.lifetime = 0
                    Cluster.id += 1

                def __repr__(self):
                    a = np.zeros_like(self.init_frame)
                    np.put(a, self.flat_position, 1)
                    return f'Cluster {self.id}:\n(' + np.array2string(a) + f', lt={self.lifetime})'

                def same_as(self, other):
                    return len(set(self.flat_position)
                               & set(other.flat_position)) * (1 / area_threshold) > len(self.flat_position) \
                        and len(set(self.flat_position)
                                & set(other.flat_position)) * (1 / area_threshold) > len(other.flat_position)

                def update_cluster(self, other, dt):
                    self.init_frame = other.init_frame
                    self.label = other.label
                    self.lifetime += dt
                    self.flat_position = np.flatnonzero(
                        self.init_frame == self.label)
                    self.position = np.transpose(
                        np.nonzero(self.init_frame == self.label))
                    self.size = len(self.position)

            # {timestep: [clusters which died on current timestep]}
            all_clusters = {i: [] for i in range(len(clusters_list))}

            # initiate clusters on frame 0
            all_clusters[0] = [Cluster(clusters_list[0], labl)
                               for labl in labels_list[0]]

            for ts in range(1, len(clusters_list)):
                new_clusters = [Cluster(clusters_list[ts], labl)
                                for labl in labels_list[ts]]
                old_clusters = all_clusters[ts - 1]

                # updating all_clusters dict
                for c_old in old_clusters:
                    for c_new in new_clusters:
                        if c_old.same_as(c_new):
                            c_old.update_cluster(c_new, dt)
                            all_clusters[ts].append(c_old)
                            all_clusters[ts - 1].remove(c_old)
                            c_new.updated = True

                all_clusters[ts].extend(
                    [c for c in new_clusters if not hasattr(c, 'updated')])

            return [i.lifetime for i in flatten(all_clusters.values())]

        print(f'🗄️ system:\t{syst}\n⌚️ time:\t{b}-{e} ns')
        print(f'looking for 🔎 {option} regions 🗺️ ...')

        mapp = np.load(
            f'{path}/notebooks/mhpmaps/data/{syst}_{b}-{e}-{dt}_data.nmp')['data']

        gmxutils = '/nfs/belka2/soft/impulse/dev/inst/gmx_utils.py'
        tpr_props = os.popen(f'{gmxutils} {path}/{syst}/md/md.tpr').read()
        box_side = float(tpr_props.split('\n')[3].split()[2])

        print('clustering 🧩 ...')

        a = np.array([])
        labels_list = []
        clusters_list = []

        for i in mapp:
            counts, labels, clusters = single_clust(i, box_side, option)
            a = np.hstack((a, counts))
            labels_list.append(labels.tolist())
            clusters_list.append(clusters)

        np.savetxt(
            f'{path}/notebooks/mhpmaps/clust/{syst}_{b}-{e}-{dt}_{option}.txt',
            a)

        print('calculating 👨‍💻 cluster 🧩 lifetimes ⏳️ ...')

        lifetimes = cluster_lifetimes(
            clusters_list, labels_list, dt, area_threshold)

        np.savetxt(f'{path}/notebooks/mhpmaps/clust/{syst}_{b}-{e}-{dt}_{option}_lt.txt',
                   np.array(lifetimes))
        print('done ✅\n')


def collect_mhp_clust(path, times, syst, deltat, dt, option, lt=False):
    data = np.array([])
    for t in times:
        if lt:
            data = np.append(data, np.loadtxt(
                f'{path}/notebooks/mhpmaps/clust/{syst}_{t}-{t+deltat}-{dt}_{option}_lt.txt'))
        else:
            data = np.append(data, np.loadtxt(
                f'{path}/notebooks/mhpmaps/clust/{syst}_{t}-{t+deltat}-{dt}_{option}.txt'))
    return data

# data_phob = [(syst, 'hydrophobic', collect_mhp_clust(
#     chol_impact, times, syst, deltat, dt, 'hydrophobic')) for syst in systems_list]
# data_phil = [(syst, 'hydrophilic', collect_mhp_clust(
#     chol_impact, times, syst, deltat, dt, 'hydrophilic')) for syst in systems_list]
# df = pd.DataFrame.from_records(data_phob + data_phil, columns=['system', 'cluster_type', 'cluster_size'])
# df = df.explode('cluster_size')
# df.system.unique()
# df = df[df['system'].str.contains('20x20')==False].copy()
# df.sort_values('system', inplace=True, ignore_index=True)
# df['CHL amount, %'] = df['system'].str.split('_chol', n=1, expand=True)[1]
# df['system'] = df['system'].str.split('_chol', n=1, expand=True)[0]
# df.replace(to_replace=[None], value=0, inplace=True)
# df = df[(df['system'] != 'dspc') | (
#     (df['CHL amount, %'] != 10) & (df['CHL amount, %'] != 50))]
# from modules.general import get_keys_by_value
# df['experiment'] = df['system'].apply(
#     lambda x: get_keys_by_value(x, experiments))
# df = df.explode('experiment')
# df_p = df[df['cluster_type'] == 'hydrophobic']
# g = sns.FacetGrid(df_p, col='experiment', height=7,
#                   aspect=0.75, sharex=False)
# g.map_dataframe(sns.violinplot, x='system', y='cluster_size', hue='CHL amount, %',
#                 cut=0, palette='RdYlGn_r', inner='quartile')
# g.axes[0][-1].legend(title='CHL amount, %')
# g.set_titles(col_template='{col_name}')


def plot_mhp_clust_kdeplots(path, experiments, exp, times, deltat, dt,
                            option='hydrophobic', lt=False):
    # if lt:
    #     cov_factor = .7

    if exp in ['chain length', 'chain saturation']:
        systs = experiments[exp]
        systs_chol10 = [i + '_chol10' for i in experiments[exp]]
        systs_chol30 = [i + '_chol30' for i in experiments[exp]]
        systs_chol50 = [i + '_chol50' for i in experiments[exp]]
    elif exp == 'head polarity':
        systs = experiments[exp]
        systs_chol30 = [i + '_chol30' for i in experiments[exp]]

    fig, axs = plt.subplots(1, len(experiments[exp]), figsize=(
        7 * len(experiments[exp]), 7), sharey=True, sharex=True)

    for syst, ax in zip(systs, axs):
        print(f'plotting {syst}...')
        data = collect_mhp_clust(path, times, syst, deltat, dt, option, lt)
        data = data[data < np.quantile(
            data, .95)] if not lt else data[data < np.quantile(
                data, .99)]
        if ax == axs[0]:
            single_histplot(ax, data, 'C0', '-', '0% CHL', lt)
        else:
            single_histplot(ax, data, 'C0', '-', None, lt)
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.set_title(syst)
        if lt:
            ax.set_xlabel('cluster lifetime, ps')
        else:
            ax.set_xlabel('cluster size, A²')

    if exp != 'head polarity':
        for syst, ax in zip(systs_chol10, axs):
            print(f'plotting {syst}...')
            data = collect_mhp_clust(path, times, syst, deltat, dt, option, lt)
            data = data[data < np.quantile(
                data, .95)] if not lt else data[data < np.quantile(
                    data, .99)]
            if ax == axs[0]:
                single_histplot(ax, data, 'C1', '-.', '10% CHL', lt)
            else:
                single_histplot(ax, data, 'C1', '-.', None, lt)

    for syst, ax in zip(systs_chol30, axs):
        print(f'plotting {syst}...')
        data = collect_mhp_clust(path, times, syst, deltat, dt, option, lt)
        data = data[data < np.quantile(
            data, .95)] if not lt else data[data < np.quantile(
                data, .99)]
        if ax == axs[0]:
            single_histplot(ax, data, 'C2', '--', '30% CHL', lt)
        else:
            single_histplot(ax, data, 'C2', '--', None, lt)

    if exp != 'head polarity':
        for syst, ax in zip(systs_chol50, axs):
            print(f'plotting {syst}...')
            data = collect_mhp_clust(path, times, syst, deltat, dt, option, lt)
            data = data[data < np.quantile(
                data, .95)] if not lt else data[data < np.quantile(
                    data, .99)]
            if ax == axs[0]:
                single_histplot(ax, data, 'C3', ':', '50% CHL', lt)
            else:
                single_histplot(ax, data, 'C3', ':', None, lt)

    axs[0].set_ylabel('Density')
    fig.suptitle(f'{exp} {option}')
    fig.legend(bbox_to_anchor=(1.04, 1))
    if lt:
        fname = '_'.join(exp.split()) + f'_mhp_clust_dt{dt}_{option}_lt.png'
    else:
        fname = '_'.join(exp.split()) + f'_mhp_clust_dt{dt}_{option}_size.png'
    plt.savefig(path / 'notebooks' / 'mhpmaps' / 'imgs' /
                fname, bbox_inches='tight', dpi=300)


def duration(func):
    def inner(*args, **kwargs):
        start_time = datetime.now()
        func(*args, **kwargs)
        end_time = datetime.now()
        print('\n⌛ duration: {}'.format(end_time - start_time))
    return inner


def sparkles(func):
    def inner(*args, **kwargs):
        print('\n' + '✨' * 30 + '\n')
        func(*args, **kwargs)
        print('\n' + '✨' * 30)
    return inner


@sparkles
@duration
def main():
    parser = argparse.ArgumentParser(
        description='Script to obtain mhp maps and plot hists of mhp values')
    parser.add_argument('--obtain',
                        action='store_true',
                        help='obtain mhp data')
    parser.add_argument('--dt', type=int, default=10,
                        help='dt in ps')
    parser.add_argument('--deltat', type=int, default=1,
                        help='time between starting and ending timepoints in ns')
    parser.add_argument('--times', nargs='+', default=[49, 99, 149, 199],
                        help='starting timepoints in ns')
    parser.add_argument('--force',
                        action='store_true',
                        help='replace mhp data if it exists')
    parser.add_argument('--calc_fractions',
                        action='store_true',
                        help='calculate and save phob phil and neutral fractions to file')
    parser.add_argument('--calc_chl_fractions',
                        action='store_true',
                        help='calculate and save phob phil and neutral fractions containing CHL to file')
    parser.add_argument('--plot_area_hists',
                        action='store_true',
                        help='plot mhp areas histograms')
    parser.add_argument('--plot_chl_area_hists',
                        action='store_true',
                        help='plot mhp areas histograms of CHL atoms')
    parser.add_argument('--plot_mhp_hists',
                        action='store_true',
                        help='plot mhp values histograms')
    parser.add_argument('--calc_mhp_values_mean',
                        action='store_true',
                        help='calculate means and std of mhp values for systems')
    parser.add_argument('--calc_mhp_clusters',
                        action='store_true',
                        help='calculate mhp clusters')
    parser.add_argument('--plot_mhp_clusters',
                        action='store_true',
                        help='plot mhp clusterization stats')
    parser.add_argument('--calc_mhp_clusts_mean',
                        action='store_true',
                        help='calculate means and std of mhp cluster sizes'
                        'and lifetimes for systems')

    if len(sys.argv) < 2:
        parser.print_usage()

    args = parser.parse_args()

    sns.set(style='ticks', context='talk', palette='bright')
    chol_impact = Path('/home/klim/Documents/chol_impact')
    systems_list = sorted([
        '20x20_dmpc',
        'dmpc_chol50',
        'dmpc_chol10',
        'dops_chol30',
        'dppc',
        'dppc_325',
        'dops_chol10',
        'dppc_325_chol10',
        '20x20_dmpc_chol30',
        'dmpc',
        'popc',
        'dopc',
        'dppc_chol30',
        'dmpc_chol30',
        'dops',
        '20x20_popc',
        'dopc_chol50',
        'dspc_chol30',
        'dppc_325_chol30',
        'popc_chol50',
        'dppc_325_chol50',
        '20x20_popc_chol30',
        'dopc_chol30',
        'popc_chol30',
        'dspc_chol10',
        'dspc_chol50',
        'popc_chol10',
        'dspc',
        'dopc_chol10',
        'dops_chol50',
        'dopc_dops20',
        'dopc_dops30',
        'dopc_dops20_chol30',
        'dopc_dops30_chol30'
    ])

    experiments = {
        'chain length': ('dmpc', 'dppc_325', 'dspc'),
        'chain saturation': ('dppc_325', 'popc', 'dopc'),
        'head polarity': ('dopc', 'dopc_dops20', 'dopc_dops30', 'dops'),
    }

    # times = [49, 99, 149, 199]
    # dt = 10
    # deltat = 1
    times = [int(i) for i in args.times]
    dt = args.dt
    deltat = args.deltat

    if args.obtain:
        print('[obtaining mhp data]')
        for s in systems_list:
            for t in times:
                if args.force:
                    mhpmap_obtainer(chol_impact, s, t, t + deltat, dt, True)
                else:
                    mhpmap_obtainer(chol_impact, s, t, t + deltat, dt)

    if args.calc_fractions:
        print('[calculating fractions area]')
        calc_fractions(chol_impact, systems_list, times, deltat, dt)

    if args.calc_chl_fractions:
        print('[calculating cholesterol fractions area]')
        calc_chl_fractions(chol_impact, [i for i in systems_list if 'chol' in i],
                           times, deltat, dt)

    if args.plot_area_hists:
        print('[plotting area hists]')
        df = pd.read_csv(chol_impact / 'notebooks' / 'mhpmaps' /
                         'for_hists_fractions_stats.csv')
        df.sort_values('system_name')
        df = df.set_index(df.columns[0])

        fig, axs = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
        for ax, exp in zip(axs, experiments):
            if ax == axs[0]:
                draw_area_exp(ax, df, experiments, exp, True)
            else:
                draw_area_exp(ax, df, experiments, exp, False)
            ticks, labels = ax.get_xticks(), ax.get_xticklabels()
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=45,
                               ha='right', rotation_mode='anchor')
        axs[0].set_ylabel('% area')
        fig.legend(bbox_to_anchor=(1.04, 1))
        fig.suptitle(
            'Surface area occupied by regions of varying degrees of hydrophobicity')
        plt.savefig(chol_impact / 'notebooks' / 'mhpmaps' / 'imgs' /
                    f'mhp_hists_area_dt{dt}.png', bbox_inches='tight', dpi=300)
        print('done.')

    if args.plot_chl_area_hists:
        print('[plotting cholesterol area hists]')
        df = pd.read_csv(chol_impact / 'notebooks' / 'mhpmaps' /
                         'chol_fractions.csv')
        df.sort_values('system')
        df = df.set_index(df.columns[0])

        fig, axs = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
        for ax, exp in zip(axs, experiments):
            if ax == axs[0]:
                draw_chl_area_exp(ax, df, experiments, exp, True)
            else:
                draw_chl_area_exp(ax, df, experiments, exp, False)
            ticks, labels = ax.get_xticks(), ax.get_xticklabels()
            ax.set_xticks(ticks)
            ax.set_ylim(0)
            ax.set_xticklabels(labels, rotation=45,
                               ha='right', rotation_mode='anchor')
        axs[0].set_ylabel('% area')
        fig.legend(bbox_to_anchor=(1.04, 1))
        fig.suptitle(
            'Surface area composed of CHL atoms occupied by regions of varying degrees of hydrophobicity')
        plt.savefig(chol_impact / 'notebooks' / 'mhpmaps' / 'imgs' /
                    f'mhp_hists_chl_area_dt{dt}.png', bbox_inches='tight', dpi=300)
        print('done.')

    if args.plot_mhp_hists:
        print('[plotting mhp hists]')
        for exp in experiments:
            print(f'plotting {exp}...')
            plot_mhp_kdeplots(chol_impact, experiments, exp, times, deltat, dt)
            print()

    if args.calc_mhp_values_mean:
        print('[calculating mhp values mean]')
        systs = []
        mhp_mean = []
        mhp_std = []

        for syst in systems_list:
            print(syst)
            systs.append(syst)
            data = collect_mhp_values_from_timepoints(
                chol_impact, syst, times, deltat, dt)
            mhp_mean.append(np.mean(data))
            mhp_std.append(np.std(data))

        df = pd.DataFrame(list(zip(systs, mhp_mean, mhp_std)),
                          columns=['system', 'mhp_mean', 'mhp_std'])

        df.to_csv(chol_impact / 'notebooks' / 'mhpmaps' /
                  'mhp_mean_values.csv', index=False)

        print('plotting...')
        df = pd.read_csv(chol_impact / 'notebooks' /
                         'mhpmaps' / 'mhp_mean_values.csv')
        fig, ax = plt.subplots(figsize=(10, 7))
        plt.bar(x=df['system'], height=df['mhp_mean'],
                yerr=df['mhp_std'], capsize=5)
        plt.xticks(rotation=90)
        plt.title('Average MHP values')
        plt.ylabel('MHP, log P')
        plt.xlabel('system')
        plt.savefig(chol_impact / 'notebooks' / 'mhpmaps' / 'imgs' /
                    f'average_mhp_values_dt{dt}.png', bbox_inches='tight', dpi=300)

        print('done.')

    if args.calc_mhp_clusters:
        print('[calculating mhp clusters]')
        for s in systems_list:
            for t in times:
                if args.force:
                    clust(chol_impact, s, t, t + deltat, dt,
                          option='hydrophobic', force=True)
                    clust(chol_impact, s, t, t + deltat, dt,
                          option='hydrophilic', force=True)
                else:
                    clust(chol_impact, s, t, t + deltat, dt,
                          option='hydrophobic', force=False)
                    clust(chol_impact, s, t, t + deltat, dt,
                          option='hydrophilic', force=False)

    if args.plot_mhp_clusters:
        print('[plotting fractions area]')
        for exp in experiments:
            print(f'plotting {exp}...')
            print('hydrophobic size...')
            plot_mhp_clust_kdeplots(chol_impact, experiments, exp, times, deltat, dt,
                                    option='hydrophobic', lt=False)
            print('hydrophilic size...')
            plot_mhp_clust_kdeplots(chol_impact, experiments, exp, times, deltat, dt,
                                    option='hydrophilic', lt=False)
            print('hydrophobic lifetime...')
            plot_mhp_clust_kdeplots(chol_impact, experiments, exp, times, deltat, dt,
                                    option='hydrophobic', lt=True)
            print('hydrophilic lifetime...')
            plot_mhp_clust_kdeplots(chol_impact, experiments, exp, times, deltat, dt,
                                    option='hydrophilic', lt=True)
            print()

    if args.calc_mhp_clusts_mean:
        print('[calculating mhp clusts mean]')
        systs = []
        phob_mean = []
        phob_std = []
        phil_mean = []
        phil_std = []
        phob_lt_mean = []
        phob_lt_std = []
        phil_lt_mean = []
        phil_lt_std = []

        for syst in systems_list:
            print(syst)
            systs.append(syst)
            data = collect_mhp_clust(
                chol_impact, times, syst, deltat, dt, 'hydrophobic', lt=False)
            phob_mean.append(np.mean(data))
            phob_std.append(np.std(data))
            data = collect_mhp_clust(
                chol_impact, times, syst, deltat, dt, 'hydrophilic', lt=False)
            phil_mean.append(np.mean(data))
            phil_std.append(np.std(data))
            data = collect_mhp_clust(
                chol_impact, times, syst, deltat, dt, 'hydrophobic', lt=True)
            data = data[data < np.quantile(data, .98)]
            phob_lt_mean.append(np.mean(data))
            phob_lt_std.append(np.std(data))
            data = collect_mhp_clust(
                chol_impact, times, syst, deltat, dt, 'hydrophilic', lt=True)
            data = data[data < np.quantile(data, .98)]
            phil_lt_mean.append(np.mean(data))
            phil_lt_std.append(np.std(data))

        df = pd.DataFrame(list(zip(systs,
                                   phob_mean,
                                   phob_std,
                                   phil_mean,
                                   phil_std,
                                   phob_lt_mean,
                                   phob_lt_std,
                                   phil_lt_mean,
                                   phil_lt_std,
                                   )),
                          columns=['system',
                                   'phob_mean',
                                   'phob_std',
                                   'phil_mean',
                                   'phil_std',
                                   'phob_lt_mean',
                                   'phob_lt_std',
                                   'phil_lt_mean',
                                   'phil_lt_std',
                                   ])

        df.to_csv(chol_impact / 'notebooks' / 'mhpmaps' /
                  'mhp_clust_mean_values.csv', index=False)

        print('plotting...')
        df = pd.read_csv(chol_impact / 'notebooks' /
                         'mhpmaps' / 'mhp_clust_mean_values.csv')
        fig, axs = plt.subplots(2, 2, figsize=(
            25, 16))
        plt.subplots_adjust(hspace=0.8)
        for i, ax, title in zip(range(1, len(df.columns), 2), axs.flatten(),
                                ['hydrophobic cluster sizes', 'hydrophilic cluster sizes', 'hydrophobic cluster lifetimes', 'hydrophilic cluster lifetimes']):
            ax.bar(x=df['system'], height=df[df.columns[i]],
                   yerr=df[df.columns[i + 1]], capsize=5)
            ax.xaxis.set_tick_params(rotation=90)
            ax.set_ylim(0)
            ax.set_title(title)

        axs.flatten()[0].set_ylabel('cluster size, A²')
        axs.flatten()[2].set_ylabel('cluster lifetime, ps')
        axs.flatten()[2].set_xlabel('system')
        axs.flatten()[3].set_xlabel('system')
        # axs.flatten()[0].sharey(axs.flatten()[1])
        axs.flatten()[2].sharey(axs.flatten()[3])
        plt.savefig(chol_impact / 'notebooks' / 'mhpmaps' / 'imgs' /
                    f'average_mhp_clust_values_dt{dt}.png', bbox_inches='tight', dpi=300)

        print('done.')


if __name__ == '__main__':
    main()
