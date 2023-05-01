#!/usr/bin/python3

# pylint:disable=protected-access

import argparse
import logging
import os
import shutil
import subprocess
import sys
from datetime import timedelta as td
from pathlib import Path

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
from MDAnalysis.analysis import leaflet
from modules.constants import EXPERIMENTS, TO_RUS
from modules.general import initialize_logging  # , multiproc)
from modules.general import duration, flatten, progress_bar, sparkles
from modules.tg_bot import send_message
from rich.traceback import install
from scipy.ndimage import gaussian_filter, label, uniform_filter, zoom
from scipy.stats import gaussian_kde

# NOW pbcmol.xtc IS REPLACED WITH pbcmol_201.xtc,
# turn it back to use full dt=10 trajectories


def mhpmap_obtainer(path, syst, b, e, dt, force=False):
    '''
    obtain mhp map for specified system
    '''
    desired_datafile = path / 'notebooks' / 'mhpmaps' / \
        'data' / f'{syst}_{b}-{e}-{dt}_data.nmp'
    if desired_datafile.is_file() and force is False:
        logging.info(
            'data for %s %s-%s ns, dt=%s ps already calculated, skipping...',
            syst, b, e, dt)
    else:
        logging.info('🗄️ system:\t%s\n⌚️ time:\t%s-%s ns, dt=%s ps',
                     syst, b, e, dt)

        (path / syst / 'mhp').mkdir(parents=True, exist_ok=True)
        os.chdir(path / syst / 'mhp')
        tpr = str(path / syst / 'md' / '201_ns.tpr')
        xtc = str(path / syst / 'md' / 'pbcmol_201.xtc')

        args = f'TOP={str(tpr)}\nTRJ={str(xtc)}' \
            f'\nBEG={int(b*1000)}\nEND={int(b*1000)}\nDT={dt}\nNX=150\nNY=150' \
            f'\nMAPVAL="M"\nMHPTBL="98"\nPRJ="P"\nUPLAYER=1\nMOL="lip///"' \
            '\nSURFSEL=$MOL\nPOTSEL=$MOL\nDUMPDATA=1\nNOIMG=1'

        with open('args', 'w', encoding='utf-8') as f:
            f.write(args)

        logging.info('calculating mhp 👨‍💻 maps 🗾 ...')

        impulse = Path('/nfs/belka2/soft/impulse/dev/inst/runtask.py')
        prj = Path(
            '/home/krylov/Progs/IBX/AMMP/test/postpro/maps/galaxy/new/prj.json')
        subprocess.run([impulse, '-f', Path.cwd() /
                       'args', '-t', prj], check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT)

        shutil.copy(
            (Path.cwd() / '1_data.nmp'),
            path / 'notebooks' / 'mhpmaps' / 'data' /
            f'{syst}_{b}-{e}-{dt}_data.nmp'
        )
        shutil.copy(
            (Path.cwd() / '1_pa.nmp'),
            path / 'notebooks' / 'mhpmaps' /
            'data' / f'{syst}_{b}-{e}-{dt}_pa.nmp'
        )
        logging.info('done ✅\n')


def calc_fractions(path, systems_list, times, timedelta, dt):
    sys_names = []
    timesteps = []
    phob_frs = []
    phil_frs = []
    neutr_frs = []
    for syst in systems_list:
        logging.info(syst)
        for t in times:
            try:
                data = np.load(path / 'notebooks' / 'mhpmaps' /
                               'data' / f'{syst}_{t}-{t+timedelta}-{dt}_data.nmp')['data']
            except FileNotFoundError:
                logging.error('file does not exist')
                continue
            logging.info(t)
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

        logging.info('')

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

    logging.info('done ✅\n')


def draw_bars(ax, df, systs, width, pos_definitions, pos, alpha, chl_label, show_label=False):
    if show_label:
        ax.bar(pos_definitions[pos][0], df.loc[systs, :]['phob_fr'], width,
               yerr=df.loc[systs, :]['phob_std'], capsize=5,
               label='phob ' + chl_label, color='C0', alpha=alpha)
        ax.bar(pos_definitions[pos][1], df.loc[systs, :]['neutr_fr'], width,
               yerr=df.loc[systs, :]['neutr_std'], capsize=5,
               label='neutr ' + chl_label, color='C1', alpha=alpha)
        ax.bar(pos_definitions[pos][2], df.loc[systs, :]['phil_fr'], width,
               yerr=df.loc[systs, :]['phil_std'], capsize=5,
               label='phil ' + chl_label, color='C2', alpha=alpha)
    else:
        ax.bar(pos_definitions[pos][0], df.loc[systs, :]['phob_fr'], width,
               yerr=df.loc[systs, :]['phob_std'], capsize=5, color='C0', alpha=alpha)
        ax.bar(pos_definitions[pos][1], df.loc[systs, :]['neutr_fr'], width,
               yerr=df.loc[systs, :]['neutr_std'], capsize=5, color='C1', alpha=alpha)
        ax.bar(pos_definitions[pos][2], df.loc[systs, :]['phil_fr'], width,
               yerr=df.loc[systs, :]['phil_std'], capsize=5, color='C2', alpha=alpha)

    ax.bar(pos_definitions[pos][0], df.loc[systs, :]['phob_fr'], width,
           yerr=df.loc[systs, :]['phob_std'], capsize=5, ec='k', fill=False, lw=2)
    ax.bar(pos_definitions[pos][1], df.loc[systs, :]['neutr_fr'], width,
           yerr=df.loc[systs, :]['neutr_std'], capsize=5, ec='k', fill=False, lw=2)
    ax.bar(pos_definitions[pos][2], df.loc[systs, :]['phil_fr'], width,
           yerr=df.loc[systs, :]['phil_std'], capsize=5, ec='k', fill=False, lw=2)


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
        draw_bars(ax, df, experiments[exp],
                  width, pos_definitions, 1, 1, '0% ХС', show_label)
        draw_bars(ax, df, [i + '_chol10' for i in experiments[exp]],
                  width, pos_definitions,
                  2, 0.5, '10% ХС', show_label)
        draw_bars(ax, df, [i + '_chol30' for i in experiments[exp]],
                  width, pos_definitions,
                  3, 0.3, '30% ХС', show_label)
        draw_bars(ax, df, [i + '_chol50' for i in experiments[exp]],
                  width, pos_definitions,
                  4, 0.1, '50% ХС', show_label)

    elif exp == 'head polarity':
        x = np.arange(len(experiments[exp]))
        width = 0.12
        pos_definitions = {
            1: (x - 2 * width, x, x + 2 * width),
            2: (x - width, x + width, x + 3 * width),
        }
        draw_bars(ax, df, experiments[exp],
                  width, pos_definitions, 1, 1, '0% ХС', show_label)
        draw_bars(ax, df, [i + '_chol30' for i in experiments[exp]],
                  width, pos_definitions,
                  2, 0.3, '30% ХС', show_label)

    else:
        raise ValueError('unknown exp value')

    x = np.arange(len(experiments[exp]))
    ax.set_title(TO_RUS[exp])
    ax.xaxis.set_ticks(x)
    ax.set_xticklabels([TO_RUS[i] for i in experiments[exp]])


def draw_chl_area_exp(ax, df, experiments, exp, show_label):
    if exp in ['chain length', 'chain saturation']:
        x = np.arange(len(experiments[exp]))
        width = 0.1
        pos_definitions = {
            1: (x - 4 * width, x - width, x + 2 * width),
            2: (x - 3 * width, x, x + 3 * width),
            3: (x - 2 * width, x + width, x + 4 * width),
        }
        draw_bars(ax, df, [i + '_chol10' for i in experiments[exp]],
                  width, pos_definitions, 1, 1, '10% ХС', show_label)
        draw_bars(ax, df, [i + '_chol30' for i in experiments[exp]],
                  width, pos_definitions,
                  2, 0.5, '30% ХС', show_label)
        draw_bars(ax, df, [i + '_chol50' for i in experiments[exp]],
                  width, pos_definitions,
                  3, 0.3, '50% ХС', show_label)

    elif exp == 'head polarity':
        x = np.arange(len(experiments[exp]))
        width = 0.2
        pos_definitions = {
            1: (x - width, x, x + width),
        }
        draw_bars(ax, df, [i + '_chol30' for i in experiments[exp]],
                  width, pos_definitions,
                  1, 0.5, '30% ХС', show_label)

    else:
        raise ValueError('unknown exp value')

    x = np.arange(len(experiments[exp]))
    ax.set_title(TO_RUS[exp])
    ax.xaxis.set_ticks(x)
    ax.set_xticklabels([TO_RUS[i] for i in experiments[exp]])


def calc_chl_fractions(path, systems_list, times, timedelta, dt):

    def obtain_chol_atom_ids(path, syst):
        logging.info('obtaining chl ids...')
        u = mda.Universe(f'{path}/{syst}/md/md.tpr',
                         f'{path}/{syst}/md/md_190000.xtc', refresh_offsets=True)
        cutoff, _ = leaflet.optimize_cutoff(
            u, 'name P* or name O3', dmin=7, dmax=17)
        leaflet_ = leaflet.LeafletFinder(
            u, 'name P* or name O3', pbc=True, cutoff=cutoff)
        if len(leaflet_.groups()) != 2:
            raise ValueError('Created more than 2 leaflets! '
                             '({len(leaflet_.groups())} groups found)')
        leaflet0 = leaflet_.group(0)

        # leaflet1 = L.groups(1)

        upper = leaflet0.residues.atoms
        # lower = leaflet1.residues.atoms

        return upper.residues[upper.residues.resnames == 'CHL'].atoms.ix

    def chl_percentage(mask, at_flat, map_flat, chol_atom_ids, result=None):
        if result is None:
            result = []
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
        logging.info('🗄️ system:\t%s', syst)

        phob_chol_perc = []
        neutral_chol_perc = []
        phil_chol_perc = []

        cmd = 'source `ls -t /usr/local/gromacs*/bin/GMXRC | head -n 1 ` && ' \
            f'echo 0 | gmx trjconv -f {path}/{syst}/md/pbcmol_201.xtc ' \
            f'-s {path}/{syst}/md/md.tpr ' \
            f'-b 190000 -e 190001 -dt 1 -o {path}/{syst}/md/md_190000.xtc'
        os.popen(cmd).read()

        chol_atom_ids = obtain_chol_atom_ids(path, syst)

        for ts in times:
            logging.info('⏱️ time:\t%s-%s ns', ts, ts + timedelta)
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

        logging.info('')

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

    logging.info('\n🏁')

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


def single_kdeplot(ax, data, color, linestyle, label_, cov_factor=.01):
    x = np.arange(np.min(data), np.max(data),
                  (np.max(data) - np.min(data)) / 500)
    density = gaussian_kde(data)
    density.covariance_factor = lambda: cov_factor
    density._compute_covariance()
    ax.plot(x, density(x), lw=2, color=color, ls=linestyle, label=label_)


def single_histplot(ax, data, color, linestyle, label_, lt=False):
    if lt:
        kwargs = {'edgecolor': color, 'linestyle': linestyle, 'linewidth': 2, }
        ax.hist(data, histtype='step', density=True, label=label_, **kwargs)
    else:
        hist, bins = np.histogram(data, bins=10, density=True)
        ax.plot((bins[1:] + bins[:-1]) / 2, hist, lw=2,
                color=color, ls=linestyle, label=label_)


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
        logging.info('plotting %s...', syst)
        data = collect_mhp_values_from_timepoints(
            path, syst, [times[-2]], deltat, dt)
        if ax == axs[0]:
            single_kdeplot(ax, data, 'C0', '-', '0% ХС')
        else:
            single_kdeplot(ax, data, 'C0', '-', None)
        ax.set_title(TO_RUS[syst])
        ax.set_ylim(0)
        ax.set_xlabel('MГП, log P')

    if exp != 'head polarity':
        for syst, ax in zip(systs_chol10, axs):
            logging.info('plotting %s...', syst)
            data = collect_mhp_values_from_timepoints(
                path, syst, [times[-2]], deltat, dt)
            if ax == axs[0]:
                single_kdeplot(ax, data, 'C1', '-.', '10% ХС')
            else:
                single_kdeplot(ax, data, 'C1', '-.', None)

    for syst, ax in zip(systs_chol30, axs):
        logging.info('plotting %s...', syst)
        data = collect_mhp_values_from_timepoints(
            path, syst, [times[-2]], deltat, dt)
        if ax == axs[0]:
            single_kdeplot(ax, data, 'C2', '--', '30% ХС')
        else:
            single_kdeplot(ax, data, 'C2', '--', None)

    if exp != 'head polarity':
        for syst, ax in zip(systs_chol50, axs):
            logging.info('plotting %s...', syst)
            data = collect_mhp_values_from_timepoints(
                path, syst, [times[-2]], deltat, dt)
            if ax == axs[0]:
                single_kdeplot(ax, data, 'C3', ':', '50% ХС')
            else:
                single_kdeplot(ax, data, 'C3', ':', None)

    axs[0].set_ylabel('Плотность вероятности')
    fig.suptitle(TO_RUS[exp])
    fig.legend(bbox_to_anchor=(1.04, 1))
    fname = '_'.join(exp.split()) + f'_mhp_hists_dt{dt}_rus.png'
    plt.savefig(path / 'notebooks' / 'mhpmaps' / 'imgs' /
                fname, bbox_inches='tight', dpi=300)


def clust(path, syst, b, e, dt, option='hydrophobic', force=False, area_threshold=0.5):
    desired_datafiles = (
        Path(path) / 'notebooks' / 'mhpmaps' / 'clust' / f'{syst}_{b}-{e}-{dt}_{option}.txt',
        Path(path) / 'notebooks' / 'mhpmaps' / 'clust' / f'{syst}_{b}-{e}-{dt}_{option}_lt.txt'
    )

    if [i.is_file() for i in desired_datafiles] == [True, True] and force is False:
        logging.info(
            'data for %s %s-%s ns, dt=%s ps, %s already calculated, skipping...',
            syst, b, e, dt, option)

    else:

        def single_clust(mapp, bside, option='hydrophobic'):
            matrix = gaussian_filter(mapp, sigma=1)
            shrinkby = mapp.shape[0] / bside
            matrix_filtered = uniform_filter(input=matrix, size=shrinkby)
            matrix = zoom(input=matrix_filtered, zoom=1. / shrinkby, order=0)

            # MHP>=0.5 => hydrophobic, MHP<=-0.5 => hydrophilic
            if option == 'hydrophilic':
                threshold_indices = matrix > -0.5
            if option == 'hydrophobic':
                threshold_indices = matrix < 0.5
            matrix[threshold_indices] = 0
            s = [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]]
            clusters, _ = label(matrix, structure=s)

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
                ID = 1

                def __init__(self, clusters_list_slice, l):
                    self.id = Cluster.ID
                    self.init_frame = clusters_list_slice
                    self.label = l
                    self.position = np.transpose(
                        np.nonzero(self.init_frame == self.label))
                    self.flat_position = np.flatnonzero(
                        self.init_frame == self.label)
                    self.size = len(self.position)
                    self.lifetime = 0
                    Cluster.ID += 1

                def __repr__(self):
                    a = np.zeros_like(self.init_frame)
                    np.put(a, self.flat_position, 1)
                    return f'Cluster {self.id}:\n(' + np.array2string(a) + f', lt={self.lifetime})'

                def same_as(self, other):
                    return len(set(self.flat_position)
                               & set(other.flat_position)) * (1 / area_threshold) \
                        > len(self.flat_position) \
                        and len(set(self.flat_position)
                                & set(other.flat_position)) * (1 / area_threshold) \
                        > len(other.flat_position)

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

        logging.info('🗄️ system:\t%s\n⌚️ time:\t%s-%s ns', syst, b, e)
        logging.info('looking for 🔎 %s regions 🗺️ ...', option)

        mapp = np.load(
            f'{path}/notebooks/mhpmaps/data/{syst}_{b}-{e}-{dt}_data.nmp')['data']

        gmxutils = '/nfs/belka2/soft/impulse/dev/inst/gmx_utils.py'
        tpr_props = os.popen(f'{gmxutils} {path}/{syst}/md/md.tpr').read()
        box_side = float(tpr_props.split('\n')[3].split()[2])

        logging.info('clustering 🧩 ...')

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

        logging.info('calculating 👨‍💻 cluster 🧩 lifetimes ⏳️ ...')

        lifetimes = cluster_lifetimes(
            clusters_list, labels_list, dt, area_threshold)

        np.savetxt(f'{path}/notebooks/mhpmaps/clust/{syst}_{b}-{e}-{dt}_{option}_lt.txt',
                   np.array(lifetimes))
        logging.info('done ✅\n')


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
        logging.info('plotting %s...', syst)
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
            logging.info('plotting %s...', syst)
            data = collect_mhp_clust(path, times, syst, deltat, dt, option, lt)
            data = data[data < np.quantile(
                data, .95)] if not lt else data[data < np.quantile(
                    data, .99)]
            if ax == axs[0]:
                single_histplot(ax, data, 'C1', '-.', '10% CHL', lt)
            else:
                single_histplot(ax, data, 'C1', '-.', None, lt)

    for syst, ax in zip(systs_chol30, axs):
        logging.info('plotting %s...', syst)
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
            logging.info('plotting %s...', syst)
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
                        help='calculate and save phob phil and neutral '
                        'fractions containing CHL to file')
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
    parser.add_argument('--n_workers', type=int, default=8,
                        help='n of workers for multiprocessing')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='show debugging messages')

    if len(sys.argv) < 2:
        parser.logging.info_usage()

    args = parser.parse_args()

    initialize_logging('mhp_stats.log', args.verbose)
    sns.set(style='ticks', context='talk', palette='muted')
    chol_impact = Path('/home/klim/Documents/chol_impact')

    systems_list = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                            for i in flatten(EXPERIMENTS.values())])
    systems_list = list(dict.fromkeys(systems_list))

    experiments = EXPERIMENTS

    # times = [49, 99, 149, 199]
    # dt = 10
    # deltat = 1
    times = [int(i) for i in args.times]
    dt = args.dt
    deltat = args.deltat

    if args.obtain:
        with progress_bar as p:
            for s in p.track(systems_list, description='obtaining mhp data'):
                for t in times:
                    if args.force:
                        mhpmap_obtainer(chol_impact, s, t,
                                        t + deltat, dt, True)
                    else:
                        mhpmap_obtainer(chol_impact, s, t, t + deltat, dt)
                task = p.tasks[0]
                send_message(
                    f'*{task.description}*: \n'
                    f'`{s}` done; {int(task.completed)}/{int(task.total)} steps completed',
                    silent=True)
            send_message(
                f'task *{task.description}* completed\n'
                f'finished time: {str(td(seconds=task.finished_time))} s\n'
                f'finished speed: {round(task.speed, 2): .2f} it/s', silent=True)

    if args.calc_fractions:
        logging.info('[calculating fractions area]')
        calc_fractions(chol_impact, systems_list, times, deltat, dt)

    if args.calc_chl_fractions:
        logging.info('[calculating cholesterol fractions area]')
        calc_chl_fractions(chol_impact, [i for i in systems_list if 'chol' in i],
                           times, deltat, dt)

    if args.plot_area_hists:
        logging.info('[plotting area hists]')
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
        axs[0].set_ylabel('% площади')
        fig.legend(bbox_to_anchor=(1.04, 1))
        # fig.suptitle(
        #     'Доля площади поверхности липидного бислояя, занимаемая участками различной полярности')
        plt.savefig(chol_impact / 'notebooks' / 'mhpmaps' / 'imgs' /
                    f'mhp_hists_area_dt{dt}_rus.png', bbox_inches='tight', dpi=300)
        logging.info('done.')

    if args.plot_chl_area_hists:
        logging.info('[plotting cholesterol area hists]')
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
        axs[0].set_ylabel('% площади')
        fig.legend(bbox_to_anchor=(1.04, 1))
        # fig.suptitle(
        #     'Доля площади поверхности липидного бислоя, занимаемая участками различной полярности,'
        #     ' состоящих из атомов ХС')
        plt.savefig(chol_impact / 'notebooks' / 'mhpmaps' / 'imgs' /
                    f'mhp_hists_chl_area_dt{dt}_rus.png', bbox_inches='tight', dpi=300)
        logging.info('done.')

    if args.plot_mhp_hists:
        logging.info('[plotting mhp hists]')
        for exp in experiments:
            logging.info('plotting %s...', exp)
            plot_mhp_kdeplots(chol_impact, experiments, exp, times, deltat, dt)
            logging.info('')

    if args.calc_mhp_values_mean:
        logging.info('[calculating mhp values mean]')
        systs = []
        mhp_mean = []
        mhp_std = []

        for syst in systems_list:
            logging.info(syst)
            systs.append(syst)
            data = collect_mhp_values_from_timepoints(
                chol_impact, syst, times, deltat, dt)
            mhp_mean.append(np.mean(data))
            mhp_std.append(np.std(data))

        df = pd.DataFrame(list(zip(systs, mhp_mean, mhp_std)),
                          columns=['system', 'mhp_mean', 'mhp_std'])

        df.to_csv(chol_impact / 'notebooks' / 'mhpmaps' /
                  'mhp_mean_values.csv', index=False)

        logging.info('plotting...')
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

        logging.info('done.')

    if args.calc_mhp_clusters:
        logging.info('[calculating mhp clusters]')
        with progress_bar as p:
            for s in p.track(systems_list, description='calculating mhp clusters'):
                task = p.tasks[0]
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
                send_message(
                    f'*{task.description}*: \n'
                    f'`{s}` done; {int(task.completed)}/{int(task.total)} steps completed',
                    silent=True)
            send_message(
                f'task *{task.description}* completed\n'
                f'finished time: {str(td(seconds=task.finished_time))} s\n'
                f'finished speed: {round(task.speed, 2): .2f} it/s', silent=True)

        # beg = flatten([times for _ in systems_list])
        # end = flatten([[t+deltat for t in times] for _ in systems_list])
        # dt = [dt for _ in systems_list]
        # option = ['hydrophobic' for _ in systems_list]
        #
        # multiproc(clust, systems_list, beg, end, dt, option, n_workers=args.n_workers,
        #           descr='mhp clustering (hydrophobic)', messages=True)
        #
        # option = ['hydrophilic' for _ in systems_list]
        # multiproc(clust, systems_list, beg, end, dt, option, n_workers=args.n_workers,
        #           descr='mhp clustering (hydrophilic)', messages=True)

    if args.plot_mhp_clusters:
        logging.info('[plotting fractions area]')
        for exp in experiments:
            logging.info('plotting %s...', exp)
            logging.info('hydrophobic size...')
            plot_mhp_clust_kdeplots(chol_impact, experiments, exp, times, deltat, dt,
                                    option='hydrophobic', lt=False)
            logging.info('hydrophilic size...')
            plot_mhp_clust_kdeplots(chol_impact, experiments, exp, times, deltat, dt,
                                    option='hydrophilic', lt=False)
            logging.info('hydrophobic lifetime...')
            plot_mhp_clust_kdeplots(chol_impact, experiments, exp, times, deltat, dt,
                                    option='hydrophobic', lt=True)
            logging.info('hydrophilic lifetime...')
            plot_mhp_clust_kdeplots(chol_impact, experiments, exp, times, deltat, dt,
                                    option='hydrophilic', lt=True)
            logging.info('')

    if args.calc_mhp_clusts_mean:
        logging.info('[calculating mhp clusts mean]')
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
            logging.info(syst)
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

        logging.info('plotting...')
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

        logging.info('done.')


if __name__ == '__main__':
    install()  # rich traceback
    main()
