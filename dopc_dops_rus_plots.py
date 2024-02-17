import os
import subprocess

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import seaborn as sns
from contacts import process_df_for_chl_lip_groups_hb
from contacts_new import get_n_pl_df
from lateral_clusterization import obtain_cluster_labels
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from mhp import plot_mhp_area_single_exp
from mhp_chol_phob_neighbors import get_chl_phob_neighbors_df
from modules.constants import EXPERIMENTS, PATH, TO_RUS
from modules.density import plot_density_profile
from modules.general import chunker, flatten, opener
from modules.traj import System, TrajectorySlice
from relief_mhp_maps import (get_where_chols_in_trj, plot_chols_surface,
                             plot_mhpmap, plot_relief)
from scipy import integrate, stats
from scipy.optimize import curve_fit
from sol_inside_bilayer_hb import get_sol_indexes_df
from z_slices_angles_contacts import get_n_chl_df, merge_hb_or_dc_tables

sns.set(style='ticks', context='talk', palette='muted')
systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                   for i in flatten(EXPERIMENTS.values())])
systems = list(dict.fromkeys(systems))
trj_slices = [TrajectorySlice(System(
    PATH, s), 200.0, 201.0, 1) for s in systems]
# %%
# thickness
df = pd.read_csv(PATH / 'notebooks' / 'integral_parameters' / 'thickness_150-200-1000.csv')

data = df[df['system'].isin(['dopc', 'dops'])].copy()
data['system'] = data['system'].map(TO_RUS)
data.columns = list(map(lambda x: TO_RUS[x], data.columns))

fig, ax = plt.subplots(figsize=(7, 7))
sns.violinplot(data=data, x=TO_RUS['system'], y=TO_RUS['thickness, nm'],
               hue=TO_RUS['CHL amount, %'],
               cut=0, palette='RdYlGn_r', inner='quartile',
               ax=ax)
sns.move_legend(ax, loc='upper center',
                bbox_to_anchor=(0.5, -0.2), ncol=6)

fig.patch.set_facecolor('white')
fig.savefig(
    PATH / 'notebooks' / 'dopc_dops' / 'integral' /
    f'thickness_rus.png',
    bbox_inches='tight', dpi=300)

# %%
# arperlip
df = pd.read_csv(PATH / 'notebooks' / 'integral_parameters' / 'arperlip_150-200-1000.csv')

data = df[df['system'].isin(['dopc', 'dops'])].copy()
data['system'] = data['system'].map(TO_RUS)
data.columns = list(map(lambda x: TO_RUS[x], data.columns))

fig, ax = plt.subplots(figsize=(7, 7))
sns.violinplot(data=data, x=TO_RUS['system'], y=TO_RUS['area per lipid, nm²'],
               hue=TO_RUS['CHL amount, %'],
               cut=0, palette='RdYlGn_r', inner='quartile',
               ax=ax)
sns.move_legend(ax, loc='upper center',
                bbox_to_anchor=(0.5, -0.2), ncol=6)

fig.patch.set_facecolor('white')
fig.savefig(
    PATH / 'notebooks' / 'dopc_dops' / 'integral' /
    f'area_per_lipid_rus.png',
    bbox_inches='tight', dpi=300)


# %%
# chl_p distance

# %%
df = pd.read_csv(PATH / 'notebooks' / 'integral_parameters' /
                 'chols_o_phosphates_distances_150-200-1000.csv')

data = df[df['system'].isin(['dopc', 'dops'])].copy()
data['system'] = data['system'].map(TO_RUS)
data.columns = list(map(lambda x: TO_RUS[x], data.columns))

fig, ax = plt.subplots(figsize=(7, 7))
sns.violinplot(data=data, x=TO_RUS['system'], y=TO_RUS['distance, nm'],
               hue=TO_RUS['CHL amount, %'],
               cut=0, palette='RdYlGn_r', inner='quartile',
               ax=ax)
sns.move_legend(ax, loc='upper center',
                bbox_to_anchor=(0.5, -0.2), ncol=6)

fig.patch.set_facecolor('white')
fig.savefig(
    PATH / 'notebooks' / 'dopc_dops' / 'integral' /
    f'chl_o_p_distance_rus.png',
    bbox_inches='tight', dpi=300)

# %%
# scd


def dfol(df: pd.DataFrame, system: str, chain: str, chl_amount: int) -> pd.DataFrame:
    '''
    extract data for one line in plot from df
    '''
    return df[(df['system'] == system)
              & (df['chain'] == chain)
              & (df['CHL amount, %'] == chl_amount)]


rus = True
df = pd.read_csv(PATH / 'notebooks' /
                 'integral_parameters' / f'scd_150-200-1000.csv')
df['atom_n'] = df['atom'].apply(lambda x: int(x[2:]))
# df.sort_values(['system', 'CHL amount, %', 'chain', 'atom_n'], inplace=True)
scd_ms = df.drop(columns=['timepoint', 'atom']).groupby(
    ['system', 'CHL amount, %', 'chain', 'atom_n'],
    as_index=False).agg(['mean', 'std'])
scd_ms = scd_ms.reset_index(level=1).reset_index(
    level=1).reset_index(level=1).reset_index()
systs = ['dopc', 'dops']
chls = 'ХС' if rus else 'CHL'
scd_ms_part = scd_ms[scd_ms['system'].str.fullmatch('|'.join(systs))]
fig, axs = plt.subplots(1, 2, figsize=(
    15, 7), sharex=True, sharey=True)
for s,  ax in zip(systs, axs):
    for c, chl in enumerate((0, 10, 30, 50)):
        for sn, ls in zip(('sn-1', 'sn-2'),
                          ('-', '--')):
            ax.errorbar(x=dfol(scd_ms_part, s, sn, chl)['atom_n'].astype(int),
                        y=dfol(scd_ms_part, s, sn, chl)['scd']['mean'],
                        yerr=dfol(scd_ms_part, s, sn, chl)[
                'scd']['std'],
                ls=ls, color=sns.color_palette('cubehelix')[c],
                elinewidth=1, label=f'{chl} % {chls}, {sn}'
            )
    s = TO_RUS[s] if rus else s
    ax.set_title(s)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if rus:
        ax.set_xlabel('Номер атома углерода')
    else:
        ax.set_xlabel('C atom number')
axs[0].set_ylabel('Scd')
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='upper center', bbox_to_anchor=(0.5, 0), ncol=4)


fig.patch.set_facecolor('white')
fig.savefig(
    PATH / 'notebooks' / 'dopc_dops' / 'integral' /
    f'scd_rus.png',
    bbox_inches='tight', dpi=300)


# %%
df = pd.read_csv(PATH / 'notebooks' / 'integral_parameters' / 'chl_tilt_to_plot.csv')
df = df[['system', 'α, °', 'CHL amount, %']]
data = df[df['system'].isin(['dopc', 'dops'])].copy()
data['system'] = data['system'].map(TO_RUS)
data.columns = list(map(lambda x: TO_RUS[x], data.columns))

fig, ax = plt.subplots(figsize=(7, 7))
sns.violinplot(data=data, x=TO_RUS['system'], y='α, °',
               hue=TO_RUS['CHL amount, %'],
               cut=0, palette='RdYlGn_r', inner='quartile',
               ax=ax)
sns.move_legend(ax, loc='upper center',
                bbox_to_anchor=(0.5, -0.2), ncol=6)

fig.patch.set_facecolor('white')
fig.savefig(
    PATH / 'notebooks' / 'dopc_dops' / 'integral' /
    f'chl_tilt_rus.png',
    bbox_inches='tight', dpi=300)

# %%
# angle components


def plot_comps(syst: str, ax, n_comps: int = 4):
    lines = opener(PATH / 'notebooks' / 'chol_tilt' /
                   f'{syst}_100-200-100_tilt.xvg')
    a = list(map(float, flatten([i.split()[1:] for i in lines])))
    a = np.array([i if i <= 90 else i - 180 for i in a])

    def func(x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            ctr = params[i]
            amp = params[i + 1]
            wid = params[i + 2]
            y = y + amp * np.exp(-((x - ctr) / wid)**2)
        return y

    x = np.arange(np.min(a), np.max(a),
                  (np.max(a) - np.min(a)) / 500)
    my_kde = sns.kdeplot(a,
                         ax=ax,
                         # fill=False,
                         lw=0,
                         color='black',
                         # element='poly',
                         # stat='density',
                         bw_adjust=0.4,
                         cut=0
                         )
    line = my_kde.lines[0]
    x, y = line.get_data()

    if n_comps == 4:
        guess = [-20, 0.005, 10, -15, 0.02, 7, 10, 0.02, 7, 20, 0.005, 10]
    elif n_comps == 2:
        guess = [-20, 0.005, 10, 20, 0.005, 10]

    popt, _, _, _, _ = curve_fit(func, x, y, p0=guess, full_output=True)
    df = pd.DataFrame(popt.reshape(int(len(guess) / 3), 3),
                      columns=['ctr', 'amp', 'wid'])
    df['area'] = df.apply(lambda row: integrate.quad(func, np.min(
        x), np.max(x), args=(row['ctr'], row['amp'], row['wid']))[0], axis=1)
    fit = func(x, *popt)

    sns.histplot(a,
                 # fill=False,
                 ax=ax,
                 lw=0,
                 element='poly',
                 stat='density',
                 alpha=0.2,
                 edgecolor='black'
                 )

    ax.plot(x, fit, '-k', lw=2)
    if n_comps == 4:
        ax.plot(x, func(x, *popt[:3]), '--')
        ax.plot(x, func(x, *popt[3:6]), '--')
        ax.plot(x, func(x, *popt[6:9]), '--')
        ax.plot(x, func(x, *popt[9:]), '--')
    elif n_comps == 2:
        ax.plot(x, func(x, *popt[:3]), '--')
        ax.plot(x, func(x, *popt[3:6]), '--')
    ax.set_xlabel('ɑ, °')
    ax.set_ylabel('Плотность вероятности')
    syst_name, chl_amount = syst.split('_chol', 1)
    if n_comps == 4:
        ax.set_title(f'{TO_RUS[syst_name]}, {chl_amount}% ХС, 2 компоненты')
    elif n_comps == 2:
        ax.set_title(f'{TO_RUS[syst_name]}, {chl_amount}% ХС, 1 компонента')
    ks_stat, p_val = stats.kstest(y, func(x, *popt))
    ax.text(0.62, 0.88,
            f'KS stat={round(ks_stat,3)}\np-value={"%.1E" % p_val}',
            size=15, transform=ax.transAxes)


# %%
for syst in ['dopc', 'dops']:
    fig, axs = plt.subplots(3, 2, figsize=(14, 21), sharex=True, sharey=True)
    axs = axs.flatten()

    for c, i in enumerate([10, 30, 50]):
        plot_comps(f'{syst}_chol{i}', axs[c * 2], 2)
        plot_comps(f'{syst}_chol{i}', axs[c * 2 + 1], 4)

    fig.patch.set_facecolor('white')
    fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'angles' /
                f'{syst}_comps_comparison.png',
                bbox_inches='tight', dpi=300)


# %%
# density profiles

systs = ['dopc']#, 'dops']
detailed = True

systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                   for i in flatten(EXPERIMENTS.values())])
systems = list(dict.fromkeys(systems))
trj_slices = [TrajectorySlice(System(
    PATH, s), 150, 200, 1000) for s in systems]

fig, axs = plt.subplots(2, 2, figsize=(15, 12.5),
                        # gridspec_kw={'wspace': 0.3},
                        sharex=True, sharey=True)
axs=axs.flatten()

# axs = axs.reshape(-1, order='F')
for c, trj in enumerate(
    [trj for trj in trj_slices
     if trj.system.name
     in flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                for i in systs])]):
    plot_density_profile(axs[c], trj, lw=5)
    try:
        name, chl = trj.system.name.split('_chol', 1)
    except ValueError:
        name, chl = trj.system.name, '0'
    axs[c].set_title(f'{TO_RUS[name]}, {chl}% ХС')
    if detailed:
        axs[c].set_xlim(0, 3)

    if c not in [2, 3]:
        axs[c].set_xlabel('')
    else:
        axs[c].set_xlabel('Z, нм')

    if c not in [0,2]:
        axs[c].set_ylabel('')
    else:
        axs[c].set_ylabel('Плотность, кг/м³')

handles, labels = axs[3].get_legend_handles_labels()
labels = map(lambda x: TO_RUS[x], labels)
fig.legend(handles, labels, loc='upper center',
           bbox_to_anchor=(0.5, 0.05), ncol=6)

fig.patch.set_facecolor('white')
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'dp_ib_space' /
            f'dopc_dp_detailed.png',
            bbox_inches='tight', dpi=300)


# %%
# intrabilayer space
def fv_figure_single(ax, trj, label=None, color='C0'):
    df = pd.read_csv(
        PATH / 'notebooks' / 'intrabilayer_space' /
        f'{trj.system.name}_{trj.b}-{trj.e}-{trj.dt}_fvoz.csv',
        header=None, skiprows=1, names=['Z, Å', 'Free Volume, Å³'])

    df['Z, Å'] = df['Z, Å'] + \
        (df['Z, Å'].max() - df['Z, Å'].min()) / 2 - df['Z, Å'].max()

    ax.plot(df['Z, Å'], df['Free Volume, Å³'], color=color, label=label)


palette = sns.color_palette('crest_r', 4)
chl_amounts = [0, 10, 30, 50]
fig, axs = plt.subplots(1, 2, figsize=(15, 7),
                        sharex=True, sharey=True)
axs = axs.reshape(-1, order='F').flatten()
c = 0

systs = ['dopc', 'dops']

trjs = [trj for trj in trj_slices
        if trj.system.name
        in flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                    for i in systs])]
for chunk in chunker(trjs, 4):
    for it, trj in enumerate(chunk):
        fv_figure_single(axs[c], trj, chl_amounts[it], palette[it])
    axs[c].set_title(TO_RUS[chunk[0].system.name])
    if c != 1:
        axs[c].set_ylabel('Свободный объем, Å³')
    axs[c].set_xlabel('Z, Å')
    c += 1

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Концентрация ХС, %', loc='upper center',
           bbox_to_anchor=(0.5, 0), ncol=6)

fig.patch.set_facecolor('white')
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'dp_ib_space' /
            f'ib_space.png',
            bbox_inches='tight', dpi=300)


# %%
# angle_z
angle_mhp_z = pd.read_csv(
    PATH / 'notebooks' / 'integral_parameters' /
    'angle_mhp_z_200.0-201.0-1.csv')

# %%
angle_mhp_z['distance to bilayer center, Å'] = pd.cut(
    angle_mhp_z['distance to bilayer center'],
    bins=[0, 3, 6, 9, 12, 15, 100],
    labels=['<= 3', '3-6', '6-9', '9-12', '12-15', '> 15'])
angle_mhp_z['α, °'] = angle_mhp_z['α, °'].abs()
angle_mhp_z.rename(columns={
    'distance to bilayer center, Å': TO_RUS['distance to bilayer center, Å']},
    inplace=True)


# %%
systs = ['dopc', 'dops']

for syst in systs:
    fig, axs = plt.subplots(1, 3, figsize=(20, 7),
                            sharex=True, sharey=True)
    for c, chl_amount in enumerate([10, 30, 50]):
        data = angle_mhp_z[
            (angle_mhp_z['system'] == syst) &
            (angle_mhp_z['CHL amount, %'] == chl_amount)]
        sns.kdeplot(data=data, x='α, °',
                    hue=TO_RUS['distance to bilayer center, Å'],
                    ax=axs.flatten()[c], palette='crest_r', fill=True,
                    common_norm=False, legend=c == 1)

        axs[c].set_title(f'{TO_RUS[syst]}, {chl_amount}% ХС')
        data = data[data['surface'] == 1]
    axs[0].set_ylabel('Плотность вероятности')

    sns.move_legend(axs.flatten()[1], loc='upper center',
                    bbox_to_anchor=(0.5, -0.2), ncol=6)
    fig.patch.set_facecolor('white')
    fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'angles' /
                f'{syst}_angle_z.png',
                bbox_inches='tight', dpi=300)


# %%
angle_mhp_z = pd.read_csv(
    PATH / 'notebooks' / 'integral_parameters' /
    'angle_mhp_z_'
    f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')

angle_mhp_z['α, °'] = angle_mhp_z['α, °'].abs()
angle_mhp_z['surface'] = angle_mhp_z['surface'].map({1: 'yes', 0: 'no'})


# %%
systs = ['dopc', 'dops']

for syst in systs:
    fig, axs = plt.subplots(1, 3, figsize=(20, 7),
                            sharex=True, sharey=True)

    for c, chl_amount in enumerate([10, 30, 50]):
        data = angle_mhp_z[
            (angle_mhp_z['system'] == syst) &
            (angle_mhp_z['CHL amount, %'] == chl_amount)]
        sns.kdeplot(data=data, x='α, °',
                    hue='surface',
                    hue_order=['no', 'yes'],
                    ax=axs[c], fill=True,
                    common_norm=False, legend=c == 1)

        axs[c].set_title(f'{TO_RUS[syst]}, {chl_amount}% ХС')
        data = data[data['surface'] == 1]
    axs[0].set_ylabel('Плотность вероятности')

    sns.move_legend(axs[1], loc='upper center',
                    bbox_to_anchor=(0.5, -0.2), ncol=6)

    fig.patch.set_facecolor('white')
    fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'angles' /
                f'{syst}_angle_surface.png',
                bbox_inches='tight', dpi=300)

# %%
# contacts_per_chl

systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                   for i in flatten(EXPERIMENTS.values())])
systems = list(dict.fromkeys(systems))
trj_slices = [TrajectorySlice(System(
    PATH, s), 200.0, 201.0, 1) for s in systems]

n_chols_df = get_n_chl_df(trj_slices)
postfix, df = merge_hb_or_dc_tables(trj_slices, False)

# %%
for hue in ['surface', 'tilt component',
            'distance to bilayer center, Å']:

    table_with_probs = (df.groupby(
        ['system', 'CHL amount, %', 'timepoint', 'other_name', hue],
        as_index=False)['chl_index']
        .agg('count')
        .rename(columns={'chl_index': 'n_contacts'})
    )
    table_with_probs = table_with_probs.merge(
        n_chols_df,
        on=['system', 'CHL amount, %', 'timepoint'], how='left')

    table_with_probs['n contacts per molecule'] = (
        table_with_probs['n_contacts']
        / table_with_probs['n_chl'])

    order = (['<=3', '3-6', '6-9', '9-12', '12-15', '>15']
             if hue == 'distance to bilayer center, Å'
             else ['vertical', 'horizontal'] if hue == 'tilt component'
             else None)
    palette = 'crest_r' if hue == 'distance to bilayer center, Å' else 'muted'

    systs = ['dopc', 'dops']

    fig, axs = plt.subplots(2, 3, figsize=(20, 15),
                            sharex=True, sharey=True)
    axs = axs.flatten()

    c = 0
    for syst in systs:
        for chl_amount in [10, 30, 50]:
            data = table_with_probs[
                (table_with_probs['system'] == syst) &
                (table_with_probs['CHL amount, %'] == chl_amount)]
            sns.barplot(data=data,
                        x='other_name', y='n contacts per molecule',
                        order=['CHL', 'PL', 'SOL'],
                        hue=hue, hue_order=order, ax=axs[c],
                        edgecolor='k', palette=palette, errorbar='sd'
                        )

            labels = [TO_RUS[item.get_text()] for item in axs[c].get_xticklabels()]
            axs[c].set_xticklabels(labels)

            if c != 4:
                axs[c].legend([], [], frameon=False)
            if c not in [0, 3]:
                axs[c].set_ylabel('')
            else:
                axs[c].set_ylabel(TO_RUS['n contacts per CHL molecule'])

            if c not in [3, 4, 5]:
                axs[c].set_xlabel('')
            else:
                axs[c].set_xlabel('Контактирующая молекула')
            axs[c].xaxis.set_tick_params(which='both', labelbottom=True)

            axs[c].set_title(f'{TO_RUS[syst]}, {chl_amount}% ХС')
            c += 1

    handles, labels = axs[4].get_legend_handles_labels()
    axs[4].legend(
        handles,
        (map(lambda x: TO_RUS[x], labels)
         if hue != 'distance to bilayer center, Å'
         else labels),
        title=TO_RUS[hue],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2), ncol=6)

    fig.patch.set_facecolor('white')
    fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'contacts' / 'chol' /
                f'{postfix}_{hue}.png',
                bbox_inches='tight', dpi=300)


# %%
# hb with PL groups
systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                   for i in flatten(EXPERIMENTS.values())])
systems = list(dict.fromkeys(systems))
trj_slices = [TrajectorySlice(System(
    PATH, s), 200.0, 201.0, 1) for s in systems]

n_chols_df = get_n_chl_df(trj_slices)
postfix, df = merge_hb_or_dc_tables(trj_slices, True)

# %%
df = df[df['other_name'] == 'PL']


# %%
for hue in ['surface', 'tilt component',
            'distance to bilayer center, Å']:
    table_with_probs = (df.groupby(
        ['system', 'CHL amount, %', 'timepoint', 'aan', hue],
        as_index=False)['chl_index']
        .agg('count')
        .rename(columns={'chl_index': 'n_contacts'})
    )

    # aan to PL group
    determine_pl_group = {'Ser': ['OH9', 'OH10', 'NH7'],
                          'PO4': ['OH2', 'OH3', 'OH4', 'OG3'],
                          "ROR'": ['OG1', 'OG2'],
                          "RC(O)R'": ['OA1', 'OB1']}
    # need to reverse this dict for vectorized
    # assigning of groups later
    reverse_lookup_dict = {}
    for key, values in determine_pl_group.items():
        for value in values:
            reverse_lookup_dict[value] = key

    table_with_probs['PL group'] = table_with_probs['aan'].map(
        reverse_lookup_dict)

    table_with_probs = table_with_probs.merge(
        n_chols_df,
        on=['system', 'CHL amount, %', 'timepoint'], how='left')

    table_with_probs['n contacts per molecule'] = (
        table_with_probs['n_contacts']
        / table_with_probs['n_chl'])

    hue_order = (['<=3', '3-6', '6-9', '9-12', '12-15', '>15']
                 if hue == 'distance to bilayer center, Å'
                 else ['vertical', 'horizontal'] if hue == 'tilt component'
                 else None)
    palette = 'crest_r' if hue == 'distance to bilayer center, Å' else 'muted'

    systs = ['dopc', 'dops']
    fig, axs = plt.subplots(2, 3, figsize=(20, 15),
                            sharey=True)
    axs = axs.flatten()

    c = 0
    for syst in systs:
        order = (['PO4', "RC(O)R'", "ROR'", 'Ser'] if 'dops' in syst
                 else ['PO4', "RC(O)R'", "ROR'"])
        for chl_amount in [10, 30, 50]:
            data = table_with_probs[
                (table_with_probs['system'] == syst) &
                (table_with_probs['CHL amount, %'] == chl_amount)]
            sns.barplot(data=data,
                        x='PL group', y='n contacts per molecule',
                        order=order, hue=hue, hue_order=hue_order,
                        ax=axs[c],
                        edgecolor='k', palette=palette, errorbar='sd'
                        )
            if c != 4:
                axs[c].legend([], [], frameon=False)
            if c not in [0, 3]:
                axs[c].set_ylabel('')
            else:
                axs[c].set_ylabel(TO_RUS['n contacts per CHL molecule'])

            if c not in [3, 4, 5]:
                axs[c].set_xlabel('')
            else:
                axs[c].set_xlabel('группа ФЛ')
            axs[c].xaxis.set_tick_params(which='both', labelbottom=True)

            axs[c].set_title(f'{TO_RUS[syst]}, {chl_amount}% ХС')
            c += 1

    handles, labels = axs[4].get_legend_handles_labels()
    axs[4].legend(
        handles,
        (map(lambda x: TO_RUS[x], labels)
         if hue != 'distance to bilayer center, Å'
         else labels),
        title=TO_RUS[hue],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2), ncol=6)

    fig.patch.set_facecolor('white')
    fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'contacts' / 'chol' /
                f'chol_pl_groups_{postfix}_{hue}.png',
                bbox_inches='tight', dpi=300)


# %%
# hbonds lifetimes
interactions = ['CHL-CHL', 'CHL-PL', 'CHL-SOL', 'PL-SOL', 'PL-PL']
for interaction in interactions:
    df = pd.read_csv(PATH / 'notebooks' / 'contacts' /
                     f'{interaction}_hb_lt_distr_'
                     f'200.0-201.0-1.csv')

    if 'CHL' in interaction:
        df = df[df['CHL amount, %'] != 0]
    df.rename(columns={interaction: 'lifetime, ps'}, inplace=True)
    df = df[df['lifetime, ps'] != 0]

    systs = ['dopc', 'dops']

    fig, axs = plt.subplots(1, 2, figsize=(15, 7),
                            sharey=True, sharex=True)
    for syst, ax in zip(systs, axs):
        data = df[df['system'] == syst]
        try:
            sns.histplot(data=data, x='lifetime, ps', alpha=.2,
                         hue='CHL amount, %',
                         palette='RdYlGn_r', stat='density', ax=ax,
                         binwidth=.15, log_scale=True,
                         legend=False, common_norm=False)
        except ValueError as e:
            print(syst, str(e))
        try:
            sns.kdeplot(data=data, x='lifetime, ps', lw=3,
                        hue='CHL amount, %',
                        palette='RdYlGn_r', ax=ax, common_norm=False,
                        log_scale=True, legend=ax == axs[-1], cut=0)
        except np.linalg.LinAlgError as e:
            print(syst, str(e))
        ax.set_title(TO_RUS[syst])
        ax.set_xlabel('Время жизни, пс')
    axs[0].set_ylabel('Плотность вероятности')

    legend = axs[-1].get_legend()
    handles = legend.legend_handles
    labels = data['CHL amount, %'].unique()
    legend.remove()
    fig.legend(handles, labels, title='Концентрация ХС, %', loc='upper center',
               bbox_to_anchor=(0.5, 0), ncol=6)
    fig.patch.set_facecolor('white')
    fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'contacts' /
                'hb_lt_distr' /
                f'hb_lt_{interaction}.png',
                bbox_inches='tight', dpi=300)


# %%
# hbonds chl no pl sol
df = pd.read_csv(PATH / 'notebooks' / 'contacts' /
                 'hb_chl_angle_z_mhp_'
                 '200.0-201.0-1.csv',
                 usecols=['system', 'CHL amount, %', 'timepoint',
                          'chl_index', 'other_name'],
                 low_memory=False)
df = df[df['CHL amount, %'] != 0]
n_chl_df = get_n_chl_df(trj_slices)

new_df = df.groupby(['system', 'CHL amount, %', 'timepoint', 'other_name'],
                    as_index=False)['chl_index'].nunique().rename(
    columns={'chl_index': 'n_chl_with_hbonds'}
)
probs_df = new_df.merge(
    n_chl_df,
    on=['system', 'CHL amount, %', 'timepoint'], how='left')
probs_df['% of CHL'] = (
    probs_df['n_chl_with_hbonds'] / probs_df['n_chl'] * 100)
new_df2 = df.groupby(['system', 'CHL amount, %', 'timepoint'],
                     as_index=False)['chl_index'].nunique().rename(
    columns={'chl_index': 'n_chl_with_hbonds'})
no_hb_df = new_df2.merge(
    n_chl_df,
    on=['system', 'CHL amount, %', 'timepoint'], how='left')
no_hb_df['% of CHL'] = (
    1 - no_hb_df['n_chl_with_hbonds'] / no_hb_df['n_chl']) * 100
no_hb_df['other_name'] = 'No hbonds'
final_df = pd.concat([probs_df, no_hb_df],
                     ignore_index=True).sort_values(
    ['system', 'CHL amount, %', 'timepoint', 'other_name'],
    ignore_index=True)


# %%
systs = ['dopc', 'dops']

fig, axs = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
for ax, syst in zip(axs, systs):
    data = final_df[final_df['system'] == syst]
    sns.barplot(data=data,
                x='other_name', y='% of CHL',
                hue='CHL amount, %',
                order=['No hbonds', 'PL', 'SOL'], ax=ax,
                edgecolor='k', palette='RdYlGn_r', errorbar='sd'
                )
    ax.legend([], [], frameon=False)
    ax.set_xlabel('Контактирующая молекула')
    ax.set_ylim(0)
    ax.set_title(TO_RUS[syst])
    labels = [TO_RUS[item.get_text()] for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)
    if ax != axs[0]:
        ax.set_ylabel('')
    else:
        ax.set_ylabel('% молекул ХС')

handles, labels = axs[1].get_legend_handles_labels()
fig.legend(handles, labels,
           title='Концентрация ХС, %',
           loc='upper center', bbox_to_anchor=(0.5, 0), ncol=4)


fig.patch.set_facecolor('white')
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'contacts' /
            'chol' /
            f'perc_of_chl_with_hb.png',
            bbox_inches='tight', dpi=300)


# %%
# hbonds with sol
chl = True
fname = (PATH / 'notebooks' / 'contacts' /
         f'chl_sol_over_lip_sol_ratios_'
         f'{trj_slices[0].b}-{trj_slices[0].e}-'
         f'{trj_slices[0].dt}.csv' if chl else
         PATH / 'notebooks' / 'contacts' /
         f'pl_sol_over_lip_sol_ratios_'
         f'{trj_slices[0].b}-{trj_slices[0].e}-'
         f'{trj_slices[0].dt}.csv')

mol = 'CHL' if chl else 'PL'

df = pd.read_csv(fname)
df.set_index('index', inplace=True)

fig, ax = plt.subplots(figsize=(7, 7))

width = 0.25
palette = sns.color_palette('RdYlGn_r', 3)

systems = ['dopc', 'dops']

x = np.arange(len(systems))
positions = (x - width, x, x + width)
for c, chl_amount in enumerate([10, 30, 50]):
    systs = (
        systems if chl_amount == 0
        else [i + f'_chol{chl_amount}' for i in systems])
    label = chl_amount
    ax.bar(positions[c], df.loc[systs, f'{mol}-SOL / LIP-SOL, %'],
           yerr=df.loc[systs, 'std'], width=width, ec='k',
           color=palette[c], capsize=5,
           error_kw={'elinewidth': 2}, label=label)

ax.xaxis.set_ticks(x)
ax.set_xticklabels(map(lambda x: TO_RUS[x], systems))
ax.set_ylabel(f'{TO_RUS[mol]}-Вода / Липиды-Вода, %')
ax.set_ylim(0)
if ax == axs[0]:
    ax.set_ylabel(f'{mol}-SOL / LIP-SOL, %')
fig.legend(loc='upper center', title='Концентрация ХС, %',
           bbox_to_anchor=(0.5, 0), ncol=3)
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'contacts' /
            'sol' /
            f'hb_lt_ratios_{mol}.png',
            bbox_inches='tight', dpi=300)


# %%
# sol inside bilayer hbonds

sol_indexes = get_sol_indexes_df(trj_slices)
sol_indexes.rename(columns={'timestep': 'timepoint',
                            'SOL index': 'dmi'}, inplace=True)
lip_sol_rchist_full = pd.read_csv(PATH / 'notebooks' / 'contacts' /
                                  'lip_SOL_hb_hb_full_'
                                  f'{trj_slices[0].b}-{trj_slices[0].e}'
                                  f'-{trj_slices[0].dt}_'
                                  'rchist_full.csv')

# %%
merged_df = sol_indexes.merge(
    lip_sol_rchist_full,
    on=['index', 'timepoint', 'dmi'],
    how='inner')

merged_df['other_name'] = merged_df['amn']
merged_df.loc[merged_df['amn'].str.endswith(
    ('PC', 'PS')), 'other_name'] = 'PL'

# %%
df_prob = merged_df.groupby(
    ['system', 'CHL amount, %', 'timepoint', 'other_name'],
    as_index=False)['dmi'].count().rename(
    columns={'dmi': 'n_hbonds'}
)
df_sum = merged_df.groupby(['system', 'CHL amount, %', 'timepoint'],
                           as_index=False)['dmi'].count().rename(
    columns={'dmi': 'n_hbonds_sum'})
df_prob = df_prob.merge(df_sum, on=['system', 'CHL amount, %', 'timepoint'],
                        how='left')
df_prob['% of SOL'] = (df_prob['n_hbonds']
                       / df_prob['n_hbonds_sum'] * 100)

systs = ['dopc', 'dops']
fig, axs = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
for ax, syst in zip(axs, systs):
    data = df_prob[df_prob['system'] == syst]
    sns.barplot(data=data,
                x='other_name', y='% of SOL',
                hue='CHL amount, %',
                order=['PL', 'CHL'], ax=ax,
                edgecolor='k', palette='RdYlGn_r', errorbar='sd'
                )

    ax.legend([], [], frameon=False)
    labels = [TO_RUS[item.get_text()] for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)
    ax.set_xlabel('Контактирующая молекула')
    ax.set_ylim(0)
    ax.set_title(TO_RUS[syst])
    if ax != axs[0]:
        ax.set_ylabel('')
    else:
        ax.set_ylabel('% молекул воды')

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Концентрация ХС, %',
           loc='upper center', bbox_to_anchor=(0.5, 0), ncol=4)
fig.patch.set_facecolor('white')
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'contacts' /
            'sol' /
            f'intrabilayer_sol_hb_ratio.png',
            bbox_inches='tight', dpi=300)


# %%
# contacta and hbonds per PL

lip_rchist_full = (PATH / 'notebooks' / 'contacts' /
                   'lip_hb_hb_chl_angle_z_mhp_'
                   f'{trj_slices[0].b}-{trj_slices[0].e}'
                   f'-{trj_slices[0].dt}_'
                   'rchist_full.csv')

if not lip_rchist_full.is_file():
    raise ValueError(
        '%s does not exist. use z_slices_angles_contacts script.')

df_chl = pd.read_csv(lip_rchist_full)
n_pl_df = get_n_pl_df(trj_slices)

table_with_probs_chl = (df_chl[(df_chl['other_name'] != 'CHL') &
                               (df_chl['other_name'] != 'SOL')].groupby(
    ['system', 'CHL amount, %', 'timepoint'],
    as_index=False)['other_name']
    .agg('count')
    .rename(columns={'other_name': 'n_contacts'})
)

table_with_probs_chl = table_with_probs_chl.merge(
    n_pl_df,
    on=['system', 'CHL amount, %', 'timepoint'], how='left')

table_with_probs_chl['n contacts per molecule'] = (
    table_with_probs_chl['n_contacts']
    / table_with_probs_chl['n_pl'])

systs = ['dopc', 'dops']

fig, ax = plt.subplots(figsize=(7, 7))
data = table_with_probs_chl[
    table_with_probs_chl['system'].isin(systs)]
sns.barplot(data=data,
            x='system', y='n contacts per molecule',
            hue='CHL amount, %', ax=ax,
            edgecolor='k', palette='RdYlGn_r', errorbar='sd'
            )

ax.set_ylabel('Число контактов на молекулу ФЛ')
ax.set_xlabel('Система')
ax.set_xticklabels(map(lambda x: TO_RUS[x], systems))
ax.legend(loc='upper center', title='Концентрация ХС, %',
          bbox_to_anchor=(0.5, -0.2), ncol=6)
fig.patch.set_facecolor('white')
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'contacts' /
            'pl' / 'hb_n_chl_per_pl.png',
            bbox_inches='tight', dpi=300)


# %%
lip_rchist_full = (PATH / 'notebooks' / 'contacts' /
                   'lip_dc_dc_chl_angle_z_mhp_'
                   '200.0-201.0'
                   '-1_'
                   'rchist_full.csv')

if not lip_rchist_full.is_file():
    raise ValueError(
        '%s does not exist. use z_slices_angles_contacts script.')

df_chl = pd.read_csv(lip_rchist_full)

n_pl_df = get_n_pl_df(trj_slices)


table_with_probs_chl = (df_chl[(df_chl['other_name'] != 'CHL') &
                               (df_chl['other_name'] != 'SOL')].groupby(
    ['system', 'CHL amount, %', 'timepoint'],
    as_index=False)['other_name']
    .agg('count')
    .rename(columns={'other_name': 'n_contacts'})
)

table_with_probs_chl = table_with_probs_chl.merge(
    n_pl_df,
    on=['system', 'CHL amount, %', 'timepoint'], how='left')

table_with_probs_chl['n contacts per molecule'] = (
    table_with_probs_chl['n_contacts']
    / table_with_probs_chl['n_pl'])

systs = ['dopc', 'dops']

# %%
fig, ax = plt.subplots(figsize=(7, 7))
data = table_with_probs_chl[
    table_with_probs_chl['system'].isin(systs)]
sns.barplot(data=data,
            x='system', y='n contacts per molecule',
            hue='CHL amount, %', ax=ax,
            edgecolor='k', palette='RdYlGn_r', errorbar='sd'
            )

ax.set_ylabel('Число контактов на молекулу ФЛ')
ax.set_xlabel('Система')
ax.set_xticklabels(map(lambda x: TO_RUS[x], systems))
ax.legend(loc='upper center', title='Концентрация ХС, %',
          bbox_to_anchor=(0.5, -0.2), ncol=6)
fig.patch.set_facecolor('white')
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'contacts' /
            'pl' / 'dc_n_chl_per_pl.png',
            bbox_inches='tight', dpi=300)


# %%
lip_sol_rchist_full = (PATH / 'notebooks' / 'contacts' /
                       'lip_SOL_hb_hb_full_'
                       f'{trj_slices[0].b}-{trj_slices[0].e}'
                       f'-{trj_slices[0].dt}_'
                       'rchist_full.csv')

df_sol = pd.read_csv(lip_sol_rchist_full)
n_pl_df = get_n_pl_df(trj_slices)

table_with_probs_sol = (df_sol.groupby(
    ['system', 'CHL amount, %', 'timepoint'],
    as_index=False)['dmn']
    .agg('count')
    .rename(columns={'dmn': 'n_contacts'})
)
table_with_probs_sol = table_with_probs_sol.merge(
    n_pl_df,
    on=['system', 'CHL amount, %', 'timepoint'], how='left')

table_with_probs_sol['n contacts per molecule'] = (
    table_with_probs_sol['n_contacts']
    / table_with_probs_sol['n_pl'])

systs = ['dopc', 'dops']
# %%
fig, ax = plt.subplots(figsize=(7, 7))
data = table_with_probs_sol[
    table_with_probs_sol['system'].isin(systs)]
sns.barplot(data=data,
            x='system', y='n contacts per molecule',
            hue='CHL amount, %', ax=ax,
            edgecolor='k', palette='RdYlGn_r', errorbar='sd'
            )

ax.set_ylabel('Число контактов на молекулу ФЛ')
ax.set_xlabel('Система')
ax.set_xticklabels(map(lambda x: TO_RUS[x], systems))
ax.legend(loc='upper center', title='Концентрация ХС, %',
          bbox_to_anchor=(0.5, -0.2), ncol=6)
fig.patch.set_facecolor('white')
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'contacts' /
            'pl' / 'hb_n_sol_per_pl.png',
            bbox_inches='tight', dpi=300)


# %%
lip_rchist_full = (PATH / 'notebooks' / 'contacts' /
                   'lip_dc_dc_full_'
                   f'{trj_slices[0].b}-{trj_slices[0].e}'
                   f'-{trj_slices[0].dt}_'
                   'rchist_full.csv')

df_pl = pd.read_csv(lip_rchist_full)
n_pl_df = get_n_pl_df(trj_slices)

table_with_probs_pl = (df_pl[(df_pl['dmn'] != 'CHL') &
                             (df_pl['dmn'] != 'SOL') &
                             (df_pl['amn'] != 'CHL') &
                             (df_pl['amn'] != 'SOL')].groupby(
    ['system', 'CHL amount, %', 'timepoint'],
    as_index=False)['dmn']
    .agg('count')
    .rename(columns={'dmn': 'n_contacts'})
)

table_with_probs_pl = table_with_probs_pl.merge(
    n_pl_df,
    on=['system', 'CHL amount, %', 'timepoint'], how='left')

table_with_probs_pl['n contacts per molecule'] = (
    table_with_probs_pl['n_contacts'] * 2
    / table_with_probs_pl['n_pl'])

# %%
fig, ax = plt.subplots(figsize=(7, 7))
data = table_with_probs_pl[
    table_with_probs_sol['system'].isin(systs)]
data = pd.concat([pd.DataFrame({'system': ['dopc'],
                                'n contacts per molecule': [0],
                                'CHL amount, %': [0]}),
                  data])

sns.barplot(data=data,
            x='system', y='n contacts per molecule',
            hue='CHL amount, %', ax=ax,
            edgecolor='k', palette='RdYlGn_r', errorbar='sd'
            )

ax.set_ylabel('Число контактов на молекулу ФЛ')
ax.set_xlabel('Система')
ax.set_xticklabels(map(lambda x: TO_RUS[x], systems))
ax.legend(loc='upper center', title='Концентрация ХС, %',
          bbox_to_anchor=(0.5, -0.2), ncol=6)
fig.patch.set_facecolor('white')
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'contacts' /
            'pl' / 'hb_n_pl_per_pl.png',
            bbox_inches='tight', dpi=300)


# %%
# dc lt ratios:
df = pd.read_csv(PATH / 'notebooks' / 'contacts' /
                 f'chl_chl_over_lip_lip_ratios_'
                 f'{trj_slices[0].b}-{trj_slices[0].e}-'
                 f'{trj_slices[0].dt}.csv')
df.set_index('index', inplace=True)

fig, ax = plt.subplots(figsize=(7, 7))

width = 0.25
palette = sns.color_palette('RdYlGn_r', 3)

systems = ['dopc', 'dops']

x = np.arange(len(systems))
positions = (x - width, x, x + width)
for c, chl_amount in enumerate([10, 30, 50]):
    systs = [i + f'_chol{chl_amount}' for i in systems]
    label = chl_amount
    ax.bar(positions[c], df.loc[systs, 'CHL-CHL / LIP-LIP, %'],
           yerr=df.loc[systs, 'std'], width=width, ec='k',
           color=palette[c], capsize=5,
           error_kw={'elinewidth': 2}, label=label)
ax.xaxis.set_ticks(x)
ax.set_xticklabels(map(lambda x: TO_RUS[x], systems))
ax.set_ylim(0)
ax.set_ylabel('ХС-ХС / Липид-Липид, %')
fig.legend(loc='upper center', title='Концентрация ХС, %',
           bbox_to_anchor=(0.5, 0), ncol=3)
fig.patch.set_facecolor('white')
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'contacts' /
            'dc_lt_ratios' / 'chl_chl_over_lip_lip.png',
            bbox_inches='tight', dpi=300)


# %%
df = pd.read_csv(PATH / 'notebooks' / 'contacts' /
                 f'pl_pl_over_lip_lip_ratios_'
                 f'{trj_slices[0].b}-{trj_slices[0].e}-'
                 f'{trj_slices[0].dt}.csv')
df.set_index('index', inplace=True)

fig, ax = plt.subplots(figsize=(7, 7))

width = 0.25
palette = sns.color_palette('RdYlGn_r', 3)

systems = ['dopc', 'dops']

x = np.arange(len(systems))
positions = (x - width, x, x + width)
for c, chl_amount in enumerate([10, 30, 50]):
    systs = [i + f'_chol{chl_amount}' for i in systems]
    label = chl_amount
    ax.bar(positions[c], df.loc[systs, 'PL-PL / LIP-LIP, %'],
           yerr=df.loc[systs, 'std'], width=width, ec='k',
           color=palette[c], capsize=5,
           error_kw={'elinewidth': 2}, label=label)
ax.xaxis.set_ticks(x)
ax.set_xticklabels(map(lambda x: TO_RUS[x], systems))
ax.set_ylim(0)
ax.set_ylabel('ФЛ-ФЛ / Липид-Липид, %')
fig.legend(loc='upper center', title='Концентрация ХС, %',
           bbox_to_anchor=(0.5, 0), ncol=3)
fig.patch.set_facecolor('white')
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'contacts' /
            'dc_lt_ratios' / 'pl_pl_over_lip_lip.png',
            bbox_inches='tight', dpi=300)


# %%
df = pd.read_csv(PATH / 'notebooks' / 'contacts' /
                 f'chl_pl_over_lip_lip_ratios_'
                 f'{trj_slices[0].b}-{trj_slices[0].e}-'
                 f'{trj_slices[0].dt}.csv')
df.set_index('index', inplace=True)

fig, ax = plt.subplots(figsize=(7, 7))

width = 0.25
palette = sns.color_palette('RdYlGn_r', 3)

systems = ['dopc', 'dops']

x = np.arange(len(systems))
positions = (x - width, x, x + width)
for c, chl_amount in enumerate([10, 30, 50]):
    systs = [i + f'_chol{chl_amount}' for i in systems]
    label = chl_amount
    ax.bar(positions[c], df.loc[systs, 'CHL-PL / LIP-LIP, %'],
           yerr=df.loc[systs, 'std'], width=width, ec='k',
           color=palette[c], capsize=5,
           error_kw={'elinewidth': 2}, label=label)
ax.xaxis.set_ticks(x)
ax.set_xticklabels(map(lambda x: TO_RUS[x], systems))
ax.set_ylim(0)
ax.set_ylabel('ХС-ФЛ / Липид-Липид, %')
fig.legend(loc='upper center', title='Концентрация ХС, %',
           bbox_to_anchor=(0.5, 0), ncol=3)
fig.patch.set_facecolor('white')
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'contacts' /
            'dc_lt_ratios' / 'chl_pl_over_lip_lip.png',
            bbox_inches='tight', dpi=300)

# %%
# hbonds % CHL - PL group
df = pd.read_csv(PATH / 'notebooks' / 'contacts' /
                 'chl_lip_hbonds_pl_groups_'
                 f'{trj_slices[0].b}-{trj_slices[0].e}-'
                 f'{trj_slices[0].dt}.csv')

df = process_df_for_chl_lip_groups_hb(df, 1000)


# %%
fig, ax = plt.subplots(figsize=(15, 7))

sss = ['dopc', 'dops']
x = np.arange(len(sss))
width = 0.085
positions = (
    (x - 6 * width, x - 3 * width, x,             x + 3 * width),
    (x - 5 * width, x - 2 * width, x + 1 * width, x + 4 * width),
    (x - 4 * width, x - 1 * width, x + 2 * width, x + 5 * width))
alphas = (1, 0.66, 0.33)

chl_labels = ['10% ХС', '30% ХС', '50% ХС']

groups = ['PO4', "ROR'", "RC(O)R'", 'Ser']

for pos, alpha, label, chl_amount in zip(
        positions, alphas, chl_labels, df['CHL amount, %'].unique()):

    systs = [i + f'_chol{chl_amount}' for i in sss]

    if label is not None:
        labels = [i + ', ' + label for i in groups]
    else:
        labels = [None for _ in groups]

    for c, group in enumerate(groups):
        values_to_draw = df.loc[systs, :][
            df.loc[systs, :]['PL group'] == group]
        for syst in systs:
            if syst not in values_to_draw.index:
                values_to_draw.loc[syst, 'dt_fr'] = 0
                values_to_draw.loc[syst, 'dt_fr_std'] = 0
        values_to_draw = values_to_draw.loc[systs]

        ax.bar(pos[c], values_to_draw['dt_fr'], width,
               label=labels[c], color=f'C{c}', alpha=alpha)
        # single black edges independent on alpha
        ax.bar(pos[c], values_to_draw['dt_fr'], width,
               yerr=values_to_draw['dt_fr_std'], ec='k',
               fill=False, lw=2, error_kw={'elinewidth': 2})

ax.xaxis.set_ticks(x)
ax.set_xticklabels(map(lambda x: TO_RUS[x], sss))
ax.set_ylim(0)
ax.set_ylabel('P образования водородной связи, %')
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3)
fig.patch.set_facecolor('white')
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'contacts' /
            'chol' / 'chol_pl_groups_prob.png',
            bbox_inches='tight', dpi=300)

# %%
# surface CHL exposure
df = pd.read_csv(
    PATH / 'notebooks' / 'mhpmaps' /
    'chol_from_all_fractions_comps_stats_'
    f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')

df.set_index('index', inplace=True)
palette = sns.color_palette('Paired')

fig, ax = plt.subplots(figsize=(7, 7))

sss = ['dopc', 'dops']

systems = (
    [i + '_chol10' for i in sss],
    [i + '_chol30' for i in sss],
    [i + '_chol50' for i in sss],
)
width = 0.25
x = np.arange(len(sss))
positions = [x - width, x, x + width]
chl_amounts = [10, 30, 50]
for c, systs in enumerate(systems):
    data = df.loc[systs, :]
    ax.bar(positions[c], data['vert_perc_of_area'],
           yerr=data['vert_perc_of_area_std'],
           width=width, ec='k', capsize=5,
           color=palette[c * 2 + 1],
           label=f'"вертикальная", {chl_amounts[c]}% ХС')
    ax.bar(positions[c], data['hor_perc_of_area'],
           yerr=data['hor_perc_of_area_std'],
           width=width, ec='k',
           color=palette[c * 2], capsize=5,
           bottom=data['vert_perc_of_area'],
           label=f'"горизонтальная", {chl_amounts[c]}% ХС')

ax.xaxis.set_ticks(x)
ax.set_xticklabels(map(lambda x: TO_RUS[x], sss))

ax.set_ylabel('% площади')
ax.legend(title='Компонента угла α, концентрация ХС',
          loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)

fig.patch.set_facecolor('white')
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'surface' /
            'chl_exposure.png',
            bbox_inches='tight', dpi=300)


# %%
# MHP distribution
df = pd.read_csv(PATH / 'notebooks' / 'mhpmaps' / 'info_mhp_atoms_'
                 f'{trj_slices[0].b}-{trj_slices[0].e}-'
                 f'{trj_slices[0].dt*10}.csv',
                 usecols=['system', 'CHL amount, %', 'mhp', 'mol_name'])

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
for ax, syst in zip(axs, ['dopc', 'dops']):
    data = df[df['system'] == syst]
    sns.kdeplot(data=data, x='mhp', hue='CHL amount, %', ax=ax,
                legend=ax == axs[-1], palette='RdYlGn_r',
                common_norm=False, fill=True, alpha=.2)
    ax.axvline(-0.5, ls=':', c='k')
    ax.axvline(0.5, ls=':', c='k')
    ax.set_title(TO_RUS[syst])
    ax.set_xlabel('МГП, log P')
axs[0].set_ylabel('Плотность вероятности')
legend = axs[-1].get_legend()
handles = legend.legend_handles
labels = data['CHL amount, %'].unique()
legend.remove()
fig.legend(handles, labels, title='Концентрация ХС, %', loc='upper center',
           bbox_to_anchor=(0.5, 0), ncol=6)
fig.patch.set_facecolor('white')
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'surface' /
            'mhp_values_distr.png',
            bbox_inches='tight', dpi=300)

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
for ax, syst in zip(axs, ['dopc', 'dops']):
    data = df[(df['system'] == syst)
              & (df['mol_name'] == 'CHL')
              & (df['CHL amount, %'] != 0)]
    sns.kdeplot(data=data, x='mhp', hue='CHL amount, %', ax=ax,
                legend=ax == axs[-1], palette='crest_r',
                common_norm=False, fill=True, alpha=.2)
    ax.axvline(-0.5, ls=':', c='k')
    ax.axvline(0.5, ls=':', c='k')
    ax.set_title(TO_RUS[syst])
    ax.set_xlabel('МГП, log P')
axs[0].set_ylabel('Плотность вероятности')
legend = axs[-1].get_legend()
handles = legend.legend_handles
labels = data['CHL amount, %'].unique()
legend.remove()
fig.legend(handles, labels, title='Концентрация ХС, %', loc='upper center',
           bbox_to_anchor=(0.5, 0), ncol=6)
fig.patch.set_facecolor('white')
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'surface' /
            'chl_mhp_values_distr.png',
            bbox_inches='tight', dpi=300)


# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
for ax, syst in zip(axs, ['dopc', 'dops']):
    data = df[(df['system'] == syst)
              & (df['mol_name'].isin(
                  ['DMPC', 'DPPC', 'DSPC', 'POPC', 'DOPC', 'DOPS']))]
    sns.kdeplot(data=data, x='mhp', hue='CHL amount, %', ax=ax,
                legend=ax == axs[-1], palette='RdYlGn_r',
                common_norm=False, fill=True, alpha=.2)
    ax.axvline(-0.5, ls=':', c='k')
    ax.axvline(0.5, ls=':', c='k')
    ax.set_title(TO_RUS[syst])
    ax.set_xlabel('МГП, log P')
axs[0].set_ylabel('Плотность вероятности')
legend = axs[-1].get_legend()
handles = legend.legend_handles
labels = data['CHL amount, %'].unique()
legend.remove()
fig.legend(handles, labels, title='Концентрация ХС, %', loc='upper center',
           bbox_to_anchor=(0.5, 0), ncol=6)
fig.patch.set_facecolor('white')
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'surface' /
            'pl_mhp_values_distr.png',
            bbox_inches='tight', dpi=300)


# %%

# %%
# mhp area ratios

chol = False
df = pd.read_csv(
    PATH / 'notebooks' / 'mhpmaps' /
    'for_hists_fractions_stats_pl_'
    f'{trj_slices[0].b}-{trj_slices[0].e}-'
    f'{trj_slices[0].dt}.csv') if not chol else pd.read_csv(
        PATH / 'notebooks' / 'mhpmaps' /
        'for_hists_fractions_stats_chol_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-'
        f'{trj_slices[0].dt}.csv')

df[['phob', 'phil',	'neutr', 'phob_std', 'phil_std', 'neutr_std']] = (
    df[['phob', 'phil',	'neutr', 'phob_std', 'phil_std', 'neutr_std']]
    * 100)
df.set_index('index', inplace=True)

EXPERIMENTS['add'] = ['dopc', 'dops']

fig, ax = plt.subplots(figsize=(15, 7))
n = 3 if chol else 4
plot_mhp_area_single_exp(ax, df, 'add', True, n)
ax.set_title('')
if not chol:
    ax.set_ylim(0, df.phil.max() + df.phil_std.max())
ax.set_ylabel('% площади')
labels = [TO_RUS[item.get_text()] for item in ax.get_xticklabels()]
ax.set_xticklabels(labels)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles,
           map(lambda x:
               f'{TO_RUS[x.split(", ",1)[0]]}, '
               f'{x.split(", ",1)[1][:-3]}ХС', labels),
           title='Полярность участков, концентрация ХС',
           loc='upper center', bbox_to_anchor=(0.5, 0), ncol=n)
fig.patch.set_facecolor('white')
if not chol:
    fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'surface' /
                'pl_mhp_area_ratios.png',
                bbox_inches='tight', dpi=300)
else:
    fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'surface' /
                'chl_mhp_area_ratios.png',
                bbox_inches='tight', dpi=300)


# %%
# mhp clusterization data
option = 'hydrophobic'
lifetimes = True

x = 'cluster lifetime, ps' if lifetimes else 'cluster size, Å'
lt = '_lt' if lifetimes else ''

systs = ['dopc', 'dops']
systs_exp = flatten([
    [syst] + [syst + f'_chol{i}' for i in (10, 30, 50)]
    for syst in systs])
trjs = [[i for i in trj_slices if i.system.name == s][0]
        for s in systs_exp]

df = pd.DataFrame.from_dict({
    'index': systs_exp,
    'system': [i.split('_chol', 1)[0] for i in systs_exp],
    'CHL amount, %': ['0' if len(i.split('_chol', 1)) == 1
                      else i.split('_chol', 1)[1]
                      for i in systs_exp],
    x: [np.loadtxt(
        PATH / 'notebooks' / 'mhpmaps' / 'clust' /
        f'{trj.system.name}_{trj.b}-{trj.e}-'
        f'{trj.dt}_{option}{lt}.txt') for trj in trjs],
})

df = df.explode(x, ignore_index=True)
fig, axs = plt.subplots(1, 2, figsize=(15, 7),
                        sharey=True, sharex=True)
for syst, ax in zip(df['system'].unique(), axs):
    data = df[df['system'] == syst]
    if lifetimes:
        data = data[data[x] >= 10]
        binwidth = .1
        sns.histplot(data=data, x=x, alpha=.2, hue='CHL amount, %',
                     stat='density', fill=True, binwidth=binwidth,
                     legend=ax == axs[-1], common_norm=False,
                     log_scale=True,
                     palette='RdYlGn_r', ax=ax)

        sns.histplot(data=data, x=x, hue='CHL amount, %',
                     stat='density', fill=False, binwidth=binwidth,
                     legend=False, common_norm=False,
                     log_scale=True,
                     palette='RdYlGn_r', ax=ax, lw=2)
    else:
        sns.histplot(data=data, x=x, alpha=.2, lw=0,
                     hue='CHL amount, %',
                     palette='RdYlGn_r', stat='density', ax=ax,
                     binwidth=.15, log_scale=True,
                     common_norm=False,
                     legend=False)
        sns.histplot(data=data, x=x, lw=2, fill=False, alpha=.5,
                     legend=False,
                     element='step', hue='CHL amount, %',
                     palette='RdYlGn_r', stat='density', ax=ax,
                     binwidth=.15, log_scale=True,
                     common_norm=False,)
        sns.kdeplot(data=data, x=x, lw=5,
                    hue='CHL amount, %',
                    palette='RdYlGn_r', ax=ax, common_norm=False,
                    log_scale=True,
                    legend=ax == axs[-1])

        ax.set_xlim(10)
    ax.set_title(TO_RUS[syst])
    ax.set_xlabel(TO_RUS[x])
axs[0].set_ylabel('Плотность вероятности')

legend = axs[-1].get_legend()
handles = legend.legend_handles
labels = data['CHL amount, %'].unique()
legend.remove()
fig.legend(handles, labels, title='Концентрация ХС, %', loc='upper center',
           bbox_to_anchor=(0.5, 0), ncol=6)
fig.patch.set_facecolor('white')
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'surface' /
            f'clust_{option}{lt}_more10.png',
            bbox_inches='tight', dpi=300)


# %%
# relief and chl position + mhp maps
dopc_trj_slices = [TrajectorySlice(
    System(PATH, s, 'pbcmol_201.xtc', '201_ns.tpr'), 200, 201, 50)
    for s in ['dopc'] + [f'dopc_chol{i}' for i in [10, 30, 50]]]
dops_trj_slices = [TrajectorySlice(
    System(PATH, s, 'pbcmol_201.xtc', '201_ns.tpr'), 200, 201, 50)
    for s in ['dops'] + [f'dops_chol{i}' for i in [10, 30, 50]]]


# %%

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
        for trj in dopc_trj_slices}

for ts in (range(int(
        (dopc_trj_slices[0].e * 1000 - dopc_trj_slices[0].b * 1000)
        / dopc_trj_slices[0].dt))):

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    axs = axs.flatten()

    for ax, trj in zip(axs, dopc_trj_slices):
        mh = plot_mhpmap(data[trj]['mhp'][ts], ax,
                         bside=data[trj]['bside'],
                         resolution=75,
                         title=trj.system.name)

        h = plot_relief(data[trj]['relief'][ts], ax)
        # if 'chol' in trj.system.name:
        plot_chols_surface(data[trj]['where_chols'][ts], ax)
        try:
            name, chl = trj.system.name.split('_chol', 1)
        except ValueError:
            name, chl = trj.system.name, '0'
        ax.set_title(f'{TO_RUS[name]}, {chl}% ХС')

    cax1 = fig.add_axes([0.3, 0.05, 0.4, 0.015])
    cax2 = fig.add_axes([0.95, 0.3, 0.015, 0.4])
    clb1 = fig.colorbar(h, cax=cax1, orientation='horizontal',
                        shrink=0.5)
    clb2 = fig.colorbar(mh, cax=cax2,
                        ticks=[-2, -(2 / 2), 0, 2 / 2, 2],
                        shrink=0.7)

    clb1.ax.set_title('Высота')
    clb2.ax.set_ylabel('МГП, log P', rotation=270)
    handles, labels = axs[1].get_legend_handles_labels()
    axs[1].legend(handles, map(lambda x: TO_RUS[x], labels),
                  loc='upper right', bbox_to_anchor=(1.3, 1.02),
                  frameon=False)
    # fig.suptitle(
    #     f'{int(trj_slices[0].b*1000 + trj_slices[0].dt * ts)} пс')

    fig.patch.set_facecolor('white')
    fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'surface' /
                f'dopc_relief_mhp.png',
                bbox_inches='tight', dpi=300)
    break


# %%
# MHP near hydrophobic clusters


# %%

df = get_chl_phob_neighbors_df(trj_slices)

# %%
systs = ['dopc', 'dops']

fig, ax = plt.subplots(figsize=(7, 7))
data = df[df['system'].isin(systs)]
sns.violinplot(data=data,
               x='system', y='% of CHL near phob',
               hue='CHL amount, %', ax=ax, inner='quartile',
               edgecolor='k', palette='RdYlGn_r'
               )
ax.set_ylim(0)
labels = [TO_RUS[item.get_text()] for item in ax.get_xticklabels()]
ax.set_xticklabels(labels)
ax.set_ylabel('% точек поверхности ХС\nна границе с гидрофобными участками')
ax.set_xlabel('Система')
sns.move_legend(ax,
                title='Концентрация ХС, %',
                loc='upper center',
                bbox_to_anchor=(0.5, -0.2), ncol=6)

fig.patch.set_facecolor('white')
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'surface' /
            f'chl_border_with_phob.png',
            bbox_inches='tight', dpi=300)


# %%
# scd near chol
df_near_chol = pd.read_csv(PATH / 'notebooks' / 'scd' / 'near_chl' /
                           f'scd_near_chl_dt{trj_slices[0].dt}.csv')
df_not_near_chol = pd.read_csv(PATH / 'notebooks' / 'scd' / 'near_chl' /
                               f'scd_not_near_chl_dt{trj_slices[0].dt}.csv')
df_near_chol['near_chol'] = 'yes'
df_not_near_chol['near_chol'] = 'no'
df = pd.concat([df_near_chol, df_not_near_chol], ignore_index=True)


def dfol(
        df: pd.DataFrame, system: str,
        chain: str, chl_amount: int) -> pd.DataFrame:
    '''
    extract data for one line in plot from df
    '''
    return df[(df['system'] == system)
              & (df['chain'] == chain)
              & (df['CHL amount, %'] == chl_amount)]


df['atom_n'] = df['atom'].apply(lambda x: int(x[2:]))
scd_ms = df.drop(columns=['timepoint', 'atom']).groupby(
    ['system', 'CHL amount, %', 'chain', 'atom_n', 'near_chol'],
    as_index=False).agg(['mean', 'std'])
scd_ms = scd_ms.reset_index(level=1).reset_index(
    level=1).reset_index(level=1).reset_index()


# %%
for syst in scd_ms['system'].unique():
    fig, axs = plt.subplots(1, 3, figsize=(20, 7),
                            sharex=True, sharey=True)
    for ax, chl_amount in zip(axs, scd_ms['CHL amount, %'].unique()):
        for c, near_chol in enumerate(scd_ms['near_chol'].unique()):
            scd_ms_part = scd_ms[scd_ms['near_chol'] == near_chol]
            for sn, ls in zip(('sn-1', 'sn-2'),
                              ('-', '--')):
                subdf = dfol(scd_ms_part, syst, sn, chl_amount)
                ax.errorbar(
                    x=subdf['atom_n'],
                    y=subdf['scd']['mean'],
                    yerr=subdf['scd']['std'],
                    ls=ls, color=f'C{c}',
                    elinewidth=1, label=f'{sn}, '
                    f'{near_chol}'
                )
        ax.set_title(f'{chl_amount}% ХС')
        ax.set_xlabel('Номер атома углерода')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    axs[0].set_ylabel('Scd')
    handles, labels = axs[0].get_legend_handles_labels()
    labels = [i.split(', ', 1)[0] + ', ' + TO_RUS[i.split(', ', 1)[1]].lower() for i in labels]
    fig.legend(handles, labels, title='Ацильная цепь, молекула контактирует с ХС',
               loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2)
    fig.patch.set_facecolor('white')
    fig.savefig(PATH / 'notebooks' / 'dopc_dops' /
                f'{syst}_scd_near_chol.png',
                bbox_inches='tight', dpi=300)


# %%
# lateral distribution
trj_slices = [TrajectorySlice(System(
    PATH, s), 199, 200, 10) for s in systems]
mol = 'CHL'
thresh = 12

all_counts = pd.read_csv(PATH / 'notebooks' / 'gclust' /
                         f'{mol}_clusters_{trj_slices[0].b}-'
                         f'{trj_slices[0].e}-'
                         f'{trj_slices[0].dt}_thresh_{thresh}_coms.csv')


# %%

all_counts['component'] = np.where(
    all_counts['1'] == 1,
    'vertical',
    np.where(all_counts['2'] == 1,
             'horizontal', np.nan))

df2 = all_counts.groupby(
    ['timepoint', 'system', 'CHL amount, %', 'monolayer', 'component']).agg(
    cluster_size=('label', 'value_counts')).reset_index()


# %%
palette = sns.color_palette('Paired')
fig, axs = plt.subplots(1, 2, figsize=(
    15, 7), sharex=True, sharey=True)
for syst, ax in zip(['dopc', 'dops'], axs):
    hists = {}
    for chol_amount in df2['CHL amount, %'].unique():
        for comp in df2['component'].unique():
            hists[(chol_amount, comp)] = np.histogram(
                df2[(df2['system'] == syst) &
                    (df2['CHL amount, %'] == chol_amount) &
                    (df2['component'] == comp)]['cluster_size'],
                bins=np.arange(
                    1, 40, 3),
                density=True)

    width = 1

    ax.bar(hists[(10, 'vertical')][1][:-1] - width,
           hists[(10, 'vertical')][0], width, ec='k',
           color=palette[1], label='"вертикальная", 10% ХС')
    ax.bar(hists[(10, 'horizontal')][1][:-1] - width,
           hists[(10, 'horizontal')][0], width, ec='k',
           color=palette[0], label='"горизонтальная", 10% ХС',
           bottom=hists[(10, 'vertical')][0])
    ax.bar(hists[(30, 'vertical')][1][:-1],
           hists[(30, 'vertical')][0], width, ec='k',
           color=palette[3], label='"вертикальная", 30% ХС')
    ax.bar(hists[(30, 'horizontal')][1][:-1],
           hists[(30, 'horizontal')][0], width, ec='k',
           color=palette[2], label='"горизонтальная", 30% ХС',
           bottom=hists[(30, 'vertical')][0])
    ax.bar(hists[(50, 'vertical')][1][:-1] + width,
           hists[(50, 'vertical')][0], width, ec='k',
           color=palette[5], label='"вертикальная", 50% ХС')
    ax.bar(hists[(50, 'horizontal')][1][:-1] + width,
           hists[(50, 'horizontal')][0], width, ec='k',
           color=palette[4], label='"горизонтальная", 50% ХС',
           bottom=hists[(50, 'vertical')][0])
    # ax.set_xlim(left=0)
    ax.set_xlabel('Размер кластера (число молекул)')
    ax.set_title(TO_RUS[syst])
    # ax.set_yscale('log')
axs[0].set_ylabel('Плотность вероятности')
fig.suptitle(f'Отсечка по расстоянию: {thresh} Å', fontsize=18)
handles, labels = axs[-1].get_legend_handles_labels()
fig.legend(handles, labels,
           title='Компонента угла α, концентрация ХС',
           loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3)
fig.patch.set_facecolor('white')
# fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'lateral_distr' /
#             f'{mol}_{thresh}_cluster_sizes.png',
#             bbox_inches='tight', dpi=300)


# %%
mol = 'CHL'
thresh = 12

all_counts = pd.read_csv(PATH / 'notebooks' / 'gclust' /
                         f'{mol}_clusters_{trj_slices[0].b}-'
                         f'{trj_slices[0].e}-'
                         f'{trj_slices[0].dt}_thresh_{thresh}.csv')


# %%
all_counts['o_to_bil_c'] = np.abs(all_counts['z_com'] - all_counts['zmem'])
grouped_df = all_counts.groupby(['timepoint', 'system', 'CHL amount, %', 'monolayer'])
median_values = grouped_df['o_to_bil_c'].transform('mean')


# %%
# Creating the new column based on the median comparison
all_counts['o_position'] = np.where(all_counts['o_to_bil_c'] < median_values, 'deep', 'shallow')


# %%

df2 = all_counts.groupby(
    ['timepoint', 'system', 'CHL amount, %', 'monolayer', 'o_position']).agg(
    cluster_size=('label', 'value_counts')).reset_index()

palette = sns.color_palette('Paired')


# %%
fig, axs = plt.subplots(1, 2, figsize=(
    15, 7), sharex=True, sharey=True)
for syst, ax in zip(['dopc', 'dops'], axs):
    hists = {}
    for chol_amount in df2['CHL amount, %'].unique():
        for comp in df2['o_position'].unique():
            hists[(chol_amount, comp)] = np.histogram(
                df2[(df2['system'] == syst) &
                    (df2['CHL amount, %'] == chol_amount) &
                    (df2['o_position'] == comp)]['cluster_size'],
                bins=np.arange(
                    1, 40, 3),
                density=True)

    width = 1

    ax.bar(hists[(10, 'deep')][1][:-1] - width,
           hists[(10, 'deep')][0], width, ec='k',
           color=palette[1], label='заглублен, 10% ХС')
    ax.bar(hists[(10, 'shallow')][1][:-1] - width,
           hists[(10, 'shallow')][0], width, ec='k',
           color=palette[0], label='у поверхности, 10% ХС',
           bottom=hists[(10, 'deep')][0])
    ax.bar(hists[(30, 'deep')][1][:-1],
           hists[(30, 'deep')][0], width, ec='k',
           color=palette[3], label='заглублен, 30% ХС')
    ax.bar(hists[(30, 'shallow')][1][:-1],
           hists[(30, 'shallow')][0], width, ec='k',
           color=palette[2], label='у поверхности, 30% ХС',
           bottom=hists[(30, 'deep')][0])
    ax.bar(hists[(50, 'deep')][1][:-1] + width,
           hists[(50, 'deep')][0], width, ec='k',
           color=palette[5], label='заглублен, 50% ХС')
    ax.bar(hists[(50, 'shallow')][1][:-1] + width,
           hists[(50, 'shallow')][0], width, ec='k',
           color=palette[4], label='у поверхности, 50% ХС',
           bottom=hists[(50, 'deep')][0])
    # ax.set_xlim(left=0)
    ax.set_xlabel('Размер кластера (число молекул)')
    ax.set_title(TO_RUS[syst])
    # ax.set_yscale('log')
axs[0].set_ylabel('Плотность вероятности')
fig.suptitle(f'Отсечка по расстоянию: {thresh} Å', fontsize=18)
handles, labels = axs[-1].get_legend_handles_labels()
fig.legend(handles, labels,
           title='Положение O ХС, концентрация ХС',
           loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3)
fig.patch.set_facecolor('white')
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'lateral_distr' /
            f'{mol}_{thresh}_O_pos_cluster_sizes.png',
            bbox_inches='tight', dpi=300)
# %%
mol = 'PL'
thresh = 7
max_clsize = None

all_counts = pd.read_csv(PATH / 'notebooks' / 'gclust' /
                         f'{mol}_clusters_{trj_slices[0].b}-'
                         f'{trj_slices[0].e}-'
                         f'{trj_slices[0].dt}_thresh_{thresh}.csv')

df2 = all_counts.groupby(
    ['timepoint', 'system', 'CHL amount, %', 'monolayer']).agg(
    cluster_size=('label', 'value_counts')).reset_index()


fig, axs = plt.subplots(1, 2, figsize=(
    15, 7), sharex=True, sharey=True)
for syst, ax in zip(['dopc', 'dops'], axs):
    subdf = df2[(df2['system'] == syst)].copy()
    # FIXME: dirty fix cluster sizes up to 200 in dopc and dppc_325
    if syst in ['dopc', 'dppc_325']:
        subdf['cluster_size'] = subdf['cluster_size'] / 2

    sns.histplot(data=subdf, x='cluster_size',
                 ax=ax, hue='CHL amount, %',
                 palette='RdYlGn_r', common_norm=False,
                 # fill=True, cut=0, bw_adjust=1,
                 stat='density',
                 binwidth=.25,
                 multiple='dodge',
                 edgecolor='black', linewidth=2,
                 log_scale=True,
                 legend=ax == axs[-1])

    ax.set_xlabel('Размер кластера (число молекул)')
    ax.set_title(TO_RUS[syst])
    ax.set_xlim(1, max_clsize)
axs[0].set_ylabel('Плотность вероятности')
fig.suptitle(f'Отсечка по расстоянию: {thresh} Å', fontsize=18)

legend = axs[-1].get_legend()
handles = legend.legend_handles
labels = subdf['CHL amount, %'].unique()
legend.remove()
fig.legend(handles, labels, title='Концентрация ХС, %', loc='upper center',
           bbox_to_anchor=(0.5, 0), ncol=6)
fig.patch.set_facecolor('white')
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'lateral_distr' /
            f'{mol}_{thresh}_cluster_sizes.png',
            bbox_inches='tight', dpi=300)


# %%
# rdfs
systems = flatten([(i + '_chol10', i + '_chol30', i + '_chol50')
                   for i in ['dopc', 'dops']])
systems = list(dict.fromkeys(systems))
trj_slices = [TrajectorySlice(System(
    PATH, s),
    150, 200, 20) for s in systems]

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
        # if 'dops' in systs[0].system.name:
        #     df_oh9 = pd.read_csv(
        #         PATH / trj.system.name / 'a' / 'OH9_rdf.csv')
        #     df_oh10 = pd.read_csv(
        #         PATH / trj.system.name / 'a' / 'OH10_rdf.csv')
        #     ax.plot(df_n['# r'], df_oh9['g'], label='OH9')
        #     ax.plot(df_p['# r'], df_oh10['g'], label='OH10')
        ax.set_xlabel('Расстояние до атома O ХС, Å')
        sname, chl = trj.system.name.split('_chol', 1)
        ax.set_title(f'{TO_RUS[sname]}, {chl}% ХС')
        ax.set_xlim(0, 20)
    axs[0].set_ylabel('g(r)')
    axs[1].legend(title='Атом ФЛ',
                  loc='upper center',
                  bbox_to_anchor=(0.5, -0.15), ncol=4)
    fig.patch.set_facecolor('white')
    fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'lateral_distr' /
                f'{systs[0].system.name.split("_chol",1)[0]}_only_P_N_chl_O_rdf.png',
                bbox_inches='tight', dpi=300)


# %%
systems = flatten([(i + '_chol10', i + '_chol30', i + '_chol50')
                   for i in ['dopc', 'dops']])
systems = list(dict.fromkeys(systems))
trj_slices = [TrajectorySlice(System(
    PATH, s),
    150, 200, 20) for s in systems]


fig, axs = plt.subplots(1, 3, figsize=(20, 7),
                        sharex=True, sharey=True)
for ax, c in zip(axs, range(3)):
    for trj in trj_slices[c::3]:
        df_o = pd.read_csv(
            PATH / trj.system.name / 'a' / 'O_rdf.csv')
        df_c17 = pd.read_csv(
            PATH / trj.system.name / 'a' / 'C_rdf.csv')
        df_c6 = pd.read_csv(
            PATH / trj.system.name / 'a' / 'C6_rdf.csv')
        df_c11 = pd.read_csv(
            PATH / trj.system.name / 'a' / 'C11_rdf.csv')
        # ax.plot(df_o['# r'], df_o['g'],
        #     label=TO_RUS[trj.system.name.split('_chol')[0]])
        ax.plot(df_c17['# r'], df_c17['g'],
                label=TO_RUS[trj.system.name.split('_chol')[0]])
    # ax.plot(df_c6['# r'], df_c6['g'], label='C11')
    # ax.plot(df_c11['# r'], df_c11['g'], label='C17')
    ax.set_xlabel('Расстояние между атомами ХС, Å')
    sname, chl = trj.system.name.split('_chol', 1)
    ax.set_title(f'{chl}% ХС')
    ax.set_xlim(0, 6)
axs[0].set_ylabel('g(r)')
axs[1].legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15), ncol=4)
fig.patch.set_facecolor('white')


# %%
fig.savefig(PATH / 'notebooks' / 'rdf' /
            f'{systs[0].system.name.split("_chol",1)[0]}_C6_chl.png',
            bbox_inches='tight', dpi=300)

# %%
# LATERAL CLUSTERIZATIO
angle_mhp_z = pd.read_csv(
    PATH / 'notebooks' / 'integral_parameters' /
    'angle_mhp_z_200.0-201.0-1.csv')

angle_mhp_z['distance to bilayer center, Å'] = pd.cut(
    angle_mhp_z['distance to bilayer center'],
    bins=[0, 3, 6, 9, 12, 15, 100],
    labels=['<= 3', '3-6', '6-9', '9-12', '12-15', '> 15'])

# %%
# obtain_cluster_labels(trj_slices, thresh=12)

# %%
mol = 'CHL'
thresh = 12
hue = 'distance to bilayer center, Å'
system = 'dopc'
# %%


all_counts = pd.read_csv(PATH / 'notebooks' / 'gclust' /
                         f'{mol}_clusters_{trj_slices[0].b}-'
                         f'{trj_slices[0].e}-'
                         f'{trj_slices[0].dt}_thresh_{thresh}.csv')

all_counts = all_counts.merge(angle_mhp_z[['system', 'timepoint', 'chl_index', 'CHL amount, %', 'surface', 'distance to bilayer center, Å']], on=[
                              'system', 'timepoint', 'chl_index', 'CHL amount, %'], how='left')

all_counts['surface'] = all_counts['surface'].map({1: 'yes', 0: 'no'})

all_counts['tilt component'] = np.where(
    all_counts['1'] == 1,
    'vertical',
    np.where(all_counts['2'] == 1,
             'horizontal', np.nan))
all_counts = all_counts[all_counts['tilt component'] != 'nan']
df2 = all_counts.groupby(
    ['timepoint', 'system', 'CHL amount, %', 'monolayer', hue]).agg(
    cluster_size=('label', 'value_counts')).reset_index()
# %%

data = df2[df2['system'] == system]

fig, axs = plt.subplots(1, 3, figsize=(
    20, 7), sharex=True, sharey=True)
for ax, chl_amount in zip(axs, data['CHL amount, %'].unique()):
    sns.histplot(data=data[data['CHL amount, %'] == chl_amount],
                 x='cluster_size',
                 ax=ax, hue=hue,
                 hue_order=['<=3', '3-6', '6-9', '9-12', '12-15', '>15'],
                 palette=sns.color_palette('crest_r'),
                 multiple='stack',
                 stat='density',
                 discrete=True,
                 legend=ax == axs[1],
                 edgecolor='k')
    ax.set_xlabel('Размер кластера (число молекул)')
    ax.set_title(f'{chl_amount}% ХС')
# ax.set_yscale('log')
axs[0].set_ylabel('Плотность вероятности')
fig.suptitle(f'Отсечка по расстоянию: {thresh} Å', fontsize=18)


legend = axs[1].get_legend()
handles = legend.legend_handles
labels = ['<=3', '3-6', '6-9', '9-12', '12-15', '>15']
legend.remove()
fig.legend(handles, labels, title=TO_RUS[hue], loc='upper center',
           bbox_to_anchor=(0.5, 0), ncol=6)
fig.patch.set_facecolor('white')
fig.savefig(PATH / 'notebooks' / 'dopc_dops' / 'lateral_distr' /
            f'{system}_{mol}_{thresh}_{hue}.png',
            bbox_inches='tight', dpi=300)

# %%

# some arbitrary data to plot
x = numpy.linspace(0, 2 * numpy.pi, 30)
y = numpy.linspace(0, 2 * numpy.pi, 20)
[X, Y] = numpy.meshgrid(x, y)
Z = numpy.sin(X) * numpy.cos(Y)

fig = plt.figure(figsize=(15, 7))
plt.ion()
plt.set_cmap('bwr')  # a good start: blue to white to red colormap

# a plot ranging from -1 to 1, hence the value 0 (the average) is colorcoded in white
ax = fig.add_subplot(1, 2, 1)
plt.pcolor(X, Y, Z)
plt.colorbar()

# a plot ranging from -0.2 to 0.8 hence 0.3 (the average) is colorcoded in white
ax = fig.add_subplot(1, 2, 2)

# define your scale, with white at zero
vmin = -1.1
vmax = 1.19
norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

plt.pcolor(X, Y, Z, vmin=vmin, vmax=vmax)
cbar = plt.colorbar(fraction=0.04)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('МГП, log P', rotation=270)
fig.savefig('colors.png',
            bbox_inches='tight', dpi=300)


# %%
lip_sol_rchist_full = (PATH / 'notebooks' / 'contacts' /
                       'lip_SOL_hb_hb_full_'
                       f'{trj_slices[0].b}-{trj_slices[0].e}'
                       f'-{trj_slices[0].dt}_'
                       'rchist_full.csv')


df_sol = pd.read_csv(lip_sol_rchist_full)

# %%
df_lip = pd.read_csv(PATH / 'notebooks' / 'contacts' /
                     'lip_hb_hb_full_'
                     f'{trj_slices[0].b}-{trj_slices[0].e}'
                     f'-{trj_slices[0].dt}_'
                     'rchist_full.csv')


# %%
df = pd.concat([df_sol, df_lip], ignore_index=True)

# %%


df = df[(df['dmn'] == 'CHL') | (df['amn'] == 'CHL')]
df['chl_index'] = np.where(
    df['dmn'] == 'CHL', df['dmi'],
    np.where(df['amn'] == 'CHL', df['ami'], df['dmi']))
df['other_index'] = np.where(
    df['dmn'] != 'CHL', df['dmi'], df['ami'])
df['other_name'] = np.where(
    df['dmn'] != 'CHL', df['dmn'], df['amn'])

df_dup = df.loc[df['other_name'] == 'CHL'].copy()
(df_dup['other_index'], df_dup['chl_index']) = (
    df_dup['chl_index'], df_dup['other_index'])
df = pd.concat([df, df_dup], ignore_index=True)

# %%

determine_pl_group = {'Ser': ['OH9', 'OH10', 'NH7'],
                      'PO4': ['OH2', 'OH3', 'OH4', 'OG3'],
                      "ROR'": ['OG1', 'OG2'],
                      "RC(O)R'": ['OA1', 'OB1']}
# need to reverse this dict for vectorized
# assigning of groups later
reverse_lookup_dict = {}
for key, values in determine_pl_group.items():
    for value in values:
        reverse_lookup_dict[value] = key

table_with_probs['PL group'] = table_with_probs['aan'].map(
    reverse_lookup_dict)


# %%
systems = flatten([(i + '_chol10', i + '_chol30', i + '_chol50')
                   for i in ['dopc', 'dops']])
systems = list(dict.fromkeys(systems))
trj_slices = [TrajectorySlice(System(
    PATH, s),
    150, 200, 20) for s in systems]

# %%


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
S1 = "CHOL///C11"
S2 = "CHOL///C11"'''

    with open(outname, 'w', encoding='utf-8') as f:
        f.write(args)


# %%
for trj in trj_slices:
    create_args(trj, 'C11', 'rdf_C11.args')
    os.chdir(PATH / trj.system.name / 'a')
    cmd = '/nfs/belka2/soft/impulse/dev/inst/runtask.py '\
        '-t /nfs/belka2/soft/impulse/tasks/post/rdf/rdf.mk '\
        '-f rdf_C11.args'
    subprocess.run(cmd, shell=True, check=True)


# %%

for systs in chunker(trj_slices, 3):
    fig, axs = plt.subplots(1, 3, figsize=(20, 7),
                            sharex=True, sharey=True)
    for ax, trj in zip(axs, systs):
        df_o = pd.read_csv(
            PATH / trj.system.name / 'a' / 'C11_rdf.csv')
        ax.plot(df_o['# r'], df_o['g'], label='O')
        ax.set_xlabel('distance to CHL O, Å')
        sname, chl = trj.system.name.split('_chol', 1)
        ax.set_title(f'{sname}, {chl}% CHL')
        ax.set_xlim(0, 8)
    axs[0].set_ylabel('g(r)')
    # axs[1].legend(loc='upper center',
    #               bbox_to_anchor=(0.5, -0.15), ncol=4)


# %%
    fig.savefig(PATH / 'notebooks' / 'rdf' /
                f'{systs[0].system.name.split("_chol",1)[0]}_C11_chl.png',
                bbox_inches='tight', dpi=300)
