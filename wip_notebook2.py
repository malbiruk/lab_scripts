

# %% md
# This notebook is for testing and troubleshooting code
#
# # cholesterol_angle_components_density.py


# %% md
# ### Imports and define functions

# %%

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering as AC
from matplotlib.ticker import MaxNLocator
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy import integrate
from scipy.optimize import curve_fit
import MDAnalysis as mda
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from modules import ndtest
from modules.general import flatten, opener, sparkles, duration
from modules.traj import System, TrajectorySlice
from modules.density import plot_density_profile
from modules.constants import PATH, TO_RUS, EXPERIMENTS
from integral_parameters_script import chl_tilt_angle


def add_comps_to_chl_tilt(path_to_df, b_comp=None, e_comp=None, dt_comp=None):
    '''
    add info which component is every angle in chl_tilt_{b}-{e}-{dt}.to_csv
    (file produced by chl_tilt_angle() function)
    df should have columns 'system', 'α, °' 'CHL amount, %'
    '''
    df = pd.read_csv(path_to_df)
    print('determining angle components...')
    if (b_comp, e_comp, dt_comp) == (None, None, None):
        b_comp, e_comp, dt_comp = b, e, dt
    df['comp'] = df.apply(
        lambda row: angle_refers_to_component(
            row['system'], row['α, °'], row['CHL amount, %'],
            b_comp, e_comp, dt_comp), axis=1)
    df = df.explode(['comp'])
    df['comp'] = df.comp.astype(str)
    df = df.join(pd.get_dummies(df['comp']))
    df = df.drop(columns=['comp']).drop_duplicates()
    df.to_csv(str(path_to_df).rsplit('.', 1)[0] + '_with_comps.csv',
              index=False)
    print('done.')


def obtain_df_of_density(trj_slices: list[TrajectorySlice]) -> pd.DataFrame:
    '''
    summarize data from different files of density to one df
    '''
    records = []
    for trj in trj_slices:
        for fr in range(trj_slices[0].b, trj_slices[0].e, int(trj_slices[0].dt / 1000)):
            lines = opener(Path(trj.system.dir) / 'density_profiles' /
                           f'chols_{fr}-{fr}-0_dp.xvg')
            z = np.array(
                list(map(float, [i.split()[0] for i in lines])))
            a = np.array(
                list(map(float, flatten([i.split()[1:] for i in lines]))))
            n_chols = int(a.shape[0] / z.shape[0])
            z_col = [i for i in z for _ in range(n_chols)]
            chl_indices = flatten([list(range(n_chols)) for _ in z])
            records.append(
                (trj.system.name, fr * 1000, chl_indices, z_col, a))
    df = pd.DataFrame.from_records(
        records, columns=['system', 'timepoint', 'chl_index', 'z', 'density'])
    df.sort_values('system', inplace=True, ignore_index=True)
    df['CHL amount, %'] = df['system'].str.split(
        '_chol', n=1, expand=True)[1]
    df['system'] = df['system'].str.split('_chol', n=1, expand=True)[0]
    df.replace(to_replace=[None], value=0, inplace=True)
    return df.explode(['chl_index', 'z', 'density'])


def angle_refers_to_component(syst, angle, chl, b, e, dt) -> tuple:
    '''
    depending on system and chl amount determine component:
    1 vertical, 2 horizontal
    returns tuple (1,), (2,) or (1,2)

    ctr +- 1/2 wid => angle in component
    components are located in notebooks/chol_tilt/{syst}_{b}-{e}-{dt}_4_comps.csv
    '''
    comps = pd.read_csv('/home/klim/Documents/chol_impact/'
                        f'notebooks/chol_tilt/{syst}_chol{chl}_{b}-{e}-{dt}_4_comps.csv')
    comps = comps.sort_values(['ctr']).reset_index(drop=True)
    res = []
    if angle >= 0:
        if comps.loc[3, 'ctr'] - comps.loc[3, 'wid'] / 2 <= angle:
            res.append(2)
        if angle <= comps.loc[2, 'ctr'] + comps.loc[2, 'wid'] / 2:
            res.append(1)
    if angle < 0:
        if angle <= comps.loc[0, 'ctr'] + comps.loc[0, 'wid'] / 2:
            res.append(2)
        if angle >= comps.loc[1, 'ctr'] - comps.loc[1, 'wid'] / 2:
            res.append(1)
    return tuple(res)


def create_tilt_density_table(trj_slices: list[TrajectorySlice],
                              comp_b: int, comp_e: int, comp_dt: int) -> None:
    '''
    calculate fractions of horizontal component depending on density and z coordinate
    comp_b, comp_e, comp_dt -- b, e, dt of component splitting

    formula is simple: for each system for each Z coordinta and each timepoint
    calculate (density sum by one of components) / (density sum of both components)
    '''

    df = pd.read_csv(trj_slices[0].system.path / 'notebooks' /
                     'integral_parameters' /
                     f'chl_tilt_{comp_b}-{comp_e}-{comp_dt}_with_comps.csv')
    df = df[df['timepoint'].astype(str).str.fullmatch('|'.join(
        [str(i) for i in range(trj_slices[0].b * 1000,
                               trj_slices[0].e * 1000, trj_slices[0].dt)]))].copy()
    df = df.reset_index().drop(columns=['index'])

    # print('determining angle components...')
    # df['comp'] = df.apply(
    #     lambda row: angle_refers_to_component(
    #         row['system'], row['α, °'], row['CHL amount, %'],
    #         trj_slices[0].b, trj_slices[0].e, trj_slices[0].dt), axis=1)

    print('reformatting density data...')
    dens = obtain_df_of_density(trj_slices).sort_values(
        ['system', 'CHL amount, %', 'timepoint',
         'chl_index', 'z'], ignore_index=True).drop_duplicates(ignore_index=True)
    dens['chl_index'] = dens['chl_index'].astype(int)
    dens['CHL amount, %'] = dens['CHL amount, %'].astype(int)
    common_ndx = ['system', 'CHL amount, %', 'timepoint', 'chl_index']

    print('analyzing components and densities...')
    merged = dens.merge(df, on=common_ndx, how='left').drop_duplicates(
        ignore_index=True).sort_values(
        ['system', 'CHL amount, %', 'timepoint', 'z'], ignore_index=True)
    merged['density'] = merged.density.astype(float)
    # merged = merged.explode(['comp'])
    # merged['comp'] = merged.comp.astype(str)
    # merged = merged.join(pd.get_dummies(merged['comp']))
    merged['comp_dens'] = merged['1'] * \
        merged['density'] + merged['2'] * merged['density']
    merged = merged.join(merged.groupby(
        ['system', 'CHL amount, %', 'timepoint', 'z'])['comp_dens'].sum(),
        on=['system', 'CHL amount, %', 'timepoint', 'z'], rsuffix='_sum')
    merged['% of horizontal component'] = merged['2'] * \
        merged['density'] / merged['comp_dens_sum'] * 100
    merged['% of vertical component'] = merged['1'] * \
        merged['density'] / merged['comp_dens_sum'] * 100
    merged.to_csv(trj_slices[0].system.path / 'notebooks' /
                  'integral_parameters' /
                  f'tilt_density_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv',
                  index=False)


def angl_dens_multiple_plot(experiments: dict,
                            trj_slices: list[TrajectorySlice],
                            df: pd.DataFrame, y: str, out: str, y2=None) -> None:
    '''
    plot all density_angles plots
    y -- name of variable in df which to plot
    y2 -- second variable to plot
    out -- outfile name suffix
    '''

    for exp, systs in experiments.items():
        fig, axs = plt.subplots(3, 3, figsize=(
            21, 21), sharex=True, sharey=True)
        systs = [(s + '_chol10', s + '_chol30', s + '_chol50')
                 for s in systs]
        for s_row, ax_row in zip(systs, axs):
            for s, ax in zip(s_row, ax_row):
                ax2 = ax.twinx()
                data = df[(df['system'] == s.split('_chol', 1)[0])
                          & (df['CHL amount, %'] == int(s.split('_chol', 1)[1]))]
                sns.kdeplot(data=data,
                            x='z', y=y, ax=ax2,
                            bins=50, alpha=0.5,
                            stat='density')
                if y2 is not None:
                    sns.kdeplot(data=data,
                                x='z', y=y2, ax=ax2,
                                bins=50, alpha=0.5,
                                color='C1', stat='density')

                try:
                    plot_density_profile(ax,
                                         TrajectorySlice(
                                             System(
                                                 trj_slices[0].system.path, s),
                                             trj_slices[0].b, trj_slices[0].e, trj_slices[0].dt),
                                         groups=['chols'], color='k')
                    plot_density_profile(ax,
                                         TrajectorySlice(
                                             System(
                                                 trj_slices[0].system.path, s),
                                             trj_slices[0].b, trj_slices[0].e, trj_slices[0].dt),
                                         groups=['phosphates'], color='C3')
                except FileNotFoundError as e:
                    print(e)

                if ax != ax_row[-1]:
                    ax2.set_ylabel('')
                    ax2.yaxis.set_ticklabels([])
                if ax != ax_row[0]:
                    ax.set_ylabel('')
                ax.set_xlabel('')
                ax.set_title(
                    f"{s.split('_chol', 1)[0]}, {s.split('_chol', 1)[1]} % CHL")
        for ax in axs[-1, :]:
            ax.set_xlabel('Z, nm')
        labels, handles = axs[0, 0].get_legend_handles_labels()
        fig.legend(labels, handles)
        fig.suptitle(exp)
        plt.savefig(trj_slices[0].system.path / 'notebooks' / 'integral_parameters' /
                    f'{"_".join(exp.split())}_'
                    f'density_{out}_{trj_slices[0].b}_{trj_slices[0].e}_{trj_slices[0].dt}.png',
                    bbox_inches='tight')


def plot_angles_density(experiments: dict, trj_slices: list[TrajectorySlice],
                        comp_b: int, comp_e: int, comp_dt: int) -> None:
    '''
    calculate percentage of chl tilt horizontal component in systems
    and plot it together with density plots;
    9 plots per experiment
    '''
    trj_slices = [s for s in trj_slices if 'chol' in s.system.name]
    if not (trj_slices[0].system.path / 'notebooks' /
            'integral_parameters' /
            f'tilt_density_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv'
            ).is_file():
        create_tilt_density_table(trj_slices, comp_b, comp_e, comp_dt)
    df = pd.read_csv(trj_slices[0].system.path / 'notebooks' /
                     'integral_parameters' /
                     f'tilt_density_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')
    print('plotting horizontal and vertival component distribution of angles with '
          'CHL density profiles...')
    angl_dens_multiple_plot(experiments, trj_slices, df,
                            '% of horizontal component', 'both_comps',
                            '% of vertical component')
    print('done.')


def generate_coords_comps_table(trj_slices: list[TrajectorySlice],
                                chl_tilt_b: int, chl_tilt_e: int, chl_tilt_dt: int):
    '''
    obtaining angles for trajectory slices selected if they are not calculated
    collecting coordinates of CHL cOMs and O3 positions and adding them to dataframe of angles
    then calculating components for this DataFrame
    chl_tilt_b, chl_tilt_e, chl_tilt_dt -- for components
    '''
    if not (PATH / 'notebooks' / 'integral_parameters' /
            f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}_with_comps.csv').is_file():
        chl_tilt_angle(trj_slices, no_comps=True)
    df = pd.read_csv(PATH / 'notebooks' /
                     'integral_parameters' /
                     f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')

    x_com = []
    y_com = []
    z_com = []
    x_o = []
    y_o = []
    z_o = []

    print('collecting coords...')

    for trj in trj_slices:
        print(trj)
        trj.generate_slice_with_gmx()
        u = mda.Universe(f'{trj.system.dir}/md/md.tpr',
                         f'{trj.system.dir}/md/pbcmol_{trj.b}-{trj.e}-{trj.dt}.xtc')

        for ts in u.trajectory:
            chols = u.select_atoms("resname CHL")
            chols_coms = {c: i.atoms.center_of_mass()
                          for c, i in enumerate(chols.residues)}
            o3_positions = {c: i.atoms.select_atoms("name O3").positions[0]
                            for c, i in enumerate(chols.residues)}

            for mol in chols_coms:
                # df.loc[(df['system'] == str(trj.system).rsplit('_chol',1)[0]) &
                #    (df['timepoint'] == ts.time) &
                #    (df['chl_index'] == mol) &
                #    (df['CHL amount, %'] == str(trj.system).rsplit('_chol',1)[1]),
                #    'x_com'] = chols_coms[mol][0]
                x_com.append(chols_coms[mol][0])
                y_com.append(chols_coms[mol][1])
                z_com.append(chols_coms[mol][2])
                x_o.append(o3_positions[mol][0])
                y_o.append(o3_positions[mol][1])
                z_o.append(o3_positions[mol][2])

    print('assigning lists...')

    df['x_com'] = x_com
    df['y_com'] = y_com
    df['z_com'] = z_com
    df['x_o'] = x_o
    df['y_o'] = y_o
    df['z_o'] = z_o

    df.to_csv(PATH / 'notebooks' / 'integral_parameters' /
                     f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}_coords.csv',
                     index=False)

    add_comps_to_chl_tilt((PATH / 'notebooks' / 'integral_parameters' /
                          f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}_coords.csv'),
                          chl_tilt_b, chl_tilt_e, chl_tilt_dt)


def angle_components_3d(trj_slices: list[TrajectorySlice],
                        comp_b: int, comp_e: int, comp_dt: int):
    '''
    plotting 3d graphs of CHL COMs and O3 positions
    with colorcoding the amount of horizontal component (red - 2 comp, green - 1)
    '''
    trj_slices = [s for s in trj_slices if 'chol' in s.system.name]

    if not (PATH / 'notebooks' / 'integral_parameters' /
            f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}_coords_with_comps.csv'
            ).is_file():
        generate_coords_comps_table(trj_slices, comp_b, comp_e, comp_dt)
    else:
        df = pd.read_csv(PATH / 'notebooks' / 'integral_parameters' /
                         f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}_coords_with_comps.csv'
                         )
        averaged_df = df.groupby(
            ['system', 'CHL amount, %', 'chl_index'], as_index=False).mean()
        for trj in trj_slices:
            data = averaged_df[(averaged_df['system'] == str(trj.system).rsplit('_chol', 1)[0]) &
                               (averaged_df['CHL amount, %'] == int(str(trj.system).rsplit('_chol', 1)[1]))]
            data = data.rename(columns={'z_xom': 'z_com'})
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(projection='3d')
            x = np.dstack((data['x_com'].to_numpy(), data['x_o'].to_numpy()))
            y = np.dstack((data['y_com'].to_numpy(), data['y_o'].to_numpy()))
            z = np.dstack((data['z_com'].to_numpy(), data['z_o'].to_numpy()))
            im = ax.scatter(data['x_com'], data['y_com'], data['z_com'], c=data['2'],
                            cmap='RdYlGn_r', ec='k', s=40)
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.06,
                         label='share of horizontal component')
            for i in range(len(x[0])):
                ax.plot(x[0][i], y[0][i], z[0][i], color='gray', lw=1)
            ax.set_title(str(trj.system) + ' (199-200 ns, dt=10 ps)')
            ax.set_xlabel('X, Å')
            ax.set_ylabel('Y, Å')
            ax.set_zlabel('Z, Å')
            fig.savefig(PATH / 'notebooks' / 'chol_tilt' / 'components_3d_plots' /
                        f'tilt_comps_3d_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.png',
                        bbox_inches='tight'
                        )

# %% md
# ## Creating angle_components_3d function
#
# 1. It should add x, y, z coordinates to COMs and O atoms of CHL
# for each CHL molecule in each timestep specified by user input
# (add data to subset of larger chl_tilt_with_comps table)
# > 1. use gmx trjconv to obtain slices of trajectory
# > 2. run MDAnalysis on it to obtain coords of COMs and O atoms
#
# 1. Create 3D plot from this data (average 100 frames 199-200 ns, dt=100 ps)
# > **averaging:**
# > - just get mean of coords for each molecule;
# > - for components: get sum of components for each molecule
# (i.e. in how many frames there is 1 in both columns '1' and '2', than divide n of frames
# with 1 in '1' by this sum -- it will be ratio of '1' componen than apply colormap;
# before this procedure rows with 1 in column 'nan' should be deleted)


# %%
systems = flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                   for i in flatten(EXPERIMENTS.values())])

trj_slices = [TrajectorySlice(
    System(PATH, s), 199, 200, 10) for s in systems]

chl_tilt_b, chl_tilt_e, chl_tilt_dt = 100, 200, 100


def angle_components_3d(trj_slices: list[TrajectorySlice],
                        comp_b: int, comp_e: int, comp_dt: int):
    '''
    plotting 3d graphs of CHL COMs and O3 positions
    with colorcoding the amount of horizontal component (red - 2 comp, green - 1)
    '''
    trj_slices = [s for s in trj_slices if 'chol' in s.system.name]

    if not (PATH / 'notebooks' / 'integral_parameters' /
            f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}_coords_with_comps.csv'
            ).is_file():
        generate_coords_comps_table(trj_slices, comp_b, comp_e, comp_dt)
    else:
        df = pd.read_csv(PATH / 'notebooks' / 'integral_parameters' /
                         f'chl_tilt_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}_coords_with_comps.csv'
                         )
        averaged_df = df.groupby(
            ['system', 'CHL amount, %', 'chl_index'], as_index=False).mean()
        for trj in trj_slices:
            data = averaged_df[(averaged_df['system'] == str(trj.system).rsplit('_chol', 1)[0]) &
                               (averaged_df['CHL amount, %'] == int(str(trj.system).rsplit('_chol', 1)[1]))]
            data = data.rename(columns={'z_xom': 'z_com'})
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(projection='3d')
            x = np.dstack((data['x_com'].to_numpy(), data['x_o'].to_numpy()))
            y = np.dstack((data['y_com'].to_numpy(), data['y_o'].to_numpy()))
            z = np.dstack((data['z_com'].to_numpy(), data['z_o'].to_numpy()))
            im = ax.scatter(data['x_com'], data['y_com'], data['z_com'], c=data['2'],
                            cmap='RdYlGn_r', ec='k', s=40)
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.06,
                         label='share of horizontal component')
            for i in range(len(x[0])):
                ax.plot(x[0][i], y[0][i], z[0][i], color='gray', lw=1)
            ax.set_title(str(trj.system) + ' (199-200 ns, dt=10 ps)')
            ax.set_xlabel('X, Å')
            ax.set_ylabel('Y, Å')
            ax.set_zlabel('Z, Å')
            fig.savefig(PATH / 'notebooks' / 'chol_tilt' / 'components_3d_plots' /
                        f'tilt_comps_3d_{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.png',
                        bbox_inches='tight'
                        )


# %%
trj


# %%
trj
data = df[(df['system'] == str(trj.system).rsplit('_chol', 1)[0]) & (df['timepoint'] == 199000) &
          (df['CHL amount, %'] == int(str(trj.system).rsplit('_chol', 1)[1]))]
data = data.rename(columns={'z_xom': 'z_com'})
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')
x = np.dstack((data['x_com'].to_numpy(), data['x_o'].to_numpy()))
y = np.dstack((data['y_com'].to_numpy(), data['y_o'].to_numpy()))
z = np.dstack((data['z_com'].to_numpy(), data['z_o'].to_numpy()))
ax.scatter(data['x_com'], data['y_com'], data['z_com'],
           c=data['2'], cmap='RdYlGn_r', ec='k', s=40)
for i in range(len(x[0])):
    ax.plot(x[0][i], y[0][i], z[0][i], color='gray', lw=1)

# %%


df[(df['system'] == 'dmpc') & (df['CHL amount, %'] == 10)]


# %%
len(x_com)

# %%
for ts in u.trajectory:
    ts.time

# %%
df2 = pd.read_csv(PATH / 'notebooks' / 'integral_parameters' /
                  f'chl_tilt_100-200-100_with_comps.csv')

df2
# %%
len(df2[df2['1'] == 1])

# %%
len(df2[df2['2'] == 1])
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%


fig, ax = plt.subplots()
lines = opener(f'{trj.system.path}/notebooks/chol_tilt/'
               f'dmpc_chol10_100-200-100_tilt.xvg')

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

guess = [-30, 0.005, 20, -15, 0.02, 7, 15, 0.02, 7, 30, 0.005, 20]

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
ax.plot(x, func(x, *popt[:3]), '--')
ax.plot(x, func(x, *popt[3:6]), '--')
ax.plot(x, func(x, *popt[6:9]), '--')
ax.plot(x, func(x, *popt[9:]), '--')

stats.kstest(y, func(x, *popt))


# %%


fig, ax = plt.subplots()
lines = opener(f'{trj.system.path}/notebooks/chol_tilt/'
               f'dmpc_chol10_100-200-100_tilt.xvg')

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

guess = [-30, 0.005, 20, 30, 0.005, 20]

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
ax.plot(x, func(x, *popt[:3]), '--')
ax.plot(x, func(x, *popt[3:6]), '--')
# ax.plot(x, func(x, *popt[6:9]), '--')
# ax.plot(x, func(x, *popt[9:]), '--')

stats.kstest(y, func(x, *popt))


# %%


X = a.reshape(-1, 1)

N = np.arange(1, 10)
models = [None for i in range(len(N))]

for i in range(len(N)):
    models[i] = GaussianMixture(
        N[i], tol=0.00001, max_iter=1000, reg_covar=1e-12).fit(X)

AIC = [m.aic(X) for m in models]
BIC = [m.bic(X) for m in models]

M_best = models[1]


# %%
# plot AIC and BIC
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].plot(N, AIC, '-k', label='AIC')
ax[0].plot(N, BIC, '--k', label='BIC')
ax[0].set_xlabel('n. components')
ax[0].set_ylabel('information criterion')
ax[0].legend()
ax[0].set_title('BIC and AIC for each n of components')

ax[1].plot(N, np.gradient(BIC), '-bs')
ax[1].set_xlabel('n. components')
ax[1].set_ylabel('grad (BIC)')
ax[1].set_title('Gradient of BIC for each n of components')
