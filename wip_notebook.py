import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA

from modules.traj import System, TrajectorySlice
from modules.constants import PATH

sns.set(context='notebook', palette='muted', style='ticks', rc={'figure.figsize':(9,6)})

# %%
syst = 'dmpc_chol10'

# %%
u = mda.Universe(str(PATH / syst / 'md' / 'md.tpr'),
                 str(PATH / syst / 'md' / 'last_ns.xtc'))


# %%
hbonds = HBA(universe=u,
             between=['resname CHL', 'resname DMPC'],
             donors_sel='name O3')
hbonds.run()

# %%
hbonds.results.hbonds.shape

# %%
def hb_lifetime(tau_max: int = 20, step: int = 10):
    '''
    calculate lifetimes adjusting tau_max
    '''
    tau_timeseries, timeseries = hbonds.lifetime(tau_max)
    if tau_timeseries[np.argmin(timeseries)] < tau_timeseries.max():
        return tau_timeseries, timeseries
    else:
        tau_max += step
        return hb_lifetime(tau_max, step)


def obtain_hbonds_data(syst: str):
    '''
    calculate hbonds using MDAnalysis
    '''

    print(f'calculating hbonds for {syst}...')
    u = mda.Universe(str(PATH / syst / 'md' / 'md.tpr'),
                     str(PATH / syst / 'md' / 'last_ns.xtc'))

    hbonds = HBA(universe=u,
                 between=['resname CHL', 'resname DMPC'])
#                  donors_sel='name O3')
    hbonds.run()

    print(f'calculating hbonds lifetimes for {syst}...')
    tau_timeseries, timeseries = hb_lifetime()
    np.save(PATH / 'notebooks' / 'contacts' / f'{syst}_chl_dmpc_199_200_1_hb.npy', hbonds.results.hbonds)
    np.save(PATH / 'notebooks' / 'contacts' / f'{syst}_chl_dmpc_199_200_1_hblt.npy', hb_lifetime())
    print('done.')

# %%


tau_timeseries, timeseries = hb_lifetime()

plt.plot(tau_timeseries, timeseries)
plt.xlim(0)
plt.ylim(0)
plt.ylabel(r'$C(\tau)$')
plt.xlabel(r'$\tau$, ps')

# %%
u.select_atoms('SOL')

# %%

np.unique(u.atoms.resnames)


# %%
import nglview

# %%
chols = u.select_atoms('resname CHL')
v = nglview.show_mdanalysis(chols)



# %% last ns trajectories
import numpy as np
from modules.general import flatten
from modules.constants import PATH, EXPERIMENTS

systems = list(set(flatten([(i, i + '_chol10', i + '_chol30', i + '_chol50')
                        for i in flatten(EXPERIMENTS.values())])))
len(systems)
# %%
ok_systems = []
for syst in systems:
    data = np.load(PATH / syst / 'mhp' / '1_data.nmp')['data']
    if np.max(data) < 100:
        ok_systems.append(syst)
len(ok_systems)

# %%
data = np.load(PATH / 'dopc' / 'mhp' / '1_data.nmp')['data']
np.any(np.unique(data)) > 100
len(np.unique(data))


# %%
set(systems) - set(ok_systems)
