'''
simple script to obtain chol mhp data
'''

import MDAnalysis as mda
import numpy as np
import pandas as pd
from modules.constants import EXPERIMENTS, PATH
from modules.general import flatten, multiproc
from modules.traj import System, TrajectorySlice


def get_detailed_mhp_data_chol(progress: dict, task_id: int,
                               trj: TrajectorySlice) -> None:
    '''
    calculate mhp fractions by chol in single system and save them as csv
    '''
    # pylint: disable = unsubscriptable-object
    # Universe.atoms is subscriptable

    to_df = {'index': [],
             'system': [],
             'CHL amount, %': [],
             'timepoint': [],
             'mol_name': [],
             'mol_ind': [],
             'at_name': [],
             'at_ind': [],
             'mhp': []}

    prefix = PATH / trj.system.name / 'mhp_200.0-201.0-1'
    at_info = np.load(prefix / '1_pa.nmp')['data']
    mapp = np.load(prefix / '1_data.nmp')['data']

    trj.generate_slice_with_gmx()

    u = mda.Universe(
        str(PATH / trj.system.name / 'md' / trj.system.tpr),
        str(PATH / trj.system.name / 'md' / trj.system.xtc),
        refresh_offsets=True)

    for ts in range(0, mapp.shape[0]):
        map_ts = mapp[ts].flatten()
        at_ts = at_info[ts].flatten()
        for at, mhp in zip(at_ts, map_ts):
            if u.atoms[at].resname == 'CHL':
                to_df['index'].append(trj.system.name)
                to_df['system'].append(trj.system.name.split('_chol', 1)[0])
                to_df['CHL amount, %'].append(
                    trj.system.name.split('_chol', 1)[1])
                to_df['timepoint'].append(trj.b * 1000 + ts)
                to_df['mol_name'].append(u.atoms[at].resname)
                to_df['mol_ind'].append(u.atoms[at].resid)
                to_df['at_name'].append(u.atoms[at].name)
                to_df['at_ind'].append(at)
                to_df['mhp'].append(mhp)
        progress[task_id] = {'progress': ts + 1, 'total': mapp.shape[0]}

    df = pd.DataFrame.from_dict(to_df)
    df.to_csv(PATH / 'tmp' / f'{trj.system.name}_mhp.csv', index=False)


def main():
    '''
    simple script to obtain chol mhp data
    '''
    systems = flatten([(i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in flatten(EXPERIMENTS.values())])
    systems = list(dict.fromkeys(systems))
    trj_slices = [TrajectorySlice(System(
        PATH, s, 'pbcmol_201.xtc', '201_ns.tpr'),
        200.0, 201.0, 1) for s in systems]
    multiproc(get_detailed_mhp_data_chol, trj_slices,
              show_progress='multiple', n_workers=21)
    dfs = []
    for trj in trj_slices:
        df = pd.read_csv(PATH / 'tmp' / f'{trj.system.name}_mhp.csv')
        dfs.append(df)
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_csv(PATH / 'notebooks' / 'mhpmaps' / 'info_mhp_atoms_'
                    f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}'
                    '_chol.csv', index=False)
    mhp_df_mol = final_df.groupby(
        ['index', 'system', 'CHL amount, %', 'timepoint', 'mol_ind'],
        as_index=False)['mhp'].mean()
    mhp_df_mol.to_csv(
        PATH / 'notebooks' / 'mhpmaps' / 'info_mhp_atoms_'
        f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}'
        '_chol_per_mol.csv', index=False)


# %%
if __name__ == '__main__':
    main()


# %%
# systems = flatten([(i + '_chol10', i + '_chol30', i + '_chol50')
#                    for i in flatten(EXPERIMENTS.values())])
# systems = list(dict.fromkeys(systems))
# trj_slices = [TrajectorySlice(System(
#     PATH, s, 'pbcmol_201.xtc', '201_ns.tpr'),
#     200.0, 201.0, 1) for s in systems]
# dfs = []
# for trj in trj_slices:
#     df = pd.read_csv(PATH / 'tmp' / f'{trj.system.name}_mhp.csv')
#     dfs.append(df)
# final_df = pd.concat(dfs, ignore_index=True)
# final_df.to_csv(PATH / 'notebooks' / 'mhpmaps' / 'info_mhp_atoms_'
#                 f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}'
#                 '_chol.csv')
