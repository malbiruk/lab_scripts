'''
short simple script to obtain CHL COMs distribution along Z-plane
and plot them
'''

import logging
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from integral_parameters_script import plot_violins
from MDAnalysis.analysis import leaflet
from modules.constants import EXPERIMENTS, PATH
from modules.general import flatten, initialize_logging, multiproc
from modules.traj import System, TrajectorySlice

app = typer.Typer(rich_markup_mode='rich', add_completion=False)


class LeafletFinderError(Exception):
    '''
    error associated with bilayer leaflet identification
    '''


def get_chl_coms_z_coords(progress: dict, task_id: int,
                          trj: TrajectorySlice) -> None:
    '''
    determine and save positions of COMs of CHL in current
    TrajectorySlice as well as bilayer center z coordinate
    '''
    # pylint: disable = too-many-locals
    fname = Path(f'{trj.system.dir}/md/chl_coms_pos_'
                 f'{trj.b}-{trj.e}-{trj.dt}.csv')

    if fname.is_file():
        return

    logging.debug('%s started', trj.system.name)

    trj.generate_slice_with_gmx()
    u = mda.Universe(f'{trj.system.dir}/md/{trj.system.tpr}',
                     f'{trj.system.dir}/md/'
                     f'pbcmol_{trj.b}-{trj.e}-{trj.dt}.xtc')

    chols = u.select_atoms('resname CHL').residues
    num_chols = len(chols)
    num_frames = len(u.trajectory)

    to_df = {
        'timestep': np.empty(num_chols * num_frames),
        'CHL index': np.empty(num_chols * num_frames),
        'x': np.empty(num_chols * num_frames),
        'y': np.empty(num_chols * num_frames),
        'z': np.empty(num_chols * num_frames),
        'bilayer center z': np.empty(num_chols * num_frames)
    }

    for i, ts in enumerate(u.trajectory):
        cutoff, _ = leaflet.optimize_cutoff(u, 'name P* or name O3',
                                            dmin=7, dmax=17)
        leaflet_ = leaflet.LeafletFinder(u, 'name P* or name O3',
                                         pbc=True, cutoff=cutoff)
        if len(leaflet_.groups()) != 2:
            raise LeafletFinderError(f'{len(leaflet_.groups())} leaflets found')
        leaflet_0 = leaflet_.group(0)
        leaflet_1 = leaflet_.group(1)
        bilayer_center = 0.5 * (leaflet_1.centroid() + leaflet_0.centroid())

        chols_coms = {i.resid: i.atoms.center_of_mass() for i in chols}
        chols_coms_coords = np.array(list(chols_coms.values()))

        start_idx = i * num_chols
        end_idx = start_idx + num_chols

        to_df['timestep'][start_idx:end_idx] = ts.time
        to_df['CHL index'][start_idx:end_idx] = list(chols_coms.keys())
        to_df['x'][start_idx:end_idx] = chols_coms_coords[:, 0]
        to_df['y'][start_idx:end_idx] = chols_coms_coords[:, 1]
        to_df['z'][start_idx:end_idx] = chols_coms_coords[:, 2]
        to_df['bilayer center z'][start_idx:end_idx] = bilayer_center[2]

        progress[task_id] = {'progress': i + 1, 'total': num_frames}

    df = pd.DataFrame.from_dict(to_df)
    df.to_csv(fname, index=False)
    logging.info('COMs positions of CHL in %s retrieved', trj.system.name)


def collect_all_chl_coms_in_one_df(trj_slices: list) -> None:
    '''
    concat all f'{trj.system.dir}/md/chl_coms_pos_'
                 f'{trj.b}-{trj.e}-{trj.dt}.csv' files to single df
    also adding names of systems
    '''
    fname = (PATH / 'notebooks' / 'integral_parameters' /
             'chols_coms_z_'
             f'{trj_slices[0].b}-{trj_slices[0].e}-{trj_slices[0].dt}.csv')

    if fname.is_file():
        return

    dfs = []
    for trj in trj_slices:
        fname = Path(f'{trj.system.dir}/md/chl_coms_pos_'
                     f'{trj.b}-{trj.e}-{trj.dt}.csv')
        df = pd.read_csv(fname)
        c = {'index': trj.system.name,
             'system': trj.system.name.split('_chol', 1)[0],
             'CHL amount, %': trj.system.name.split('_chol', 1)[1]}
        df = df.assign(**c)
        columns_to_move = ['index', 'system', 'CHL amount, %']
        new_columns = columns_to_move + [col for col in df.columns
                                         if col not in columns_to_move]
        df = df[new_columns]
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all['Distance to bilayer center, Å'] = (
        df_all['z'] - df_all['bilayer center z']).abs()
    df_all.to_csv(
        fname,
        index=False)


@ app.command()
def main(trj: str = typer.Option(
        'pbcmol_201.xtc', '-trj', help='name of trajectory files',
        rich_help_panel='Trajectory parameters'),
        tpr: str = typer.Option(
        '201_ns.tpr', '-tpr', help='name of topology files',
        rich_help_panel='Trajectory parameters'),
        b: float = typer.Option(
        200, '-b', help='beginning of trajectories (in ns)',
        rich_help_panel='Trajectory parameters'),
        e: float = typer.Option(
        201, '-e', help='end of trajectories (in ns)',
        rich_help_panel='Trajectory parameters'),
        dt: int = typer.Option(
        1, '-dt', help='timestep of trajectories (in ps)',
        rich_help_panel='Trajectory parameters'),
        n_workers: int = typer.Option(
        21, help='n of processes to start for each task',
        rich_help_panel='Script config'),
        verbose: bool = typer.Option(
        False, '--verbose', '-v', help='print debug log',
        rich_help_panel='Script config'),
        messages: bool = typer.Option(
        True, help='send updates info in telegram',
        rich_help_panel='Script config')):
    '''
    short simple script to obtain CHL COMs distribution along Z-plane
    and plot them
    '''
    # pylint: disable = too-many-arguments
    sns.set(style='ticks', context='talk', palette='muted')
    initialize_logging('get_chl_coms.log', verbose)
    systems = flatten([(i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in flatten(EXPERIMENTS.values())])
    systems = list(dict.fromkeys(systems))

    trj_slices = [TrajectorySlice(System(PATH, s, trj, tpr), b, e, dt)
                  for s in systems]

    logging.debug(trj_slices)

    # get all chl coms coords
    logging.info('getting chl coms coordinates...')
    multiproc(get_chl_coms_z_coords,
              trj_slices,
              n_workers=n_workers,
              descr='getting chl coms positions',
              show_progress='multiple',
              messages=messages)
    logging.info('collecting chl coms coordinates in one file...')
    collect_all_chl_coms_in_one_df(trj_slices)
    logging.info('plotting chl coms coordinates distributions...')
    plot_violins(PATH / 'notebooks' / 'integral_parameters' /
                 f'chols_coms_z_{trj_slices[0].b}-{trj_slices[0].e}-'
                 f'{trj_slices[0].dt}.csv',
                 y='Distance to bilayer center, Å')
    logging.info('done.')


if __name__ == '__main__':
    app()


# %%
# systems = flatten([(i + '_chol10', i + '_chol30', i + '_chol50')
#                    for i in flatten(EXPERIMENTS.values())])
# systems = list(dict.fromkeys(systems))
# trajectory_slices = [TrajectorySlice(System(
#     PATH, s, 'pbcmol_201.xtc', '201_ns.tpr'),
#     200.0, 201.0, 1) for s in systems]
# trj_slices = trajectory_slices
