'''
analysis of SOL inside bilayer: n and lt of hbonds with CHL and PL

1. obtain leaflets
2. divide each ts by grid in x_y plane (10 A)
3. in each cell obtain thickness
4. distances of water less than 0.5 of bilayer thickness to each monolayer
5. this is water of interest
'''
# pylint: disable = not-an-iterable

import logging

import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
from MDAnalysis.analysis import leaflet
from modules.constants import EXPERIMENTS, PATH
from modules.general import flatten, initialize_logging, multiproc, print_1line
from modules.traj import System, TrajectorySlice


def obtain_sol_indices_inside_bilayer(
        progress: dict, task_id: int, trj: TrajectorySlice) -> None:
    '''
    obtain water indices inside bilayer.
    for each timestep
    using grid 30 A in xy plane
    generates and saves pd df
    '''
    # pylint: disable = too-many-locals
    water_inside_bilayer = {'timestep': [],
                            'SOL index': []}

    trj.generate_slice_with_gmx()
    u = mda.Universe(f'{trj.system.dir}/md/{trj.system.tpr}',
                     f'{trj.system.dir}/md/pbcmol_'
                     f'{trj.b}-{trj.e}-{trj.dt}.xtc')

    len_of_task = len(u.trajectory)
    for it, ts in enumerate(u.trajectory):
        # find leaflets
        try:
            cutoff, n = leaflet.optimize_cutoff(
                u, 'name P*', dmin=7, dmax=17)
            print_1line(f'cutoff {cutoff} A, {n} groups')
        except IndexError:
            logging.error('IndexError in finding leaflet, skipping ts...')
        leaflet_ = leaflet.LeafletFinder(
            u, 'name P*', pbc=True, cutoff=cutoff)
        if len(leaflet_.groups()) != 2:
            logging.error('%s groups found...', len(leaflet_.groups()))
            continue
        leaflet_0 = leaflet_.group(0)
        leaflet_1 = leaflet_.group(1)

        # split by squares
        square_size = 30
        x_min = u.atoms.positions[:, 0].min()
        y_min = u.atoms.positions[:, 1].min()
        squares = {}
        for atom in u.atoms:
            x = atom.position[0]
            y = atom.position[1]
            square_x = int((x - x_min) // square_size)
            square_y = int((y - y_min) // square_size)
            square_key = (square_x, square_y)
            if square_key not in squares:
                squares[square_key] = []
            squares[square_key].append(atom)

        # determine thickness in each cell
        c = 0
        for square_key, atoms in squares.items():
            square_atoms = mda.AtomGroup(atoms)
            sq_lf_0 = square_atoms.intersection(leaflet_0)
            sq_lf_1 = square_atoms.intersection(leaflet_1)
            try:
                local_thickness = np.abs((sq_lf_0.centroid()
                                          - sq_lf_1.centroid())[2])
            except IndexError:
                c += 1
                continue

            # determine water inside bilayer
            sq_sol = square_atoms.intersection(u.select_atoms('resname SOL'))
            for water in sq_sol.residues:
                water_coords = water.atoms.center_of_mass()
                upper_distances = np.linalg.norm(
                    water_coords - sq_lf_0.positions, axis=1)
                lower_distances = np.linalg.norm(
                    water_coords - sq_lf_1.positions, axis=1)
                if (np.all(upper_distances <= local_thickness)
                        and np.all(lower_distances <= local_thickness)):
                    water_inside_bilayer['timestep'].append(ts.time)
                    water_inside_bilayer['SOL index'].append(water.resid)
        logging.debug('%s squares missed out of %s', c, len(squares))
        progress[task_id] = {'progress': it + 1, 'total': len_of_task}
    df = pd.DataFrame(water_inside_bilayer)
    df.to_csv(PATH / 'tmp' / f'{trj.system.name}_sol_ins_bilayer.csv',
              index=False)


def main():
    '''
    analysis of SOL inside bilayer: n and lt of hbonds with CHL and PL
    '''
    sns.set(style='ticks', context='talk', palette='muted')
    initialize_logging('sol_inside_bilayer.log', False)
    systems = flatten([(i + '_chol10', i + '_chol30', i + '_chol50')
                       for i in flatten(EXPERIMENTS.values())])
    systems = list(dict.fromkeys(systems))
    trj_slices = [TrajectorySlice(System(
        PATH, s, 'pbcmol_201.xtc', '201_ns.tpr'),
        200.0, 201.0, 1) for s in systems]

    multiproc(obtain_sol_indices_inside_bilayer,
              trj_slices,
              n_workers=len(trj_slices),
              show_progress='multiple',
              descr='sol ind')

    # concatenate in single table
    # obtain hbonds info
    lip_sol_rchist_full = pd.read_csv(PATH / 'notebooks' / 'contacts' /
                                      'lip_SOL_hb_hb_full_'
                                      f'{trj_slices[0].b}-{trj_slices[0].e}'
                                      f'-{trj_slices[0].dt}_'
                                      'rchist_full.csv')



# %%
if __name__ == '__main__':
    main()
