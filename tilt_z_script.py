#!/usr/bin/python3

import vars_func as vf
import MDAnalysis as mda
import pandas as pd
import numpy as np
import os
import argparse

def obtain_part_of_trajectory(syst, b, e, dt):
    '''create a slice of trajectory and put it to vf.path/tmp/
        beginning and end in ns, dt in ps'''

    stream = os.popen(f'{vf.gmxl} && '
                      'echo 0 | '
                      f'gmx trjconv -f {vf.path}/{syst}/md/pbcmol.xtc '
                      f'-s {vf.path}/{syst}/md/md.tpr '
                      f'-b {b}000 -e {e}000 -dt {dt} '
                      f'-o {vf.path}/tmp/{syst}-{b}-{e}-{dt}.xtc')
    print(stream.read())


def obtain_distances_from_bilayer_center(residues, ts):
    com_positions = np.array([mol.atoms.center_of_mass() for mol in residues])
    z = com_positions[:, 2]
    bilayer_center = ts.dimensions[2] / 2
    # return abs(z - bilayer_center)
    return z - bilayer_center


def obtain_angles_to_bilayer_plane(residues):
    C3 = residues.atoms.select_atoms('name C3')
    C17 = residues.atoms.select_atoms('name C17')
    cholesterol_vectors = C3.positions - C17.positions
    X, Y, Z = [cholesterol_vectors[:, i] for i in range(3)]
    cos_alpha = Z / np.sqrt(X**2 + Y**2 + Z**2)
    alpha = np.rad2deg(np.arccos(cos_alpha))
    # alpha = np.array([i if i < 90 else 180 - i for i in alpha])
    return alpha


def calculate_distances_and_angles(syst, b, e, dt):
    print('obtaining trajectory slice...')
    obtain_part_of_trajectory(syst, b, e, dt)

    print('✅')
    print('loading universe...')

    u = mda.Universe(f'{vf.path}/{syst}/md/md.tpr',
                     f'{vf.path}/tmp/{syst}-{b}-{e}-{dt}.xtc',
                     refresh_offsets=True)

    print('✅')

    cholesterol_residues = u.residues[u.residues.resnames == 'CHL']
    angles_all = []
    distances_all = []
    timestamps = []

    print('calculating angles and deepening...')
    for ts in u.trajectory:
        timestep_in_ns = int(ts.time) / 1000
        distances_from_bilayer_center = obtain_distances_from_bilayer_center(
            cholesterol_residues, ts)
        angles = obtain_angles_to_bilayer_plane(cholesterol_residues)
        distances_all.extend(distances_from_bilayer_center)
        angles_all.extend(angles)
        timestamps.extend(timestep_in_ns for i in cholesterol_residues)
    print('✅')
    print('saving data...')
    df = pd.DataFrame({'time': timestamps, 'angle': angles_all,
                      'distance': distances_all})

    df.to_csv(f'{vf.path}/notebooks/chol_tilt/angle_z-{syst}-{b}-{e}-{dt}.csv',
              index=False)
    print('✅')
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Calculate distances from bilayer center and angles '
                    'between C3-C17 vector to bilayer plane normal for CHL molecules')
    parser.add_argument('-s', '--systems_list', nargs='*', default=vf.chol_systs, help='list of system names in chol_impact/ for which to perform calculations')
    parser.add_argument('-b', default=100, help='beginning of trajectory, ns')
    parser.add_argument('-e', default=200, help='end of trajectory, ns')
    parser.add_argument('-dt', default=100, help='timestep for trajectory, ps')
    args = parser.parse_args()
    for i in args.systems_list:
        print(f'\n{i}')
        try:
            calculate_distances_and_angles(i, args.b, args.e, args.dt)
        except:
            vf.send_email('distances angle', f'error calculating for {i}')
    vf.send_email('distances angles', 'calculated')


if __name__ == '__main__':
    main()
