#!/usr/bin/python3

import os
import argparse
import shutil
import re
import subprocess


def realtime_output(cmd):
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    while True:
        realtime_output = process.stdout.readline()
        if realtime_output == '' and process.poll() is not None:
            break
        if realtime_output:
            print(realtime_output.strip(), flush=True)

    # process.wait()
    # if process.returncode != 0:
    #     raise ChildProcessError('something went wrong')


def gmx_version():
    return os.popen('ls -t /usr/local/gromacs*/bin/GMXRC | head -n 1').read()


def check_input(path, system_name, composition, steps):
    if composition is not None:
        for i in composition:
            if re.match(r'[a-zA-Z]+?:[0-9]*?\.?[0-9]+?', i) is None:
                raise ValueError('Wrong composition format')

    if steps is not None:
        for i in steps:
            if i not in ['create_system', 'relaxation', 'md']:
                raise ValueError(
                    'Available values for steps: create_system, relaxation, md')
        if 'create_system' in steps and composition is None:
            raise ValueError(
                'Composition should be specified while creating system')

    if steps is None and composition is None:
        raise ValueError(
            'Composition should be specified while creating system')


def update_temperature_in_mdps(chol_impact, system_name, temperature):
    system_directory = chol_impact + system_name + '/'
    mdps_list = os.popen(
        f'ls {system_directory}*.mdp').read().strip().split('\n')
    for mdp in mdps_list:
        with open(mdp, 'r') as f:
            mdp_contents = f.read()
            mdp_contents_new = mdp_contents.replace('310', str(temperature))
        with open(mdp, 'w') as f:
            f.write(mdp_contents_new)


def create_system(chol_impact, impulse, system_name, composition):
    def update_lipid_mixture(path_to_wat_lip, composition):
        with open(path_to_wat_lip, 'r') as f:
            lines = f.readlines()
        old_mixture_line = lines[0]
        other_parameters = ''.join(lines[1:])
        new_mixture = ','.join(composition)
        new_line = f'LIP_MIXT="{new_mixture}"\n'
        lipid_args_file = ''.join([new_line, other_parameters])
        with open(path_to_wat_lip, 'w') as f:
            f.write(lipid_args_file)

    if system_name in next(os.walk(chol_impact))[1]:
        if input(f'Directory {system_name} already exists. '
                 'Do you want to continue and overwrite its contents (y/n)? ') != 'y':
            print('Exiting...')
            raise SystemExit(0)

    system_directory = chol_impact + system_name + '/'
    os.makedirs(system_directory, exist_ok=True)
    print(f'created directory {system_directory}')
    os.chdir(system_directory)
    for i in next(os.walk(chol_impact + 'mdps/'))[2]:
        shutil.copy(chol_impact + 'mdps/' + i, system_directory)
    update_lipid_mixture(system_directory + 'wat_lip.arg', composition)
    print('creating system...')
    realtime_output(
        f'{impulse} -f {system_directory}wat_lip.arg -t {system_directory}wat_lip.mk')


def relaxation(chol_impact, gmxl, system_name):
    system_directory = chol_impact + system_name + '/'
    os.chdir(system_directory)
    dirs = ('em', 'pr1', 'pr2', 'pr3', 'pr4')
    for i in dirs:
        os.makedirs(i, exist_ok=True)
    print(gmx_version())
    cmd = gmxl + ' && gmx grompp -f em.mdp -c indata/system.gro ' \
        '-r indata/system.gro -n indata/grps.ndx -p indata/system.top -o em/em.tpr'
    realtime_output(cmd)

    os.chdir(system_directory + 'em')
    cmd = gmxl + ' && gmx mdrun -deffnm em'
    realtime_output(cmd)

    os.chdir(system_directory)

    for c, i in enumerate(dirs[1:], start=1):
        os.chdir(system_directory + i)
        cmd = gmxl + ' && gmx grompp -n ../indata/grps.ndx ' \
            f'-f ../{i}.mdp -c ../{dirs[c-1]}/{dirs[c-1]}.gro ' \
            f'-r ../{dirs[c-1]}/{dirs[c-1]}.gro ' \
            f'-p ../indata/system.top -o {i} -maxwarn 1 && ' \
            f'gmx mdrun -deffnm {i} -v'
        realtime_output(cmd)
        if not os.path.exists(system_directory + f'{dirs[c-1]}/{dirs[c-1]}.gro'):
            os.popen(
                f'python3 /home/klim/scripts/email_notify.py {system_name} rlx error').read()
            raise ChildProcessError('something went wrong')

    if os.path.exists(system_directory + 'pr4/pr4.gro'):
        os.popen(
            f'python3 /home/klim/scripts/email_notify.py {system_name} rlx success').read()


def md(chol_impact, gmxl, system_name, time):
    system_directory = chol_impact + system_name + '/'
    os.chdir(system_directory)
    if time is not None:
        with open(system_directory + 'md.mdp', 'r') as inp:
            mdp_contents = inp.readlines()
            nsteps_line = mdp_contents[6].replace(
                '100000000', str(time * 500000))
        with open(system_directory + 'md.mdp', 'w') as out:
            out.write(
                ''.join(mdp_contents[:6] + [nsteps_line] + mdp_contents[7:]))

    if os.path.exists(system_directory + 'pr4/pr4.gro'):
        os.makedirs('md', exist_ok=True)
        print(gmx_version())
        cmd = gmxl + '&& gmx grompp -f md.mdp -c pr4/pr4.gro ' \
            '-r pr4/pr4.gro -n indata/grps.ndx -p indata/system.top -o md/md.tpr'
        realtime_output(cmd)

        os.chdir(system_directory + 'md')
        cmd = gmxl + '&& gmx mdrun -deffnm md -v'
        realtime_output(cmd)

    if os.path.exists(system_directory + 'md/md.gro'):
        os.popen(
            f'python3 /home/klim/scripts/email_notify.py {system_name} md success').read()
    else:
        os.popen(
            f'python3 /home/klim/scripts/email_notify.py {system_name} md error').read()
        raise ChildProcessError('something went wrong')


def main():
    chol_impact = '/home/klim/Documents/chol_impact/'
    impulse = '/nfs/belka2/soft/impulse/dev/inst/runtask.py'
    gmxl = '. `ls -t /usr/local/gromacs*/bin/GMXRC | head -n 1 `'

    parser = argparse.ArgumentParser(
        description='Create system, start relaxation and md for "system_name"')
    parser.add_argument('-s', '--system_name', type=str, required=True,
                        help='system name (it is also folder name)')
    parser.add_argument('-c', '--composition', nargs='+',
                        help='system composition (example: POPC:0.7 CHOL:0.3)')
    parser.add_argument('-T', '--temperature', type=int, default=310,
                        help='relaxation and md will be executed under this temperature (default: T=310)')
    parser.add_argument('-t', '--time', type=int, default=200,
                        help='duration of md in ns (default: t=200)')
    parser.add_argument('--steps', nargs='*',
                        help='execute the following steps (available values: create_system, relaxation, md)')
    args = parser.parse_args()

    check_input(chol_impact, args.system_name, args.composition, args.steps)
    os.popen(
        f'python3 /home/klim/scripts/email_notify.py {args.system_name} rlx error').read()

    if args.steps is None:
        create_system(chol_impact, impulse, args.system_name, args.composition)
        if args.temperature != 310:
            update_temperature_in_mdps(
                chol_impact, args.system_name, args.temperature)
        relaxation(chol_impact, gmxl, args.system_name)
        md(chol_impact, gmxl, args.system_name, args.time)

    if 'create_system' in args.steps:
        create_system(chol_impact, impulse, args.system_name, args.composition)

    if args.temperature != 310:
        update_temperature_in_mdps(
            chol_impact, args.system_name, args.temperature)

    if 'relaxation' in args.steps:
        relaxation(chol_impact, gmxl, args.system_name)

    if 'md' in args.steps:
        md(chol_impact, gmxl, args.system_name, args.time)


if __name__ == '__main__':
    main()
