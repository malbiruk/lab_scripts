#!/usr/bin/python3

'''
This script can automatically create system with desired
lipids mixture and system name (same as folder name);
also it can run minimization energy relaxation steps and md
'''


import os
import argparse
import shutil
import re
import subprocess
from modules.general import realtime_output


def gmx_version():
    '''
    prints version of gromacs being used
    '''
    return os.popen('ls -t /usr/local/gromacs*/bin/GMXRC | head -n 1').read()


def check_input(composition: str, steps: list):
    '''
    throws errors when input data is in wrong format
    '''
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


def update_temperature_in_mdps(chol_impact: str, system_name: str, temperature: int):
    '''
    this function updates temperature in *.mdp files which were copied from default
    '''
    system_directory = chol_impact + system_name + '/'
    mdps_list = os.popen(
        f'ls {system_directory}*.mdp').read().strip().split('\n')
    for mdp in mdps_list:
        with open(mdp, 'r', encoding='utf-8') as f:
            mdp_contents = f.read()
            mdp_contents_new = mdp_contents.replace('310', str(temperature))
        with open(mdp, 'w', encoding='utf-8') as f:
            f.write(mdp_contents_new)


def create_system(chol_impact: str, impulse: str, system_name: str, composition: str):
    '''
    creates folder with system name and creates a system there
    with a specified composition
    '''
    def update_lipid_mixture(path_to_wat_lip: str, composition: str):
        '''
        updates wat_lip.args file writing there specified composition
        of system being created
        '''
        with open(path_to_wat_lip, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        other_parameters = ''.join(lines[1:])
        new_mixture = ','.join(composition)
        new_line = f'LIP_MIXT="{new_mixture}"\n'
        lipid_args_file = ''.join([new_line, other_parameters])
        with open(path_to_wat_lip, 'w', encoding='utf-8') as f:
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


def relaxation(chol_impact: str, gmxl: str, system_name: str, mdrun_args_dict: dict):
    '''
    perform relaxation steps for specified system
    '''
    system_directory = chol_impact + system_name + '/'
    os.chdir(system_directory)
    dirs = ('em', 'pr1', 'pr2', 'pr3', 'pr4')
    for i in dirs:
        os.makedirs(i, exist_ok=True)
    print(gmx_version())
    cmd = gmxl + ' && gmx grompp -f em.mdp -c indata/system.gro ' \
        '-r indata/system.gro -n indata/grps.ndx -p indata/system.top -o em/em.tpr -maxwarn 2'
    realtime_output(cmd)

    os.chdir(system_directory + 'em')
    if mdrun_args_dict is None:
        cmd = gmxl + '&& gmx mdrun -deffnm em -v'
    else:
        mdrun_args_str = ' '.join(
            ['-' + str(k) + ' ' + str(v) for k, v in mdrun_args_dict.items()])
        cmd = gmxl + '&& gmx mdrun -deffnm em -v ' + mdrun_args_str
    realtime_output(cmd)

    os.chdir(system_directory)

    for c, i in enumerate(dirs[1:], start=1):
        os.chdir(system_directory + i)
        if mdrun_args_dict is None:
            cmd = gmxl + ' && gmx grompp -n ../indata/grps.ndx ' \
                f'-f ../{i}.mdp -c ../{dirs[c-1]}/{dirs[c-1]}.gro ' \
                f'-r ../{dirs[c-1]}/{dirs[c-1]}.gro ' \
                f'-p ../indata/system.top -o {i} -maxwarn 2 && ' \
                f'gmx mdrun -deffnm {i} -v'
        else:
            mdrun_args_str = ' '.join(
                ['-' + str(k) + ' ' + str(v) for k, v in mdrun_args_dict.items()])
            cmd = gmxl + ' && gmx grompp -n ../indata/grps.ndx ' \
                f'-f ../{i}.mdp -c ../{dirs[c-1]}/{dirs[c-1]}.gro ' \
                f'-r ../{dirs[c-1]}/{dirs[c-1]}.gro ' \
                f'-p ../indata/system.top -o {i} -maxwarn 2 && ' \
                f'gmx mdrun -deffnm {i} -v' + mdrun_args_str

        realtime_output(cmd)
        if not os.path.exists(system_directory + f'{dirs[c-1]}/{dirs[c-1]}.gro'):
            os.popen(
                f'python3 /home/klim/scripts/email_notify.py {system_name} rlx error').read()
            raise ChildProcessError('something went wrong')

    if os.path.exists(system_directory + 'pr4/pr4.gro'):
        os.popen(
            f'python3 /home/klim/scripts/email_notify.py {system_name} rlx success').read()


def perform_md(chol_impact, gmxl, system_name, time, mdrun_args_dict: dict):
    '''
    perform md for specified system
    '''
    system_directory = chol_impact + system_name + '/'
    os.chdir(system_directory)
    if time is not None:
        with open(system_directory + 'md.mdp', 'r', encoding='utf-8') as inp:
            mdp_contents = inp.readlines()
            nsteps_line = mdp_contents[6].replace(
                '100000000', str(time * 500000))
        with open(system_directory + 'md.mdp', 'w', encoding='utf-8') as out:
            out.write(
                ''.join(mdp_contents[:6] + [nsteps_line] + mdp_contents[7:]))

    if os.path.exists(system_directory + 'pr4/pr4.gro'):
        os.makedirs('md', exist_ok=True)
        print(gmx_version())
        cmd = gmxl + '&& gmx grompp -f md.mdp -c pr4/pr4.gro ' \
            '-r pr4/pr4.gro -n indata/grps.ndx -p indata/system.top -o md/md.tpr ' \
            '-maxwarn 2'
        realtime_output(cmd)

        os.chdir(system_directory + 'md')
        if mdrun_args_dict is None:
            cmd = gmxl + '&& gmx mdrun -deffnm md -v'
        else:
            mdrun_args_str = ' '.join(
                ['-' + str(k) + ' ' + str(v) for k, v in mdrun_args_dict.items()])
            cmd = gmxl + '&& gmx mdrun -deffnm md -v ' + mdrun_args_str
        realtime_output(cmd)

    if os.path.exists(system_directory + 'md/md.gro'):
        os.popen(
            f'python3 /home/klim/scripts/email_notify.py {system_name} md success').read()
    else:
        os.popen(
            f'python3 /home/klim/scripts/email_notify.py {system_name} md error').read()
        raise ChildProcessError('something went wrong')


def main():
    '''
    parse arguments and execute all specified steps for system:
    system creation, relaxation, molecular dynamics
    '''
    chol_impact = '/home/klim/Documents/chol_impact/'
    impulse = '/nfs/belka2/soft/impulse/dev/inst/runtask.py'
    gmxl = '. /usr/local/gromacs-2020.6/bin/GMXRC'

    parser = argparse.ArgumentParser(
        description='Create system, start relaxation and md for "system_name"')
    parser.add_argument('-s', '--system_name', type=str, required=True,
                        help='system name (it is also folder name)')
    parser.add_argument('-c', '--composition', nargs='+',
                        help='system composition (example: POPC:0.7 CHOL:0.3)')
    parser.add_argument('-T', '--temperature', type=int, default=310,
                        help='relaxation and md will be executed under this temperature '
                        '(default: T=310)')
    parser.add_argument('-t', '--time', type=int, default=200,
                        help='duration of md in ns (default: t=200)')
    parser.add_argument('--steps', nargs='*',
                        help='execute the following steps (available values: '
                        'create_system, relaxation, md)')
    parser.add_argument('--mdrun_args', nargs='*',
                        help='in relaxation and md steps run mdrun with following '
                        'parameters (format: parameter1=value1 parameter2=value2)')

    args = parser.parse_args()

    check_input(args.composition, args.steps)
    os.popen(
        f'python3 /home/klim/scripts/email_notify.py {args.system_name} rlx error').read()

    if args.mdrun_args is not None:
        mdrun_args_dict = {i.split('=')[0]: i.split('=')[1]
                           for i in args.mdrun_args}
    else:
        mdrun_args_dict = None

    if args.steps is None:
        create_system(chol_impact, impulse, args.system_name, args.composition)
        if args.temperature != 310:
            update_temperature_in_mdps(
                chol_impact, args.system_name, args.temperature)
        relaxation(chol_impact, gmxl, args.system_name, mdrun_args_dict)
        perform_md(chol_impact, gmxl, args.system_name, args.time, mdrun_args_dict)

    if 'create_system' in args.steps:
        create_system(chol_impact, impulse, args.system_name, args.composition)

    if args.temperature != 310:
        update_temperature_in_mdps(
            chol_impact, args.system_name, args.temperature)

    if 'relaxation' in args.steps:
        relaxation(chol_impact, gmxl, args.system_name, mdrun_args_dict)

    if 'md' in args.steps:
        perform_md(chol_impact, gmxl, args.system_name,
                   args.time, mdrun_args_dict)


if __name__ == '__main__':
    main()
