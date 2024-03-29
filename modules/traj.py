'''
this module includes classes which make it easier to operate with trajectories
'''

import os
from dataclasses import dataclass, field
from pathlib import Path, PosixPath


@dataclass(frozen=True, unsafe_hash=True)
class System:
    '''
    stores system name and path
    '''
    path: PosixPath  # directory containing system directory
    name: str
    xtc: str = 'pbcmol.xtc'  # name of trajectory file
    tpr: str = 'md.tpr'  # name of topology file
    dir: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'dir', str(self.path / self.name))

    def __repr__(self):
        return f'System({self.name})'

    def __str__(self):
        return self.name

    def get_tpr_props(self) -> str:
        '''
        obtain properties of tpr of the system
        '''
        gmxutils = '/nfs/belka2/soft/impulse/dev/inst/gmx_utils.py'
        return os.popen(f'{gmxutils} {self.dir}/md/md.tpr').read()

    def get_n_chols(self) -> str:
        '''
        obtain number of cholesterol molecules
        '''
        return [i.strip() for i in
                [i for i in self.get_tpr_props().split('\n')
                 if i.startswith('CHOL')][0].split('|')][1]

    def resnames_from_systname(self) -> list:
        '''
        obtain residue names of system (MDAnalysis format)
        '''
        no_numbers = ''.join([i for i in self.name if not i.isdigit()])
        return [i.upper() if not i == 'chol' else 'CHL'
                for i in no_numbers.split('_')]

    def pl_selector(self, n=0) -> str:
        '''
        obtain selector string of main phospholipid (impulse format)
        '''
        return f"'{self.name.split('_')[n].upper()}///'"


@dataclass(frozen=True, unsafe_hash=True)
class TrajectorySlice:
    '''
    stores beginnig (b), ending (e) timepoints (in ns)
    with dt (in ps) between them of system trajectory
    '''
    system: System
    b: int
    e: int
    dt: int

    def __repr__(self):
        return f'TrajectorySlice({self.system.name}, ' \
            f'b={self.b}, e={self.e}, dt={self.dt})'

    def generate_slice_with_gmx(self):
        '''
        create a slice of trajectory in xtc format in the system folder using
        gmx trjconv

        generated slice is here:
        {self.system.dir}/md/pbcmol_{self.b}-{self.e}-{self.dt}.xtc
        '''
        if (Path(self.system.dir) / 'md' /
                f'pbcmol_{self.b}-{self.e}-{self.dt}.xtc').is_file():
            return
            # print(str(Path(self.system.dir) / 'md' /
            #           f'pbcmol_{self.b}-{self.e}-{self.dt}.xtc') +
            #       ' already exists')
            # else:
        cmd = ['source /usr/local/gromacs-2021.5/bin/GMXRC && ',
               f'echo 0 | gmx trjconv -s {self.system.dir}/md/{self.system.tpr}',
               f'-f {self.system.dir}/md/{self.system.xtc}',
               f'-b {self.b * 1000} -e {self.e * 1000} -dt {self.dt}',
               f'-o {self.system.dir}/md/pbcmol_{self.b}-{self.e}-{self.dt}.xtc']
        os.popen(' '.join(cmd)).read()
            # print('done ✅\n')
