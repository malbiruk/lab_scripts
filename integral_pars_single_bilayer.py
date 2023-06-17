'''
calculate and create single figure with thickness, area per lipid and Scd
for POPC with chl amounts: 0, 10, 30, 50, 70% CHL
'''

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from integral_parameters_script import (calculate_area_per_lipid,
                                        calculate_scd, calculate_thickness,
                                        lists_of_values_to_df, scd_summary)
from modules.constants import PATH
from modules.density import get_densities
from modules.general import multiproc
from modules.traj import System, TrajectorySlice


def get_data(parameter: str, trj_slices: list) -> None:
    '''
    get data for parameter
    parameter should be in [thickness, area_per_lipid, scd]
    '''
    par_to_func = {'thickness': calculate_thickness,
                   'area_per_lipid': calculate_area_per_lipid,
                   'scd': calculate_scd}

    if not parameter in par_to_func:
        raise ValueError(f'{parameter} should be '
                         'in [thickness, area_per_lipid, scd]')

    print(f'obtaining {parameter}...')
    df = lists_of_values_to_df(par_to_func[parameter], trj_slices).rename(
        columns={'data': f'{parameter}'})

    if not parameter == 'scd':
        df.to_csv(PATH / 'notebooks' / 'integral_parameters' /
                  f'popc_{parameter}.csv', index=False)
    else:
        scd_summary(trj_slices).to_csv(
            PATH / 'notebooks' / 'integral_parameters' /
            f'popc_{parameter}.csv', index=False)
    print('done.')


def create_plot():
    '''
    create single figure with thickness, area per lipid and Scd
    for POPC with chl amounts: 0, 10, 30, 50, 70% CHL
    '''

    print('plotting figure...')
    df_with_par = {
        parameter: pd.read_csv(PATH / 'notebooks' / 'integral_parameters' /
                               f'popc_{parameter}.csv')
        for parameter in ['thickness', 'area_per_lipid', 'scd']
    }


    def dfol(
            df: pd.DataFrame,
            system: str,
            chain: str,
            chl_amount: int) -> pd.DataFrame:
        '''
        extract data for one line in plot from df
        '''
        return df[(df['system'] == system)
                  & (df['chain'] == chain)
                  & (df['CHL amount, %'] == chl_amount)]


    fig, axs = plt.subplots(1, 3, figsize=(26, 9))
    for ax, par in zip(axs, df_with_par.keys()):
        if par != 'scd':
            sns.violinplot(data=df_with_par[par],
                           x='CHL amount, %',
                           y=par,
                           ax=ax,
                           palette='RdYlGn_r',
                           )
        else:
            df = df_with_par[par]
            df['atom_n'] = df['atom'].apply(lambda x: int(x[2:]))
            scd_ms = df.drop(columns=['timepoint', 'atom']).groupby(
                ['system', 'CHL amount, %', 'chain', 'atom_n'],
                as_index=False).agg(['mean', 'std'])
            scd_ms = scd_ms.reset_index(level=1).reset_index(
                level=1).reset_index(level=1).reset_index()

            s = 'popc'
            palette = sns.color_palette('cubehelix', 5)
            for c, chl in enumerate((0, 10, 30, 50, 70)):
                for sn, ls in zip(('sn-1', 'sn-2'),
                                  ('-', '--')):
                    ax.errorbar(x=dfol(scd_ms, s, sn, chl)['atom_n'],
                                y=dfol(scd_ms, s, sn, chl)['scd']['mean'],
                                yerr=dfol(scd_ms, s, sn, chl)[
                        'scd']['std'],
                        ls=ls, color=palette[c],
                        elinewidth=1, label=f'{chl} % CHL, {sn}'
                    )
            ax.set_xlim(2)
            ax.set_xlabel('C atom number')
            ax.set_ylabel('Scd')
            ax.legend(loc='upper center', bbox_to_anchor=(1.25, 1), ncol=1)

    fig.savefig(PATH / 'notebooks' / 'integral_parameters' /
                'popc_parameters_0-70.png',
                bbox_inches='tight')

    print('done.')


def main():
    '''
    calculate and create single figure with thickness, area per lipid and Scd
    for POPC with chl amounts: 0, 10, 30, 50, 70% CHL
    '''
    sns.set(style='ticks', context='talk', palette='muted')

    trj_slices = [TrajectorySlice(
        System(PATH, syst, 'pbcmol_201.xtc', '201_ns.tpr'), 150, 200, 1000)
        for syst in ['popc'] + [f'popc_chol{i}' for i in range(10, 90, 20)]]

    initialize_logging('relief.log')

    create_plot()


# %%
if __name__ == '__main__':
    main()


# # %%
#
#
#
# # %%
