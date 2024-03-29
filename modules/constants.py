'''
this module contains often used constants
'''
from pathlib import Path

PATH = Path('/home/klim/Documents/chol_impact')
EXPERIMENTS = {
    'chain length': ('dmpc', 'dppc_325', 'dspc'),
    'chain saturation': ('dppc_325', 'popc', 'dopc'),
    'head polarity': ('dopc', 'dopc_dops50', 'dops'),
}
TO_RUS = {'dmpc': 'ДМФХ', 'dppc_325': 'ДПФХ', 'dspc': 'ДСФХ', 'popc': 'ПОФХ',
          'dopc': 'ДОФХ', 'dops': 'ДОФС', 'dopc_dops50': 'ДОФХ/ДОФС',
          'chain length': 'Длина ацильных цепей',
          'chain saturation': 'Насыщенность ацильных цепей',
          'head polarity': 'Полярность "головок"',
          'thickness, nm': 'Толщина, нм',
          'distance, nm': 'Расстояние, нм',
          'area per lipid, nm²': 'Средняяя площадь на липид, нм²',
          'scd': 'Scd', 'peak width, nm':
          'Ширина пиков профилей плотности ХС, нм',
          'α, °': 'α, °',
          '% of horizontal component': '% горизонтальной компоненты',
          '% of vertical component': '% вертикальной компоненты',
          '% of area': '% площади поверхности',
          'Distance to bilayer center, Å': 'Расстояние до центра бислоя, Å',
          'distance to bilayer center, Å': 'Расстояние до центра бислоя, Å',
          'CHL-PL / LIP-LIP, %': 'ХС-ФЛ / ЛИП-ЛИП, %',
          'system': 'Система',
          'CHL amount, %': 'Концентрация ХС, %',
          'water': 'вода',
          'phosphates': 'фосфаты',
          'acyl_chains': 'ацильные цепи',
          'chols_o': 'кислород ХС',
          'chols': 'ХС',
          'CHL': 'ХС',
          'PL': 'ФЛ',
          'SOL': 'Вода',
          'No hbonds': 'Нет',
          'n contacts per CHL molecule': 'Число контактов на молекулу ХС',
          'surface': 'Доступ молекулы к поверхности бислоя',
          'yes': 'Да',
          'no': 'Нет',
          'vertical': '"вертикальная"',
          'horizontal': '"горизонтальная"',
          'tilt component': 'Компонента угла наклона ХС α',
          'hydrophobic': 'гидрофобные',
          'hydrophilic': 'гидрофильные',
          'neutral': 'нейтральные',
          'cluster size, Å': 'Размер кластера, Å',
          'cluster lifetime, ps': 'Время жизни кластера, пс',
          }
