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
          'scd': 'Scd', 'peak width, nm': 'Ширина пиков профилей плотности ХС, нм',
          'α, °': 'α, °',
          '% of horizontal component': '% горизонтальной компоненты',
          '% of vertical component': '% вертикальной компоненты'
          }
