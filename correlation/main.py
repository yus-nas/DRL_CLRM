from darts.engines import value_vector, redirect_darts_output
from correlation.model_uniform import Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

redirect_darts_output('m.log')
m = Model()

if 1:
    m.init()
    m.run_python()
    m.print_timers()
    m.print_stat()

    time_data = pd.DataFrame.from_dict(m.physics.engine.time_data)
    time_data.to_pickle("darts_time_data.pkl")
    m.save_restart_data()

else:
    m.init()
    m.load_restart_data()
    time_data = pd.read_pickle("darts_time_data.pkl")

# DARTS
X = np.array(m.physics.engine.X, copy=False)
Sw = np.zeros(m.reservoir.nx*m.reservoir.ny)
for i in range (len(Sw)):
    p = X[i*2]
    zw = X[i*2 + 1]
    state = value_vector([p, zw])
    Sw[i] = m.property_container.watersat_ev.evaluate(state)

# import darts.tools.eclipse_io as eclipse_io
# e_reader = eclipse_io.EclipseIO()
# e_reader.SetFileName("ECLIPSE\dead_oil")  # Eclipse data file name
# e_reader.read_restart_times()
# e_reader.read_restart()
# eclipse_pressure = e_reader.restart[40]['fields']['PRESSURE']
# eclipse_water_sat = e_reader.restart[40]['fields']['SWAT']
# eclipse_oil_sat = 1 - eclipse_water_sat
#
# plt.figure(num=1, figsize=(12,5))
# plt.subplot(121)
# plt.pcolor(eclipse_water_sat.reshape(m.reservoir.ny, m.reservoir.nx))
# plt.colorbar()
# plt.title('Water saturation')
# plt.subplot(122)
# plt.pcolor(abs(Sw-eclipse_water_sat).reshape(m.reservoir.ny, m.reservoir.nx))
# plt.colorbar()
# plt.title('Difference')
# plt.show()

# rate comparison
eclipse_time_data = pd.DataFrame(np.genfromtxt(r"eclipse\rate.RSM", names=True))
ecl_time = eclipse_time_data['TIME']

from darts.tools.plot_darts import *

for i, w in enumerate(m.reservoir.wells):
    if i==0:
        ecl_WIR=eclipse_time_data[w.name+'WWIR']
        ax = plot_water_rate_darts(w.name, time_data, color='b')
        plt.plot(ecl_time, ecl_WIR, color = 'r')
        ax.tick_params(labelsize=14)
        ax.set_xlabel('Days', fontsize=14)
        ax.set_ylabel('Water injection rate, sm$^3$/day', fontsize=14)
    else:
        # plot oil rate
        ecl_WOR = eclipse_time_data[w.name + 'WOPR']
        ax1 = plot_oil_rate_darts(w.name, time_data, color='b')
        plt.plot(ecl_time, -ecl_WOR, color='r')
        ax1.tick_params(labelsize=14)
        ax1.set_xlabel('Days', fontsize=14)
        ax1.set_ylabel(w.name + 'Oil production rate, sm$^3$/day', fontsize=14)

plt.show()