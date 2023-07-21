from math import fabs
import pickle
import os
import numpy as np

from darts.engines import *
from darts.engines import print_build_info as engines_pbi
from darts.physics import print_build_info as physics_pbi
from darts.print_build_info import print_build_info as package_pbi


class DartsModel:
    """
    Base class with multiple functions

    """

    def __init__(self):
        """"
           Initialize DartsModel class.
        """
        # print out build information
        engines_pbi()
        physics_pbi()
        package_pbi()
        self.timer = timer_node()  # Create time_node object for time record
        self.timer.start()  # Start time record
        self.timer.node["simulation"] = timer_node()  # Create timer.node called "simulation" to record simulation time
        self.timer.node[
            "initialization"] = timer_node()  # Create timer.node called "initialization" to record initialization time
        self.timer.node["initialization"].start()  # Start recording "initialization" time

        self.params = sim_params()  # Create sim_params object to set simulation parameters

        self.timer.node["initialization"].stop()  # Stop recording "initialization" time

    def init(self):
        """
            Function to initialize the model, which includes:
                - initialize well (perforation) position
                - initialize well rate parameters
                - initialize reservoir condition
                - initialize well control settings
                - build accumulation_flux_operator_interpolator list
                - initialize engine
        """
        self.reservoir.init_wells()
        self.physics.init_wells(self.reservoir.wells)
        self.set_initial_conditions()
        self.set_boundary_conditions()
        self.set_op_list()
        self.reset()

    def reset(self):
        """
        Function to initialize the engine by calling 'init' method.
        """
        self.physics.engine.init(self.reservoir.mesh, ms_well_vector(self.reservoir.wells),
                                 op_vector(self.op_list),
                                 self.params, self.timer.node["simulation"])

    def set_initial_conditions(self):
        pass

    def set_boundary_conditions(self):
        pass

    def set_op_list(self):
        self.op_list = [self.physics.acc_flux_itor]

    def run(self, days=0):
        if days:
            runtime = days
        else:
            runtime = self.runtime
        self.physics.engine.run(runtime)

    def run_python(self, days=0, restart_dt=0, log_3d_body_path=0):
        if days:
            runtime = days
        else:
            runtime = self.runtime

        mult_dt = self.params.mult_ts
        max_dt = self.params.max_ts
        e = self.physics.engine

        # get current engine time
        t = e.t

        # same logic as in engine.run
        if fabs(t) < 1e-15:
            dt = self.params.first_ts
        elif restart_dt > 0:
            dt = restart_dt
        else:
            dt = self.params.max_ts

        # evaluate end time
        runtime += t
        ts = 0
        #
        if log_3d_body_path and self.physics.n_vars == 3:
            self.body_path_start()

        while t < runtime:
            converged = e.run_timestep(dt, t)

            if converged:
                t += dt
                ts = ts + 1
                print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d" % (
                    ts, t, dt, e.n_newton_last_dt, e.n_linear_last_dt))

                dt *= mult_dt
                if dt > max_dt:
                    dt = max_dt

                if t + dt > runtime:
                    dt = runtime - t
                    
                # check for inverse flow and watercut constraint
                if t > 0.5:               
                    self.check_inverse_flow_and_wc_lim()

                if log_3d_body_path and self.physics.n_vars == 3:
                    self.body_path_add_bodys(t)
                    nb_begin = self.reservoir.nx * self.reservoir.ny * (self.body_path_map_layer - 1) * 3
                    nb_end = self.reservoir.nx * self.reservoir.ny * (self.body_path_map_layer) * 3

                    self.save_matlab_map(self.body_path_axes[0] + '_ts_' + str(ts), e.X[nb_begin:nb_end:3])
                    self.save_matlab_map(self.body_path_axes[1] + '_ts_' + str(ts), e.X[nb_begin + 1:nb_end:3])
                    self.save_matlab_map(self.body_path_axes[2] + '_ts_' + str(ts), e.X[nb_begin + 2:nb_end:3])
            else:
                dt /= mult_dt
                print("Cut timestep to %2.3f" % dt)
                if dt < 1e-8:
                    break
        # update current engine time
        e.t = runtime

        print("TS = %d(%d), NI = %d(%d), LI = %d(%d)" % (e.stat.n_timesteps_total, e.stat.n_timesteps_wasted,
                                                         e.stat.n_newton_total, e.stat.n_newton_wasted,
                                                         e.stat.n_linear_total, e.stat.n_linear_wasted))

    def load_restart_data(self, filename='restart.pkl'):
        """
        Function to load data from previous simulation and uses them for following simulation.
        :param filename: restart_data filename
        """
        if os.path.exists(filename):
            with open(filename, "rb") as fp:
                data = pickle.load(fp)
                days, X, arr_n = data
                self.physics.engine.t = days
                self.physics.engine.X = value_vector(X)
                self.physics.engine.Xn = value_vector(X)
                self.physics.engine.op_vals_arr_n = value_vector(arr_n)

    def save_restart_data(self, filename='restart.pkl'):
        """
        Function to save the simulation data for restart usage.
        :param filename: Name of the file where restart_data stores.
        """
        t = np.copy(self.physics.engine.t)
        X = np.copy(self.physics.engine.X)
        arr_n = np.copy(self.physics.engine.op_vals_arr_n)
        data = [t, X, arr_n]
        with open(filename, "wb") as fp:
            pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)

    # overwrite key to save results over existed
    # diff_norm_normalized_tol defines tolerance for L2 norm of final solution difference , normalized by amount of blocks and variable range
    # diff_abs_max_normalized_tol defines tolerance for maximum of final solution difference, normalized by variable range
    # rel_diff_tol defines tolerance (in %) to a change in integer simulation parameters as linear and newton iterations
    def check_performance(self, overwrite=0, diff_norm_normalized_tol=1e-10, diff_abs_max_normalized_tol=1e-9,
                          rel_diff_tol=1, perf_file=''):
        """
        Function to check the performance data to make sure whether the performance has been changed
        """
        fail = 0
        data_et = self.load_performance_data(perf_file)
        if data_et and not overwrite:
            data = self.get_performance_data()
            nb = self.reservoir.mesh.n_res_blocks
            nv = self.physics.n_vars

            # Check final solution - data[0]
            # Check every variable separately
            for v in range(nv):
                sol_et = data_et['solution'][v:nb * nv:nv]
                diff = data['solution'][v:nb * nv:nv] - sol_et
                sol_range = np.max(sol_et) - np.min(sol_et)
                diff_abs = np.abs(diff)
                diff_norm = np.linalg.norm(diff)
                diff_norm_normalized = diff_norm / len(sol_et) / sol_range
                diff_abs_max_normalized = np.max(diff_abs) / sol_range
                if diff_norm_normalized > diff_norm_normalized_tol or diff_abs_max_normalized > diff_abs_max_normalized_tol:
                    fail += 1
                    print(
                        '#%d solution check failed for variable %s: Normalized difference norm  %.2E (tol %.2E), Normalized abs difference max %.2E (tol %.2E)' \
                        % (fail, self.physics.vars[v], diff_norm_normalized, diff_norm_normalized_tol,
                           diff_abs_max_normalized, diff_abs_max_normalized_tol))
            for key, value in sorted(data.items()):
                if key == 'solution' or type(value) != int:
                    continue
                reference = data_et[key]

                if reference == 0:
                    if value != 0:
                        print('#%d parameter %s is %d (was 0)' % (fail, key, value))
                        fail += 1
                else:
                    rel_diff = (value - data_et[key]) / reference * 100
                    if abs(rel_diff) > rel_diff_tol:
                        print('#%d parameter %s is %d (was %d, %+.2f%%)' % (fail, key, value, reference, rel_diff))
                        fail += 1
            if not fail:
                print('OK, \t%.2f s' % self.timer.node['simulation'].get_timer())
                return 0
            else:
                print('FAIL, \t%.2f s' % self.timer.node['simulation'].get_timer())
                return 1
        else:
            self.save_performance_data()
            print('SAVED')
            return 0

    def get_performance_data(self):
        """
        Function to get the needed performance data
        """
        perf_data = dict()
        perf_data['solution'] = np.copy(self.physics.engine.X)
        perf_data['reservoir blocks'] = self.reservoir.mesh.n_res_blocks
        perf_data['variables'] = self.physics.n_vars
        perf_data['OBL resolution'] = self.physics.n_points
        perf_data['operators'] = self.physics.n_ops
        perf_data['timesteps'] = self.physics.engine.stat.n_timesteps_total
        perf_data['wasted timesteps'] = self.physics.engine.stat.n_timesteps_wasted
        perf_data['newton iterations'] = self.physics.engine.stat.n_newton_total
        perf_data['wasted newton iterations'] = self.physics.engine.stat.n_newton_wasted
        perf_data['linear iterations'] = self.physics.engine.stat.n_linear_total
        perf_data['wasted linear iterations'] = self.physics.engine.stat.n_linear_wasted

        sim = self.timer.node['simulation']
        jac = sim.node['jacobian assembly']
        perf_data['simulation time'] = sim.get_timer()
        perf_data['linearization time'] = jac.get_timer()
        perf_data['linear solver time'] = sim.node['linear solver solve'].get_timer() + sim.node[
            'linear solver setup'].get_timer()
        interp = jac.node['interpolation']
        perf_data['interpolation incl. generation time'] = interp.get_timer()
        perf_data['interpolation excl. generation time'] = interp.get_timer() - \
                                                           interp.node['acc flux interpolation'].node[
                                                               'body generation'].get_timer()

        return perf_data

    def save_performance_data(self, file_name=''):
        import platform
        """
        Function to save performance data for future comparison.
        :param file_name:
        :return:
        """
        if file_name == '':
            file_name = 'perf_' + platform.system().lower()[:3] + '.pkl'
        data = self.get_performance_data()
        with open(file_name, "wb") as fp:
            pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_performance_data(file_name=''):
        import platform
        """
        Function to load the performance pkl file at previous simulation.
        :param file_name: performance filename
        """
        if file_name == '':
            file_name = 'perf_' + platform.system().lower()[:3] + '.pkl'
        if os.path.exists(file_name):
            with open(file_name, "rb") as fp:
                return pickle.load(fp)
        return 0

    def print_timers(self):
        """
        Function to print the time information, including total time elapsed,
                                        time consumption at different stages of the simulation, etc..
        """
        print(self.timer.print("", ""))

    def print_stat(self):
        """
        Function to print the statistics information, including total timesteps, Newton iteration, linear iteration, etc..
        """
        self.physics.engine.print_stat()

    def plot_layer_map(self, map_data, k, name, transpose=0):
        """
        Function to plot parameter profile of certain layer.
        :param map_data: data array
        :param k: layer index
        :param name: parameter name
        :param transpose: do transpose to swap axes
        """
        import plotly
        import plotly.graph_objs as go

        nxny = self.reservoir.nx * self.reservoir.ny
        layer_indexes = np.arange(nxny * (k - 1), nxny * k)
        layer_data = np.zeros(nxny)
        # for correct vizualization of inactive cells
        layer_data.fill(np.nan)

        active_mask = np.where(self.reservoir.discretizer.global_to_local[layer_indexes] > -1)
        layer_data[active_mask] = map_data[self.reservoir.discretizer.global_to_local[layer_indexes][active_mask]]

        layer_data = layer_data.reshape(self.reservoir.ny, self.reservoir.nx)
        if transpose:
            layer_data = layer_data.transpose()
            y_axis = dict(scaleratio=1, scaleanchor='x', title='X, block')
            x_axis = dict(title='Y, block')
        else:
            x_axis = dict(scaleratio=1, scaleanchor='x', title='X, block')
            y_axis = dict(title='Y, block')

        data = [go.Heatmap(
            z=layer_data)]
        layout = go.Layout(title='%s, layer %d' % (name, k),
                           xaxis=x_axis,
                           yaxis=y_axis)
        fig = go.Figure(data=data, layout=layout)
        plotly.offline.plot(fig, filename='%s_%d_map.html' % (name, k))

    def plot_layer_map_offline(self, map_data, k, name, transpose=0):
        """
        Function to plot the profile of certain parameter within Jupyter Notebook.
        :param map_data: data array
        :param k: layer index
        :param name: parameter name
        :param transpose: do transpose to swap axes
        """
        import plotly

        plotly.offline.init_notebook_mode()

        self.plot_layer_map(map_data, k, name, transpose)

    def plot_layer_surface(self, map_data, k, name, transpose=0):
        """
        Function to plot the surface of certain parameter.
        :param map_data: data array
        :param k: layer index
        :param name: parameter name
        :param transpose: do transpose to swap axes
        """
        import plotly
        import plotly.graph_objs as go

        nxny = self.reservoir.nx * self.reservoir.ny
        layer_indexes = np.arange(nxny * (k - 1), nxny * k)
        layer_data = np.zeros(nxny)
        # for correct vizualization of inactive cells
        layer_data.fill(np.nan)

        active_mask = np.where(self.reservoir.discretizer.global_to_local[layer_indexes] > -1)
        layer_data[active_mask] = map_data[self.reservoir.discretizer.global_to_local[layer_indexes][active_mask]]

        layer_data = layer_data.reshape(self.reservoir.ny, self.reservoir.nx)
        if transpose:
            layer_data = layer_data.transpose()

        data = [go.Surface(z=layer_data)]
        plotly.offline.plot(data, filename='%s_%d_surf.html' % (name, k))

    def plot_geothermal_temp_layer_map(self, X, k, name, transpose=0):
        import plotly
        import plotly.graph_objs as go
        import numpy as np
        from darts.models.physics.iapws.iapws_property import iapws_temperature_evaluator
        nxny = self.reservoir.nx * self.reservoir.ny

        temperature = iapws_temperature_evaluator()
        layer_pres_data = np.zeros(nxny)
        layer_enth_data = np.zeros(nxny)
        layer_indexes = np.arange(nxny * (k - 1), nxny * k)
        active_mask = np.where(self.reservoir.discretizer.global_to_local[layer_indexes] > -1)
        layer_pres_data[active_mask] = X[2 * self.reservoir.discretizer.global_to_local[layer_indexes][active_mask]]
        layer_enth_data[active_mask] = X[2 * self.reservoir.discretizer.global_to_local[layer_indexes][active_mask] + 1]

        # used_data = map_data[2 * nxny * (k-1): 2 * nxny * k]
        T = np.zeros(nxny)
        T.fill(np.nan)
        for i in range(0, nxny):
            if self.reservoir.discretizer.global_to_local[nxny * (k - 1) + i] > -1:
                T[i] = temperature.evaluate([layer_pres_data[i], layer_enth_data[i]])

        layer_data = T.reshape(self.reservoir.ny, self.reservoir.nx)
        if transpose:
            layer_data = layer_data.transpose()
            y_axis = dict(scaleratio=1, scaleanchor='x', title='X, block')
            x_axis = dict(title='Y, block')
        else:
            x_axis = dict(scaleratio=1, scaleanchor='x', title='X, block')
            y_axis = dict(title='Y, block')

        data = [go.Heatmap(
            z=layer_data)]
        layout = go.Layout(title='%s, layer %d' % (name, k),
                           xaxis=x_axis,
                           yaxis=y_axis)
        fig = go.Figure(data=data, layout=layout)
        plotly.offline.plot(fig, filename='%s_%d_map.html' % (name, k))

    def plot_1d(self, map_data, name):
        """
        Function to plot the 1d parameter.
        :param map_data: data array
        :param name: parameter name
        """
        import plotly
        import plotly.graph_objs as go
        import numpy as np

        nx = self.reservoir.nx
        data = [go.Scatter(x=np.linspace(0, 1, nx), y=map_data[1:nx])]
        plotly.offline.plot(data, filename='%s_surf.html' % name)

    def plot_1d_all(self, map_data):
        """
        Function to plot all parameters of map_data in 1d.
        :param map_data: data array
        """
        import plotly
        import plotly.graph_objs as go
        import numpy as np

        nx = self.reservoir.nx
        nc = self.physics.n_components

        data = []
        for i in range(nc - 1):
            data.append(go.Scatter(x=np.linspace(0, 1, nx), y=map_data[i + 1::nc][1:nx], dash='dash'))

        plotly.offline.plot(data, filename='Compositions.html')

    def plot_cumulative_totals_mass(self):
        """
        Function to plot the cumulative injection and production mass
        """
        import plotly.offline as po
        import plotly.graph_objs as go
        import numpy as np
        import pandas as pd

        nc = self.physics.n_components

        darts_df = pd.DataFrame(self.physics.engine.time_data)
        total_df = pd.DataFrame()
        total_df['time'] = darts_df['time']
        time_diff = darts_df['time'].diff()
        time_diff[0] = darts_df['time'][0]
        for c in range(nc):
            total_df['Total injection c %d' % c] = 0
            total_df['Total production c %d' % c] = 0
            search_str = ' : c %d rate (Kmol/day)' % c
            for col in darts_df.columns:
                if search_str in col:
                    inj_mass = darts_df[col] * time_diff
                    prod_mass = darts_df[col] * time_diff
                    # assuming that any well can inject and produce over the whole time
                    inj_mass[inj_mass < 0] = 0
                    prod_mass[prod_mass > 0] = 0
                    total_df['Total injection c %d' % c] += inj_mass
                    total_df['Total production c %d' % c] -= prod_mass

        data = []
        for c in range(nc):
            data.append(go.Scatter(x=total_df['time'], y=total_df['Total injection c %d' % c].cumsum(),
                                   name='%s injection' % self.physics.components[c]))
            data.append(go.Scatter(x=total_df['time'], y=total_df['Total production c %d' % c].cumsum(),
                                   name='%s production' % self.physics.components[c]))

        layout = go.Layout(title='Cumulative total masses (kmol)', xaxis=dict(title='Time (days)'),
                           yaxis=dict(title='Mass (kmols)'))
        fig = go.Figure(data=data, layout=layout)
        po.plot(fig, filename='Cumulative_totals_mass.html')

    def plot_mass_balance_error(self):
        """
        Function to plot the total mass balance error between injection and production
        """
        import plotly.offline as po
        import plotly.graph_objs as go
        import numpy as np
        import pandas as pd

        nc = self.physics.n_components

        darts_df = pd.DataFrame(self.physics.engine.time_data)
        total_df = pd.DataFrame()
        total_df['time'] = darts_df['time']
        time_diff = darts_df['time'].diff()
        time_diff[0] = darts_df['time'][0]
        for c in range(nc):
            total_df['Total source-sink c %d' % c] = 0
            search_str = ' : c %d rate (Kmol/day)' % c
            for col in darts_df.columns:
                if search_str in col:
                    mass = darts_df[col] * time_diff
                    total_df['Total source-sink c %d' % c] += mass

        data = []
        for c in range(nc):
            total_df['Total mass balance error c %d' % c] = darts_df['FIPS c %d (kmol)' % c] - total_df[
                'Total source-sink c %d' % c].cumsum()
            total_df['Total mass balance error c %d' % c] -= darts_df['FIPS c %d (kmol)' % c][0] - \
                                                             total_df['Total source-sink c %d' % c][0]
            data.append(go.Scatter(x=total_df['time'], y=total_df['Total mass balance error c %d' % c],
                                   name='%s' % self.physics.components[c]))

        layout = go.Layout(title='Mass balance error (kmol)', xaxis=dict(title='Time (days)'),
                           yaxis=dict(title='Mass (kmols)'))
        fig = go.Figure(data=data, layout=layout)
        po.plot(fig, filename='Mass_balance_error.html')

    def plot_FIPS(self):
        import plotly.offline as po
        import plotly.graph_objs as go
        import numpy as np
        import pandas as pd

        nc = self.physics.n_components

        darts_df = pd.DataFrame(self.physics.engine.time_data)
        data = []
        for c in range(nc):
            data.append(go.Scatter(x=darts_df['time'], y=darts_df['FIPS c %d (kmol)' % c],
                                   name='%s' % self.physics.components[c]))

        layout = go.Layout(title='FIPS (kmol)', xaxis=dict(title='Time (days)'),
                           yaxis=dict(title='Mass (kmols)'))
        fig = go.Figure(data=data, layout=layout)
        po.plot(fig, filename='FIPS.html')

    def plot_totals_mass(self):
        import plotly.offline as po
        import plotly.graph_objs as go
        import numpy as np
        import pandas as pd

        nc = self.physics.n_components

        darts_df = pd.DataFrame(self.physics.engine.time_data)
        total_df = pd.DataFrame()
        total_df['time'] = darts_df['time']
        for c in range(nc):
            total_df['Total injection c %d' % c] = 0
            total_df['Total production c %d' % c] = 0
            search_str = ' : c %d rate (Kmol/day)' % c
            for col in darts_df.columns:
                if search_str in col:
                    inj_mass = darts_df[col].copy()
                    prod_mass = darts_df[col].copy()
                    # assuming that any well can inject and produce over the whole time
                    inj_mass[inj_mass < 0] = 0
                    prod_mass[prod_mass > 0] = 0
                    total_df['Total injection c %d' % c] += inj_mass
                    total_df['Total production c %d' % c] -= prod_mass

        data = []
        for c in range(nc):
            data.append(go.Scatter(x=total_df['time'], y=total_df['Total injection c %d' % c],
                                   name='%s injection' % self.physics.components[c]))
            data.append(go.Scatter(x=total_df['time'], y=total_df['Total production c %d' % c],
                                   name='%s production' % self.physics.components[c]))

        layout = go.Layout(title='Total mass rates (kmols/day)', xaxis=dict(title='Time (days)'),
                           yaxis=dict(title='Rate (kmols/day)'))
        fig = go.Figure(data=data, layout=layout)
        po.plot(fig, filename='Totals_mass_rates.html')

    def plot_1d_compare(self, map_data1, map_data2):
        """
        Function to compare the parameter values in two data array
        :param map_data1: data array 1
        :param map_data2: data array 2
        """
        import plotly
        import plotly.graph_objs as go
        import numpy as np

        nx = self.reservoir.nx
        nc = self.physics.n_components

        data = []
        for i in range(nc - 1):
            data.append(go.Scatter(x=np.linspace(0, 1, nx), y=map_data1[i + 1::nc][1:nx],
                                   name="Comp = %d, dt = 5 days" % (i + 1)))

        for i in range(nc - 1):
            data.append(go.Scatter(x=np.linspace(0, 1, nx), y=map_data2[i + 1::nc][1:nx],
                                   name="Comp = %d, dt = 50 days" % (i + 1), line=dict(dash='dot')))

        plotly.offline.plot(data, filename='Compositions.html')

    def body_path_start(self):
        with open('body_path.txt', "w") as fp:
            itor = self.physics.acc_flux_itor
            self.processed_body_idxs = set()
            for i, p in enumerate(itor.axis_points):
                fp.write('%d %lf %lf %s\n' % (p, itor.axis_min[i], itor.axis_max[i], self.body_path_axes[i]))
            fp.write('Body Index Data\n')

    def body_path_add_bodys(self, time):
        with open('body_path.txt', "a") as fp:
            fp.write('T=%lf\n' % time)
            itor = self.physics.acc_flux_itor
            all_idxs = set(itor.body_data.keys())
            new_idxs = all_idxs - self.processed_body_idxs
            for i in new_idxs:
                fp.write('%d\n' % i)
            self.processed_body_idxs = all_idxs

    def save_matlab_map(self, name, np_arr):
        import scipy.io
        scipy.io.savemat(name + '.mat', dict(x=np_arr))

    def export_vtk(self, file_name='data', local_cell_data={}, global_cell_data={}, vars_data_dtype=np.float32,
                   export_grid_data=True):

        # get current engine time
        t = self.physics.engine.t
        nb = self.reservoir.mesh.n_res_blocks
        nv = self.physics.n_vars
        X = np.array(self.physics.engine.X, copy=False)

        for v in range(nv):
            local_cell_data[self.physics.vars[v]] = X[v:nb * nv:nv].astype(vars_data_dtype)

        self.reservoir.export_vtk(file_name, t, local_cell_data, global_cell_data, export_grid_data)
