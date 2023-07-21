from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from correlation.dead_oil_uniform import DeadOil
from correlation.properties_correlations import *

import pandas as pd
import numpy as np
import os
import contextlib
import copy

class Simulator(DartsModel):
    def __init__(self, sim_input):
        # call base class constructor
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f): # disable print
            super().__init__()
                
        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()
        
        # dimension
        self.nx = sim_input["nx"]
        self.ny = sim_input["ny"]
        self.nz = sim_input["nz"]
        
        # scaler
        self.scaler = sim_input["scaler"]
        
        # create sim and reservoir objects
        self.reset_vars(sim_input)
        
    def reset_vars(self, res_param):
        
        # create copy of input data
        sim_input = copy.deepcopy(res_param)
        
        # solver parameters
        self.params.first_ts = sim_input["first_ts"]
        self.params.mult_ts = sim_input["mult_ts"]
        self.params.max_ts = sim_input["max_ts"]
        self.params.tolerance_newton = sim_input["tolerance_newton"]
        self.params.tolerance_linear = sim_input["tolerance_linear"]

        # timing
        self.runtime = sim_input["runtime"]
        self.total_time = sim_input["total_time"]
        self.num_run_per_step = sim_input["num_run_per_step"]
        
        # economic parameters
        self.oil_price = sim_input["oil_price"]
        self.wat_prod_cost = sim_input["wat_prod_cost"]
        self.wat_inj_cost = sim_input["wat_inj_cost"]
        self.opex = sim_input["opex"]
        self.discount_rate = sim_input["discount_rate"]
        self.npv_scale = sim_input["npv_scale"]

        # well parameters
        self.num_prod  = 0
        self.num_inj   = 0
        self.prod_bhp = sim_input["prod_bhp"]
        self.inj_bhp = sim_input["inj_bhp"]
        self.well_radius = sim_input["well_radius"]
        self.skin = sim_input["skin"]
        self.wc_lim = sim_input["water_cut_limit"]
        self.max_liq_prod_rate = sim_input["max_liq_prod_rate"]
        self.max_water_inj_rate = sim_input["max_water_inj_rate"]
        self.inj_stream = [0.999]
        
        # economic parameters
        self.oil_price = sim_input["oil_price"]
        self.wat_prod_cost = sim_input["wat_prod_cost"]
        self.wat_inj_cost = sim_input["wat_inj_cost"]
        self.opex = sim_input["opex"]
        self.discount_rate = sim_input["discount_rate"]
        self.reward_scale = sim_input["npv_scale"]
        
        # noise
        self.noise = sim_input["noise"]
        self.std_rate_min = sim_input["std_rate_min"]
        self.std_rate_max = sim_input["std_rate_max"]
        self.std_rate = sim_input["std_rate"]
        self.std_pres = sim_input["std_pres"]
        
        # initial P and Zw
        self.init_p = sim_input["Pi"]
        self.init_sw = sim_input["Swi"]  

        # reservoir construction
        kx = np.loadtxt(sim_input["realz_path"]+"{}.in".format(sim_input["realz"]), skiprows=1, comments='/')
        #poro = np.loadtxt(sim_input["realz_path"]+"/_poro{}.in".format(sim_input["realz"]), skiprows=1, comments='/')
        self.reservoir = None
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f): # disable print
            self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz, \
                  dx=sim_input["dx"], dy=sim_input["dy"], dz=sim_input["dz"], permx=kx, permy=kx,\
                  permz=sim_input["kz"], poro=sim_input["poro"], depth=sim_input["depth"],actnum=sim_input["actnum"])
        
        # setup physics
        self.setup_physics()
                   
        # add wells
        for comp_dat in sim_input["well_comp"]:
            self.add_well(*comp_dat)
       
        # initialize DARTS
        self.init()
        self.timer.node["initialization"].stop()
        
    def setup_physics(self):
        # physical properties
        self.physics = None
        self.property_container = None
        self.property_container = PropertyContainer(phase_name=['water', 'oil'], component_name=['water', 'oil'],
                                                    rate_name=['water', 'oil', 'liquid'])

        # Define property evaluators based on custom properties
        self.property_container.density_ev = dict([('water', DensityWater()), ('oil', DensityOil())])
        self.property_container.viscosity_ev = dict([('water', ViscosityWater()), ('oil', ViscosityOil())])
        self.property_container.watersat_ev = Watersaturation()
        self.property_container.rel_perm_ev = dict([('water', PhaseRelPerm("water")), ('oil', PhaseRelPerm("oil"))])
        self.property_container.capillary_ev = Capillarypressure()
        self.property_container.rock_compress_ev = RockCompactionEvaluator()

        # create physics
        self.grav = 0
        self.physics = DeadOil(self.timer, n_points=250, min_p=200, max_p=600, min_z=1e-13,
                               property_container=self.property_container, grav=self.grav)
    

    def set_initial_conditions(self):
        oil_dens = self.physics.property_data.density_ev['oil'].evaluate([self.init_p, []])
        wat_dens = self.physics.property_data.density_ev['water'].evaluate([self.init_p, []]) 
        Zw_init = (wat_dens * self.init_sw)/(wat_dens * self.init_sw + oil_dens * (1 - self.init_sw)) 
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=self.init_p,\
                                                    uniform_composition=[Zw_init])

                
    def set_boundary_conditions(self):           
        for well in self.reservoir.wells: 
            if well.name[0] == 'I':  
                well.control = self.physics.new_bhp_water_inj(self.inj_bhp, self.inj_stream)
                #well.constraint = self.physics.new_rate_water_inj(1000, self.inj_stream)
            else:     
                well.control = self.physics.new_bhp_prod(self.prod_bhp)
                well.constraint = self.physics.new_rate_liq_prod(self.max_liq_prod_rate)
       
    
#     def set_well_control(self, cont):
        
#         pres = self.physics.engine.X[0::2]
#         p_allowance = 0.2
#         i = 0
#         for well in self.reservoir.wells:
#             if well.name[0] == 'I':  
#                 well_blck = well.perforations[0][1]
#                 inj_bhp = np.max([cont[i], pres[well_blck]+p_allowance])
#                 well.control = self.physics.new_bhp_water_inj(inj_bhp, self.inj_stream) 
#                 i+=1
#             else:
#                 well.control = self.physics.new_bhp_prod(self.prod_bhp)
#                 well.constraint = self.physics.new_rate_liq_prod(self.max_liq_prod_rate)
    
    
    def set_well_control(self, cont):  
        assert len(cont) == len(self.reservoir.wells)
        
        pres = self.physics.engine.X[0::2]
        p_allowance = 0.2
        for well, ctrl in zip(self.reservoir.wells, cont):
            well_blck = well.perforations[0][1]
            if well.name[0] == 'I':                 
                inj_bhp = np.max([ctrl, pres[well_blck]+p_allowance])
                well.control = self.physics.new_bhp_water_inj(inj_bhp, self.inj_stream)         
            else:
                if ctrl > pres[well_blck]-p_allowance:
                    well.control = self.physics.new_rate_liq_prod(0)
                else:
                    well.control = self.physics.new_bhp_prod(ctrl)
                    well.constraint = self.physics.new_rate_liq_prod(self.max_liq_prod_rate) # needs to be repeated
      
    def check_inverse_flow(self):
        # extract data 
        td = pd.DataFrame.from_dict(self.physics.engine.time_data.copy())
        time_data = td.iloc[-1]
        pres = self.physics.engine.X[0::2]
        p_allowance = 2
        
        for well in self.reservoir.wells:  
            WWR = time_data['{} : water rate (m3/day)'.format(well.name)]
            WOR = time_data['{} : oil rate (m3/day)'.format(well.name)] 
            
            if well.name[0] == 'I' and WWR < 0:     
                well_blck = well.perforations[0][1]
                inj_bhp = pres[well_blck]+p_allowance
                well.control = self.physics.new_bhp_water_inj(inj_bhp, self.inj_stream)
                #well.control = self.physics.new_rate_water_inj(0, self.inj_stream)
            elif well.name[0] == 'P' and WOR > 0:                
                well.control = self.physics.new_rate_liq_prod(0)
     
    def add_well(self, well_name, well_type, loc_x, loc_y, loc_z1, loc_z2):               
        rad = self.well_radius
        
        # drill well
        if well_type == 'INJ': # INJECTOR
            self.num_inj += 1
            self.reservoir.add_well(well_name, wellbore_diameter=rad*2)
            skin = 0
        elif well_type == 'PROD': # PRODUCER
            self.num_prod += 1
            self.reservoir.add_well(well_name, wellbore_diameter=rad*2)
            skin = self.skin
        else:
            raise ValueError('Wrong well type for well {}'.format(well_name))
        
        # perfs   
        for loc_z in range(loc_z1, loc_z2+1):
            self.reservoir.add_perforation(self.reservoir.wells[-1], loc_x, loc_y, loc_z,\
                                           well_radius=rad, skin=skin, multi_segment=False)
            
    def run_single_ctrl_step(self):    
        
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f): # disable print
            # advance simulation 
            for run in range(self.num_run_per_step):
#                 if run == 0:
#                     self.run_python(restart_dt=self.params.first_ts)
#                 else:
                self.run_python()
                
        return self.physics.engine.t >= self.total_time
    
    
    def calculate_npv(self):
        
        # beginning of current production stage
        time_from = self.physics.engine.t - self.runtime * self.num_run_per_step
        
        # extract data for current production stage
        td = pd.DataFrame.from_dict(self.physics.engine.time_data.copy())
        time_data = td.truncate(before=td.loc[td.time > time_from].index[0])
        
        # time
        time = time_data['time'].values
        
        # time steps
        time_steps = time - np.concatenate(([time_from], time[:-1]))
        
        # production and injection data
        disc_cum_oil_prod = disc_cum_water_prod = disc_cum_water_inj = 0
        for well in self.reservoir.wells:         
            
            WOR = 6.28981 * np.abs(time_data['{} : oil rate (m3/day)'.format(well.name)].values)
            WWR = 6.28981 * np.abs(time_data['{} : water rate (m3/day)'.format(well.name)].values)
            
            # temporal production/injection volume
            WO = time_steps * WOR
            WW = time_steps * WWR

            # discounted production/injection 
            discounted_time = np.float_power(1 + self.discount_rate, -time/365.0)
            well_disc_cum_oil = np.sum(WO * discounted_time)
            well_disc_cum_wat = np.sum(WW * discounted_time)
            
            # aggregate production/injection
            if well.name[0] == 'P':               
                disc_cum_oil_prod += well_disc_cum_oil
                disc_cum_water_prod += well_disc_cum_wat
            else:
                disc_cum_water_inj += well_disc_cum_wat
                      
        # economic parameters
        po, cwp, cwi, opex = self.oil_price, self.wat_prod_cost, self.wat_inj_cost, self.opex
                
        # calculate npv
        npv = disc_cum_oil_prod * (po - opex) - disc_cum_water_prod * cwp - disc_cum_water_inj * cwi 
   
        return npv / self.npv_scale

    def get_observation(self): 
        
        # beginning of current production stage
        time_from = self.physics.engine.t - self.runtime * self.num_run_per_step
        time_of_interest = np.arange(time_from, self.physics.engine.t, self.runtime) + self.runtime
        
        # extract data for current production stage
        td = pd.DataFrame.from_dict(self.physics.engine.time_data.copy())
        time_data = td.truncate(before=td.loc[td.time > time_from].index[0])
        
        # mask for times of interest
        mask = time_data.isin({'time':time_of_interest})['time'].values
        
        obs_data = np.array([]).reshape(0, self.num_run_per_step)
        for well in self.reservoir.wells:                     
            WOR = np.abs(time_data['{} : oil rate (m3/day)'.format(well.name)].values)
            WWR = np.abs(time_data['{} : water rate (m3/day)'.format(well.name)].values)
            BHP = np.abs(time_data['{} : BHP (bar)'.format(well.name)].values)
            
            if self.noise:
                WOR = self.add_noise(WOR, "rate")
                WWR = self.add_noise(WWR, "rate")
                BHP = self.add_noise(BHP, "pressure")
            
            # observation
            obs_data = np.vstack((obs_data, BHP[mask]))
            if well.name[0] == 'I':
                obs_data = np.vstack((obs_data, WWR[mask]))     
            else:
                obs_data = np.vstack((obs_data, WOR[mask]))
                WC = WWR/(WWR+WOR+1e-6)
                obs_data = np.vstack((obs_data, WC[mask]))
                    
        unscaled = obs_data.T.flatten()    
        scaled =  self.scale_channel(unscaled, self.scaler[:,0], self.scaler[:,1])
        
        return scaled, unscaled
                
    def scale_channel(self, in_ch, min_ch, max_ch,  new_range=[0,1]):
            
        return (new_range[1] - new_range[0]) * (in_ch - min_ch)/(max_ch - min_ch) + new_range[0]
     
    def add_noise(self, qoi, qty_type):
        
        rand_vec = np.random.normal(0, 1, size=qoi.shape)
        if qty_type == "rate":
            qoi_noise = np.abs(qoi + (qoi*self.std_rate).clip(self.std_rate_min, self.std_rate_max) * rand_vec)
        elif qty_type == "pressure":
            qoi_noise = np.abs(qoi + self.std_pres * rand_vec)
        else:
            raise ValueError('Wrong quantity type')
                  
        return qoi_noise
       
