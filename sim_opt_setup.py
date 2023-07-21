import numpy as np


def Sim_opt_setup():
    sim_input = {}
    
    # Grid
    sim_input["nx"] =  60 
    sim_input["ny"] =  60
    sim_input["nz"] =  1
    sim_input["dx"] =  38.1 
    sim_input["dy"] =  38.1
    sim_input["dz"] =  9.144

    # Initialization
    sim_input["datum"] = 3000
    sim_input["Pi"] = 350
    sim_input["Swi"] =  0.15

    # Rock
    sim_input["depth"] = 3000
    sim_input["poro"] = 0.2 
    sim_input["kz"] = 100 
    sim_input["actnum"] = np.ones(sim_input["nx"]*sim_input["ny"])

    # realization
    sim_input["realz"] = 2
    sim_input["realz_path"] = "/scratch/users/nyusuf/Research_projects/Models/Chan_1000/Ensemble/_multperm"
    sim_input["num_realz"] = 997
    sim_input["cluster_labels"] = np.loadtxt("Model_clusters/cluster_label_chan_1000.txt").astype(int)
    sim_input["cluster_train_realz"] = np.loadtxt("Model_clusters/cluster_train_realz_chan_1000.txt", delimiter=",").astype(int)
#     sim_input["sampling_prob"] = np.loadtxt("Model_clusters/sampling_prob.txt")

    # well
    sim_input["well_radius"] = 0.3048*0.5
    sim_input["prod_bhp"] = 345
    sim_input["inj_bhp"] = 400
    sim_input["skin"] = 0

    # timing
    sim_input["total_time"] = 1600
    sim_input["num_run_per_step"] = 4
    sim_input["runtime"] = 50

    # economics
    sim_input["oil_price"] = 55
    sim_input["opex"] = 0
    sim_input["discount_rate"] = 0.1
    sim_input["wat_prod_cost"] = 5
    sim_input["wat_inj_cost"] = 5
    sim_input["npv_scale"] = 1e8

    # constraint
    sim_input["max_liq_prod_rate"] = 1526
    sim_input["max_water_inj_rate"] = None
    sim_input["water_cut_limit"] = None
    
    # noise
    sim_input["noise"] = True
    sim_input["std_rate_min"] = 1.5
    sim_input["std_rate_max"] = 7.94936
    sim_input["std_rate"] = 0.05
    sim_input["std_pres"] = 0.344738

    # solver
    sim_input["first_ts"] = 0.01
    sim_input["mult_ts"] = 2
    sim_input["max_ts"] = 50        # days  # max time step 
    sim_input["tolerance_newton"] = 1e-6
    sim_input["tolerance_linear"] = 1e-8

    # well completion
    sim_input["well_comp"] = []
    sim_input["well_comp"].append(['P1', 'PROD', 11, 11, 1, 1])
    sim_input["well_comp"].append(['P2', 'PROD', 9, 53, 1, 1])
    sim_input["well_comp"].append(['P3', 'PROD', 31, 31, 1, 1])
    sim_input["well_comp"].append(['P4', 'PROD', 51, 11, 1, 1])
    sim_input["well_comp"].append(['P5', 'PROD', 51, 51, 1, 1])

    sim_input["well_comp"].append(['I1', 'INJ', 8, 24, 1, 1])
    sim_input["well_comp"].append(['I2', 'INJ', 36, 11, 1, 1])
    sim_input["well_comp"].append(['I3', 'INJ', 19, 41, 1, 1])
    sim_input["well_comp"].append(['I4', 'INJ', 51, 37, 1, 1])
    
    
    # Opt controls
    prod_bhp_bound = [280, 345]
    inj_bhp_bound = [370, 500]
    hist_control_prod = 345
    hist_control_inj = 400
    num_prod = 5
    num_inj = 4
    
    
    # Preprocess control info
    sim_input["lower_bound"] = np.concatenate(([prod_bhp_bound[0]]*num_prod, [inj_bhp_bound[0]]*num_inj))
    sim_input["upper_bound"] = np.concatenate(([prod_bhp_bound[1]]*num_prod, [inj_bhp_bound[1]]*num_inj))
    sim_input["hist_ctrl"] = np.concatenate(([hist_control_prod]*num_prod, [hist_control_inj]*num_inj))
    sim_input["hist_ctrl_scaled"] = (sim_input["hist_ctrl"]-sim_input["lower_bound"])/(sim_input["upper_bound"]-sim_input["lower_bound"])

    # scaling factors
    sim_input["scaler"] = np.loadtxt("scaling_factors_chan_1000.txt", delimiter=",")
    
    return sim_input