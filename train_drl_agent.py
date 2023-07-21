import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from reservoir_env import ReservoirEnv
from network_model_attention_cnn import GTrXLNet as MyModel
from sim_opt_setup import Sim_opt_setup
import os

# parameters
num_cpus = 135 #int(sys.argv[1])
num_opt_ctrl_step = 7
num_sim_iter = 2
num_training_iter = 500

ray.init(ignore_reinit_error=True, log_to_driver=False, address=os.environ["ip_head"])#, memory=500 * 1024 * 1024)
ModelCatalog.register_custom_model("my_model", MyModel)

sim_input = Sim_opt_setup()

cur_env_config = {"sim_input": sim_input}

nstep = num_opt_ctrl_step*num_sim_iter
tune.run(
    "PPO",
    stop={ "training_iteration": num_training_iter,},
    config={
        "env": ReservoirEnv,
        "model": {
            "custom_model": "my_model",
            "max_seq_len": 7,
            "custom_model_config": {
                "num_transformer_units": 3, #base 2
                "attention_dim": 128, #128
                "num_heads": 2, #base 2
                "memory_inference": 7, 
                "memory_training": 7,  
                "head_dim": 64, #64
                "position_wise_mlp_dim": 64,  #base 64
            },
        },
        "num_workers": num_cpus,
        "num_cpus_for_driver": 5,
        "num_gpus": 0,
        "train_batch_size": num_cpus * nstep,  # Total number of steps per iterations
        
        "rollout_fragment_length": nstep,
        "sgd_minibatch_size": 128, #base 256

        #"lr": 5e-5,
        "gamma": 0.9997,
        "lr_schedule": [[0, 1e-4], [num_cpus * nstep * num_training_iter, 5e-6]], 
		"entropy_coeff_schedule": [[0, 5e-4], [num_cpus * nstep * num_training_iter, 1e-7]],
        #"entropy_coeff": 0, 
        "vf_loss_coeff": 1, 
        "num_sgd_iter": 15, #10
        "env_config": cur_env_config,
    },
   # checkpoint_at_end=True,
    checkpoint_freq = 10,
    local_dir="./logs", 
    #restore="/scratch/users/nyusuf/logs/PPO/PPO_ReservoirEnv_a1429_00000_0_2021-09-18_16-53-32/checkpoint_000300/checkpoint-300"
)
