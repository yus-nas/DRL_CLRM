#!/bin/bash
#SBATCH --job-name=DRL_chan      # User may assign any custom name
#SBATCH --partition=serc  # Partitions for SFC to use on mazama: suprib or twohour
#SBATCH --constraint="CLASS:SH3_CBASE"
#SBATCH --time=7-0:00              # Max 7 days on sherlock
#SBATCH -o %x.o%j            # File to which STDOUT will be written, %j inserts jobid, %x inserts job name
#SBATCH -e %x.e%j            # File to which STDERR will be written, %j inserts jobid, %x inserts job name
#SBATCH --nodes=7
##SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=10GB

# activate environment
##conda activate DRL 

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi


port=8579
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --block &

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 120

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --block &
    sleep 30
done
sleep 120

python -u train_drl_agent.py

