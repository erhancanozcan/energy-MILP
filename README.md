# A Distributed Optimization Framework to Regulate the Electricity Consumption of a Residential Neighborhood

This repository contains the official implementation for the following paper on demand response program:


* [A Distributed Optimization Framework to Regulate the Electricity Consumption of a Residential Neighborhood](https://arxiv.org/abs/2306.10935](https://arxiv.org/abs/2306.09954)


This paper proposes a distributed demand response program to control the overall load consumption of a residential neighborhood. The complexity of the proposed problem grows as the number of participating homes increases. To be able to solve the proposed problem efficiently, we develop a distributed optimization framework based on Dantzig-Wolfe decomposition approach. We show the benefits of utilizing our optimization approach over solving the centralized problem using a commercial solver by conducting various experiments in a simulated environment.

Please consider citing our paper as follows:

```
@misc{ozcan2023distributed,
  title={A Distributed Optimization Framework to Regulate the Electricity Consumption of a Residential Neighborhood},
  author={Ozcan, Erhan Can and Paschalidis, Ioannis Ch},
  journal={arXiv preprint arXiv:2306.09954},
  year={2023}
}
``` 

## Solvers and Results

While the first command below can be used to solve the centralized model, the second command solves the problem in a distributed way without removing any column. Finally, the third command removes the columns that have remained non-basic for 5 consecutive iterations. All of the commands require Gurobi to be installed. 


```
nohup python -u -m energy.code.solver_ca_full_no_abs --s_effect 1 --num_houses 1000  \
        --mipgap 1e-2 --opt_tolerance 1e-6 --seed 0  \
        --timelimit 900 \
        --Q -1.0 --iter_limit 2000 \
        --save_file seed_0_mipgap_1en2_opt_1en6_1000hQn1winter_season_t_900 >> /home/erhan/energy/out/distributed/seed_0_mipgap_1en2_opt_1en6_full_winter_1000h_t900.txt 2>&1 & 
  
nohup python -u -m energy.code.solver_dist_IP --s_effect 1 --num_houses 1000  \
        --mipgap 1e-2 --opt_tolerance 1e-3 --seed 0  \
        --timelimit 60  \
        --Q -1.0 --iter_limit 2000 --unused_iter_limit 2000 \
        --save_file seed_0_opt_1en3_dist_unused2000_1000hQn1_winter_IP_mipgap1en2 >> /home/erhan/energy/out/distributed/seed_0_opt_1en3_dist_unused2000_winter_1000h_IP_mipgap1en2.txt 2>&1 &

nohup python -u -m energy.code.solver_dist_IP --s_effect 1 --num_houses 1000  \
        --mipgap 1e-2 --opt_tolerance 1e-3 --seed 0  \
        --timelimit 60  \
        --Q -1.0 --iter_limit 2000 --unused_iter_limit 5 \
        --save_file seed_0_opt_1en3_dist_unused5_1000hQn1_winter_IP_mipgap1en2 >> /home/erhan/energy/out/distributed/seed_0_opt_1en3_dist_unused5_winter_1000h_IP_mipgap1en2.txt 2>&1 &
```  

For more information on the inputs accepted, you can use the --help option or reference energy/code/common/arg_parser.py. The results of the experiments are saved in the energy/logs/ folder upon completion.
