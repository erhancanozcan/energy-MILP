"""Creates command line parser for train.py."""
import argparse

parser = argparse.ArgumentParser()


# Setup
##########################################
parser.add_argument('--runs',help='number of trials',type=int,default=1)
parser.add_argument('--runs_start',help='starting trial index',
    type=int,default=0)
parser.add_argument('--cores',help='number of processes',type=int)
parser.add_argument('--seed',help='master seed',type=int,default=0)
parser.add_argument('--setup_seed',help='setup seed',type=int)

parser.add_argument('--save_path',help='save path',type=str,default='./energy/logs')
parser.add_argument('--save_file',help='save file name',type=str)


#Home
home_kwargs=['s_effect', 'flex_prob','controlled_cost','pv_generation', 'mean_uncontrollable']
parser.add_argument('--s_effect',help='seasonal effect for HVAC 1 heating minus 1 cooling',type=int,default=-1)
parser.add_argument('--flex_prob',help='flexibility probability for home',type=float,default=0.0)
parser.add_argument('--controlled_cost',help='use the controlled deviation cost structure', action='store_true')
parser.add_argument('--pv_generation',help='total pv generation per home during planning horizon. 0,5 and 20 might be good options',type=float,default=0.0)
parser.add_argument('--mean_uncontrollable',help='mean for uncontrollable load per home during planning horizon. 0, 0.1, 0.2 might be good options',type=float,default=0.0)

#Coordination Agent
ca_kwargs=['num_houses', 'horizon', 'price', 'deviation_cost', 'Q','uncontrollable_file',
           'renewable_file','powertobuy','lambda_gap', 'mipgap', 'timelimit','p_ub','iter_limit','opt_tolerance','unused_iter_limit']

parser.add_argument('--num_houses',help='number of houses in the community',type=int,default=2)
parser.add_argument('--horizon',help='number of time intervals in next 24 hours',type=int,default=96)
parser.add_argument('--price',help='mean electricity price Kwh',type=float,default=0.35)
parser.add_argument('--deviation_cost',help='mean deviation cost',type=float,default=0.01)
parser.add_argument('--Q',help='desired agregated power level in KwH',type=float,default=-1.0)
parser.add_argument('--uncontrollable_file',help='import file name for Q_modify',type=str)
parser.add_argument('--renewable_file',help='import file name for Q_modify',type=str)
parser.add_argument('--powertobuy',help='Additioanl Power amount required to maintain a balance between supply and demend. It has to be non-negative float.',type=float,default=10000)
parser.add_argument('--lambda_gap',help='duality gap penalizer coefficient',type=float,default=1.0)
parser.add_argument('--mipgap',help='mipgap value of the QCQP problem',type=float,default=1e-4)
parser.add_argument('--timelimit',help='timelimit in seconds for coordination agent problem',type=float,default=60)
parser.add_argument('--p_ub',help='upper bound on the amount of price deviation',type=float,default=2.0)
parser.add_argument('--iter_limit',help='maximum number of cg iterations',type=float,default=400)
parser.add_argument('--unused_iter_limit',help='maximum number of iteration a column can remain as non-basic',type=float,default=400)
parser.add_argument('--opt_tolerance',help='If reduced cost is larger than this iterations continue',type=float,default=1e-6)


slp_kwargs=['n_repeat']

parser.add_argument('--n_repeat',help='number of successive linear program optimizations',type=int,default=25)







# For export to solver.py
#########################################
def create_train_parser():
    return parser

all_kwargs={
    'home_kwargs': home_kwargs,
    'ca_kwargs':   ca_kwargs,
    'slp_kwargs':  slp_kwargs
    }









