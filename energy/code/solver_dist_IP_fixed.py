import os
import numpy as np
from gurobipy import *
import numpy as np
from datetime import datetime
import copy
import pickle

import sys
sys.path.append("/Users/can/Documents/GitHub")




from energy.code.home import Home
from energy.code.common.demand import initialize_demand
from energy.code.common.appliance import initialize_appliance_property
from energy.code.common.dual_constraints import solve_dual
from energy.code.common.arg_parser import create_train_parser, all_kwargs
from energy.code.common.home_change_objective import change_objective_solve
from energy.code.common.restricted_master_modified import restricted_master




def gather_inputs(args):
    """Organizes inputs to prepare for simulations."""

    args_dict = vars(args)
    inputs_dict = dict()

    for key,param_list in all_kwargs.items():
        active_dict = dict()
        for param in param_list:
            active_dict[param] = args_dict[param]
        inputs_dict[key] = active_dict

    return inputs_dict


def train(inputs_dict):
    
    #np.random.seed(inputs_dict['setup_kwargs']['setup_seed'])
    rng = np.random.default_rng(inputs_dict['setup_kwargs']['setup_seed'])
    
    dual_list=[]
    models=[]
    p_obj_list=[]
    d_obj_list=[]
    power_HVAC_list=[]
    real_power_list_before_changing_price=[] #use this list to understand how optimizing price affects the load consumption.
    dev_power_list_before_changing_price=[]
    
    s_effect=inputs_dict['home_kwargs']['s_effect']
    num_homes=inputs_dict['ca_kwargs']['num_houses']
    horizon=inputs_dict['ca_kwargs']['horizon']
    mean_price=inputs_dict['ca_kwargs']['price']
    mean_Q=inputs_dict['ca_kwargs']['Q']
    #Q=abs(np.random.normal(mean_Q,10,size=horizon))#Kw supply
    lambda_gap=inputs_dict['ca_kwargs']['lambda_gap']
    MIPGap=inputs_dict['ca_kwargs']['mipgap']
    TimeLimit=inputs_dict['ca_kwargs']['timelimit']
    p_ub=inputs_dict['ca_kwargs']['p_ub']
    iter_limit=inputs_dict['ca_kwargs']['iter_limit']
    opt_tolerance=inputs_dict['ca_kwargs']['opt_tolerance']
    unused_iter_limit=inputs_dict['ca_kwargs']['unused_iter_limit']
    
    #price=abs(np.random.normal(mean_price,0.1,size=horizon))#0.33$ per KwH
    price=abs(rng.normal(mean_price,0.1,size=horizon))#0.33$ per KwH
    #price=np.zeros(horizon)
    
    #Random demand and appliance initialization for each home.
    i=0
    while i<num_homes:
        home=Home(s_effect=s_effect)
        home=initialize_demand(home,rng=rng)
        home=initialize_appliance_property(home,s_effect,rng=rng)
        home.generate_desirable_load()
        
        assert (horizon == len(home.wm_desirable_load),"Horizon Change is detected. Check Time resolution of appliances")
        total,cost_u,daily_fee_desirable=home.total_desirable_load(price,mean_price)
        try:
            real_power,dev_power,states,dual,m,p_obj=home.optimize_mpc(cost_u,price)
            tmp_p_name="H"+str(i+1)+"_P_"
            for v in m.getVars():
                v.varname=tmp_p_name+v.varname
            m.update()
            #m=change_objective_solve(m,dual['c'])
            i=i+1
            models.append(m)
            dual_list.append(dual)
            p_obj_list.append(p_obj)
            power_HVAC_list.append(home.hvac.nominal_power)
            real_power_list_before_changing_price.append(real_power)
            dev_power_list_before_changing_price.append(dev_power)
        except Exception:
            print("demand was infeasible due to initialization skip this home.")
            pass
    
    if mean_Q > 0:
        Q=abs(np.random.normal(mean_Q,10,size=horizon))#Kw supply
    else:
        cumulative_desired_energy=0
        generated_PV=0
        for i in range(num_homes):
            real_c=real_power_list_before_changing_price[i]
            generated_PV+=np.sum(real_c['pv'])
            dev_c=dev_power_list_before_changing_price[i]
            for key in dev_c.keys():
                cumulative_desired_energy+=np.sum(real_c[key])-np.sum(dev_c[key])
        Q=np.repeat((cumulative_desired_energy-generated_PV)/horizon,horizon)
    #Information Extractors from extreme points.
    
    #D_P is a matrix of which each row is D_P_(t+k)
    D_P_list=[]
    for i in range(num_homes):
        D_P=np.zeros((horizon,len(m.getVars())))
        for k in range (horizon):
            #Please see that
               #m.getVars()[horizon*6] is the first time interval of the first real appliance
               #m.getVars()[horizon*12] is the first time interval of the last real appliance
            D_P[k,horizon*6+k]=1
            D_P[k,horizon*7+k]=1
            D_P[k,horizon*8+k]=1
            D_P[k,horizon*9+k]=power_HVAC_list[i]
            D_P[k,horizon*10+k]=1
            D_P[k,horizon*11+k]=1
            D_P[k,horizon*12+k]=1
        D_P_list.append(D_P)
    
    #D_d is a matrix of which each row is D_d_(j)
    #Please note that we have deviation on 6 variables.
    #Check the number of y variables.
    D_d=np.zeros((6,len(m.getVars())))
    for j in range(D_d.shape[0]):
        start=j*horizon
        end=(j+1)*horizon
        D_d[j,start:end]=1
    
    D_d_list=[]
    for i in range(num_homes):
        cost=dual_list[i]['c'][:6*horizon]
        cost=cost[[0,1*horizon,2*horizon,3*horizon,4*horizon,5*horizon]]
        
        cost_modified=np.multiply(D_d.T,cost).T
        cost_modified=np.sum(cost_modified,axis=0)
        D_d_list.append(cost_modified)
    
    
    r_m=restricted_master(num_homes,horizon,Q,D_P_list,D_d_list)
    

    

    
    #add first extreme points for each home.
    for i in range(num_homes):
        e_p=np.array([models[i].getVars()[idx].X for idx in range(len(models[i].getVars()))])
        r_m.add_first_extreme_points(e_p,i)
    
    r_m.prob.update()
    r_m.coupling_constrs()
    
    r_m.prob.Params.LogToConsole=0
    r_m.prob.optimize()
    r_m.objective.append(r_m.prob.ObjVal)
    r_m.prob.update()
    #r_m.prob.write("/Users/can/Desktop/tmp_model.lp")
    
    #dual variables
    #r_m.prob.Pi
    
    pos_dual=[r_m.pos[k].pi for k in range(horizon)]
    neg_dual=[r_m.neg[k].pi for k in range(horizon)]
    sum_lambda_dual=[r_m.prob.getConstrByName("sum_lambda"+str(i)).pi for i in range(num_homes)]
    coupling_dual=[r_m.prob.getConstrByName("coupling_"+str(k)).pi for k in range(horizon)]
    
    dual=pos_dual+neg_dual+sum_lambda_dual+coupling_dual
    
    
    #Check the objective.
    #np.dot(np.array(r_m.prob.RHS),np.array(dual))
    
    #extreme point extreme ray generation loop.
    
    second_vector=np.sum(np.multiply(D_P.T,np.array(coupling_dual)).T,axis=0)
    stopper=True
    iter_counter=0
    opt_time=0
    lagrangean_dual=[]
    while stopper:
    #for counter in range(10):
        iter_counter+=1
        stopper=False
        if iter_counter%10==0:
            print (iter_counter)
        max_time=0
        lagrangean_dual_tmp=np.dot(np.array(coupling_dual),Q)
        for i in range(num_homes):
            home_start_time = datetime.now()
            
            term_one_cost_vector= D_d_list[i]
            
            #term_two_cost_vector=np.zeros(horizon)
            
            #term_two_cost_vector=np.concatenate([term_two_cost_vector,np.zeros(horizon)])
            
            term_two_scalar=0+0+sum_lambda_dual[i]
            
            cost=(term_one_cost_vector-second_vector)
            
            
            m=change_objective_solve(models[i],cost)
            
            #if counter==30:
            #    print("can")
            #    #3.391590444677e+01
            #137.639 
            lagrangean_dual_tmp+=m.ObjVal
            #if m.ObjVal < term_two_scalar - opt_tolerance:
            if m.ObjVal < term_two_scalar:
                #print(m.ObjVal -term_two_scalar)
                e_p=np.array([m.getVars()[idx].X for idx in range(len(m.getVars()))])
                r_m.add_other_extreme_points(e_p,i)
                stopper=True
            home_end_time = datetime.now()
            
            time_diff=(home_end_time-home_start_time).seconds+\
                        (home_end_time-home_start_time).microseconds*1e-6
            if time_diff>max_time:
                max_time=time_diff
        
        lagrangean_dual.append(lagrangean_dual_tmp)
        
        r_m_start=datetime.now()
        r_m.prob.update()
        r_m.coupling_constrs()
        
        r_m.prob.optimize()
        r_m.objective.append(r_m.prob.ObjVal)
        if lagrangean_dual[-1]>0 and abs(r_m.objective[-2]-lagrangean_dual[-1])/abs(lagrangean_dual[-1]) < opt_tolerance:
            stopper=False
        r_m.prob.update()
        
        pos_dual=[r_m.pos[k].pi for k in range(horizon)]
        neg_dual=[r_m.neg[k].pi for k in range(horizon)]
        sum_lambda_dual=[r_m.prob.getConstrByName("sum_lambda"+str(i)).pi for i in range(num_homes)]
        coupling_dual=[r_m.prob.getConstrByName("coupling_"+str(k)).pi for k in range(horizon)]
        
        dual=pos_dual+neg_dual+sum_lambda_dual+coupling_dual
        
        second_vector=np.sum(np.multiply(D_P.T,np.array(coupling_dual)).T,axis=0)
        if iter_counter>iter_limit:
            stopper=False
        r_m_end=datetime.now()
        
        r_m_time=(r_m_end-r_m_start).seconds+\
                    (r_m_end-r_m_start).microseconds*1e-6
        opt_time+=(r_m_time)+max_time # in seconds
        
        if iter_counter==1:
            p_in_use=[np.zeros(len(r_m.lambdas[i])) for i in range(num_homes)]
        else:
            r_m.ex_columns_list=[]
            #ex_columns_list=[]
            for i in range(num_homes):
                
                if len(p_in_use[i])!=len(r_m.lambdas[i]):
                    p_in_use[i]=np.concatenate((p_in_use[i],np.array([0])))
                
                existing_columns=[]
                for j in range(len(p_in_use[i])):
                    if p_in_use[i][j]!=-1:
                        existing_columns.append(j)
                        
                for j in existing_columns:
                    if r_m.lambdas[i][j].X==0:
                        p_in_use[i][j]=p_in_use[i][j]+1
                    else:
                        p_in_use[i][j]=0
                        
                    
                    if p_in_use[i][j]==unused_iter_limit:
                        print("Home: %d Column: %d is removed"%(i,j))
                        p_in_use[i][j]=-1
                        r_m.prob.remove(r_m.lambdas[i][j])
                        
                existing_columns=[]
                for j in range(len(p_in_use[i])):
                    if p_in_use[i][j]!=-1:
                        existing_columns.append(j)
                #
                r_m.ex_columns_list.append(existing_columns)
            
                        
    print("Relaxed RMP optimization time:%f and Relaxed RMP obj:%f.5"%(opt_time,r_m.prob.ObjVal))
    

    for i in range(num_homes):
        optimal_point=0
        cols_in_use=np.where(p_in_use[i]!=-1)[0]
        home_lambda=r_m.lambdas[i]
        home_extreme_point=r_m.extreme_points[i]
        for j in cols_in_use:
            optimal_point+=home_lambda[j].X*home_extreme_point[j]
        
        if np.sum(optimal_point[6*horizon:7*horizon]>0) >4: # here >4 because wm
                                                            #remains active for 4 periods.
            for j in cols_in_use:
                r_m.lambdas[i][j].Vtype=GRB.BINARY
        elif np.sum(optimal_point[7*horizon:8*horizon]>0) >4:
            for j in cols_in_use:
                r_m.lambdas[i][j].Vtype=GRB.BINARY
        elif np.sum(optimal_point[8*horizon:9*horizon]>0) >4:
            for j in cols_in_use:
                r_m.lambdas[i][j].Vtype=GRB.BINARY
        else:
            for j in cols_in_use:
                r_m.lambdas[i][j].ub=r_m.lambdas[i][j].X
                r_m.lambdas[i][j].lb=r_m.lambdas[i][j].X
            

              
    #for i in range(num_homes):
    #    cols_in_use=np.where(p_in_use[i]!=-1)[0]
    #    for j in cols_in_use:
    #        #r_m.lambdas[i][j].set(GRB.CharAttr.VType, GRB.BINARY)
    #        r_m.lambdas[i][j].Vtype=GRB.BINARY
    
    r_m_int_start=datetime.now()
    r_m.prob.update()
    r_m.prob.Params.MIPGap = MIPGap
    r_m.prob.optimize()
    r_m_int_end=datetime.now()
    r_m_int_time=(r_m_int_end-r_m_int_start).seconds+\
                (r_m_int_end-r_m_int_start).microseconds*1e-6
    #print("can")
    
    print("I-RMP-fixed optimization time:%f and I-RMP obj:%f.5"%(r_m_int_time,r_m.prob.ObjVal))
        
    
    
    
    
            
    
    real_power_list=[] #129.145
    deviation_power_list=[]
    home_dev_cost=[]
    for i in range (num_homes):
        #home_weak_duality_epsilon.append(epsilon[i].X)
        #real power levels according to price
        P_ewh_a=np.zeros(horizon)
        P_ev_a=np.zeros(horizon)
        P_hvac_a=np.zeros(horizon)
        P_oven_a=np.zeros(horizon)
        P_wm_a=np.zeros(horizon)
        P_dryer_a=np.zeros(horizon)
        P_pv_a=np.zeros(horizon)
        #P_refrigerator_a=np.zeros(horizon)
        
        
        #deviations from desirable power level.
        P_ewh_d=np.zeros(horizon)
        P_ev_d=np.zeros(horizon)
        P_hvac_d=np.zeros(horizon)
        P_oven_d=np.zeros(horizon)
        P_wm_d=np.zeros(horizon)
        P_dryer_d=np.zeros(horizon)
        
        home_lambda=r_m.lambdas[i]
        home_extreme_point=r_m.extreme_points[i]
        
        optimal_point=0
        #for j in range(len(home_lambda)):
        #    optimal_point+=home_lambda[j].X*home_extreme_point[j]
        cols_in_use=np.where(p_in_use[i]!=-1)[0]
        for j in cols_in_use:
            optimal_point+=home_lambda[j].X*home_extreme_point[j]
        
        
        P_wm_a=optimal_point[6*horizon:7*horizon]
        P_oven_a=optimal_point[7*horizon:8*horizon]
        P_dryer_a=optimal_point[8*horizon:9*horizon]
        P_hvac_a=optimal_point[9*horizon:10*horizon]*power_HVAC_list[i]
        P_ewh_a=optimal_point[10*horizon:11*horizon]
        P_ev_a=optimal_point[11*horizon:12*horizon]
        P_pv_a=optimal_point[12*horizon:13*horizon]
        
        
        dev_start_ind=13*horizon+(horizon+1)+\
                        horizon+\
                        (horizon+1)+\
                        horizon+\
                        3*horizon+1+\
                        2*horizon+1
                        
        P_wm_d=optimal_point[(dev_start_ind+0*horizon):(dev_start_ind+1*horizon)]
        P_oven_d=optimal_point[(dev_start_ind+1*horizon):(dev_start_ind+2*horizon)]
        P_dryer_d=optimal_point[(dev_start_ind+2*horizon):(dev_start_ind+3*horizon)]
        P_hvac_d=optimal_point[(dev_start_ind+3*horizon):(dev_start_ind+4*horizon)]
        P_ewh_d=optimal_point[(dev_start_ind+4*horizon):(dev_start_ind+5*horizon)]
        P_ev_d=optimal_point[(dev_start_ind+5*horizon):(dev_start_ind+6*horizon)]
        
        
        real_power={'ewh':P_ewh_a,
         'ev':P_ev_a,
         'hvac':P_hvac_a,
         'oven':P_oven_a,
         'wm':P_wm_a,
         'dryer':P_dryer_a,
         'pv': P_pv_a}
        
        dev_power={'ewh':P_ewh_d,
         'ev':P_ev_d,
         'hvac':P_hvac_d,
         'oven':P_oven_d,
         'wm':P_wm_d,
         'dryer':P_dryer_d}
        
        cost=dual_list[i]['c']
        cost_dev=cost[:6*horizon]
        
        real_power_list.append(real_power)
        deviation_power_list.append(dev_power)
        home_dev_cost.append(cost_dev)
    
    real_power_list=[]
    deviation_power_list=[]
    real_power_list_before_changing_price=[]
    dev_power_list_before_changing_price=[]
    
    power_summary={'real_ca':real_power_list,
                   'dev_ca':deviation_power_list,
                   'real_before_changing_price':real_power_list_before_changing_price,
                   'dev_before_changing_price':dev_power_list_before_changing_price,
                   'price_lb': price,
                   'Q'        : Q,
                   'c_a_obj_list':r_m.objective,
                   'c_a_final_obj':r_m.prob.ObjVal,
                   'optimization_time':opt_time,
                   'num_homes':num_homes,
                   'home_dev_cost':home_dev_cost,
                   'opt_tolerance':opt_tolerance,
                   'r_m_int_time':r_m_int_time,
                   'total_opt_time':r_m_int_time+opt_time,
                   'lagrangean_dual':lagrangean_dual,
                   'MIPGAP':MIPGap,
                   'calculated_MIPGAP':abs(r_m.prob.ObjVal-r_m.prob.ObjBound)/abs(r_m.prob.ObjVal),
                   'unused_iter_limit':unused_iter_limit}
    return power_summary

            
 
def main():
    
    start_time = datetime.now()
    
    parser = create_train_parser()
    args = parser.parse_args()

    inputs_dict = gather_inputs(args)
    
    seeds = np.random.SeedSequence(args.seed).generate_state(3)
    setup_seeds = np.random.SeedSequence(seeds[0]).generate_state(
     args.runs+args.runs_start)[args.runs_start:]
    
    
    inputs_list = []
    
    for run in range(args.runs):
        setup_dict = dict()
        setup_dict['idx'] = run + args.runs_start
        if args.setup_seed is None:
            setup_dict['setup_seed'] = int(setup_seeds[run])

        inputs_dict['setup_kwargs'] = setup_dict
        inputs_list.append(copy.deepcopy(inputs_dict))

    if args.cores is None:
        args.cores = args.runs

    power_summary=train(inputs_list[0])
    
    
    
    #with mp.get_context('spawn').Pool(args.cores) as pool:
    #    power_summary_list = pool.map(train,inputs_list)
    
    
    

    """
    #creates a folder named as logs
    os.makedirs("/Users/can/Desktop/logs",exist_ok=True)
    #names the file name
    save_file = "deneme"
    save_filefull = os.path.join("/Users/can/Desktop/logs",save_file)
    """
    
    #os.makedirs("/home/erhan/energy/logs",exist_ok=True)
    os.makedirs(args.save_path,exist_ok=True)
    
    save_date=datetime.today().strftime('%m%d%y_%H%M%S')
    
    if args.save_file is None:
        save_file = '%s_%s_%s_%s_%s_%s_%s'%(args.num_houses,args.horizon,
            args.price,args.Q,args.lambda_gap,args.mipgap,save_date)
    else:
        save_file = '%s_%s'%(args.save_file,save_date)
    
    #save_filefull = os.path.join("/home/erhan/energy/logs",save_file)
    save_filefull = os.path.join(args.save_path,save_file)
    

    with open(save_filefull,'wb') as f:
        pickle.dump(power_summary,f)

    ########
    
    end_time = datetime.now()
    
    print('Time Elapsed: %s'%(end_time-start_time))
    
    
    

    

if __name__=='__main__':
    main()
    
    