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
    opt_tolerance=inputs_dict['ca_kwargs']['opt_tolerance']
    
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
    ##Coordination Agent Problem Initialization 
    m_c_a=Model("m_c_a")
    #price_e=m_c_a.addVars(horizon,lb=0,name="price")
    #price_lb=m_c_a.addConstrs((price_e[i]-price[i]>=0
    #                                      for i in range(horizon)),name='price_lower_bounds')
    #price_ub=m_c_a.addConstrs((price_e[i]-price[i]<=p_ub
    #                                      for i in range(horizon)),name='price_upper_bounds')
    #epsilon=m_c_a.addVars(num_homes,lb=0,name="epsilon_home")

    d_obj_list=[]
    d_obj_exp_list=[]
    
    problem_formulation_start_time = datetime.now()
    #For loop below prepares the constraints of m_c_a problem.
    for i in range(num_homes):
        
        p_m=models[i]
        #dual=dual_list[i]
        #d_m,d_obj=solve_dual(dual)
        
        
        tmp_p_name="H"+str(i+1)+"_P_"
        tmp_d_name="H"+str(i+1)+"_D_"
        
        for v in p_m.getVars():
            v.varname=tmp_p_name+v.varname
        p_m.update()
            
        #for v in d_m.getVars():
        #    v.varname=tmp_d_name+v.varname
        #d_m.update()
        
        for v in p_m.getVars():
            m_c_a.addVar(lb=v.lb, ub=v.ub, vtype=v.vtype, name=v.varname)
        #for v in d_m.getVars():
        #    m_c_a.addVar(lb=v.lb, ub=v.ub, vtype=v.vtype, name=v.varname)
        m_c_a.update()
        
        
        for c in p_m.getConstrs():
            expr = p_m.getRow(c)
            newexpr = LinExpr()
            for j in range(expr.size()):
                v = expr.getVar(j)
                coeff = expr.getCoeff(j)
                newv = m_c_a.getVarByName(v.Varname)
                newexpr.add(newv, coeff)
                
            m_c_a.addConstr(newexpr, c.Sense, c.RHS, name=tmp_p_name+c.ConstrName)
            
        
        # ind=0
        # for c in d_m.getConstrs():
        #     #print(c)
        #     expr = d_m.getRow(c)
        #     #print(expr)
        #     newexpr = LinExpr()
        #     for j in range(expr.size()):
        #         v = expr.getVar(j)
        #         coeff = expr.getCoeff(j)
        #         newv = m_c_a.getVarByName(v.Varname)
        #         newexpr.add(newv, coeff)
        #     """
        #     This if-else blog handles the dual constraints note that rhs in duals must be decision variable!
        #     """
        #     if (ind <= horizon*6-1) or (ind >= horizon*13): # horizon is 96. 
        #     #We have 6 because deviations in 6 appliances. We have 13 because 6+7 where 7 real power decision.
        #         m_c_a.addConstr(newexpr, c.Sense, c.RHS, name=tmp_d_name+c.ConstrName)
        #     else:
        #         """
        #         If else below handles the cost modification of HVAC. Please see check HOMEMPC and observe
        #         that we multiply price by power_HVAC
        #         """
        #         #print("cntrl")
        #         if (ind >= 9*horizon) and (ind <= horizon*10-1):
        #             newexpr.add(price_e[ind%horizon], -1*power_HVAC_list[i])
        #         elif (ind >= 12*horizon):
        #             newexpr.add(price_e[ind%horizon], +1)
        #         else:
        #             newexpr.add(price_e[ind%horizon], -1)
        #         #m_c_a.addConstr(newexpr, c.Sense, c.RHS, name=tmp_d_name+c.ConstrName)
        #         m_c_a.addConstr(newexpr, c.Sense, 0.0, name=tmp_d_name+c.ConstrName)
        #     ind+=1
                
            
        m_c_a.update()
        
        #
        
        # newexpr_d = LinExpr()
        # expr=d_obj
        # for j in range(expr.size()):
        #     v = expr.getVar(j)

        #     coeff = expr.getCoeff(j)
        #     newv = m_c_a.getVarByName(v.Varname)
        #     newexpr_d.add(newv, coeff)
        # d_obj_exp_list.append(newexpr_d)
        # m_c_a.update()
        
    
    home_devs=[]
    home_reals=[]
    home_signed_devs=[]
    
    for i in range(num_homes):
        names_to_retrieve=[]
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_wm_dev["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_oven_dev["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_dryer_dev["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_hvac_dev["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_ewh_dev["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_ev_dev["+str(j)+"]")
            
        home_devs.append([m_c_a.getVarByName(name) for name in names_to_retrieve])
        
        
        names_to_retrieve=[]
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_wm_signed_dev["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_oven_signed_dev["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_dryer_signed_dev["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_hvac_signed_dev["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_ewh_signed_dev["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_ev_signed_dev["+str(j)+"]")
            
        home_signed_devs.append([m_c_a.getVarByName(name) for name in names_to_retrieve])
        
        
        names_to_retrieve=[]
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_wm_real["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_oven_real["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_dryer_real["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_hvac_real["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_ewh_real["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_ev_real["+str(j)+"]")
        for j in range(horizon):
            names_to_retrieve.append("H"+str(i+1)+"_P_pv_real["+str(j)+"]")
        
        home_reals.append([m_c_a.getVarByName(name) for name in names_to_retrieve])
        
        
    #Note that primal objective function is somewhat different 
    #at the following indices. Let' s keep record of them.
    real_pow_idx=np.arange(len(home_reals[0]))
    real_pow_HVAC_idx=np.arange(3*horizon,4*horizon,1)
    real_pow_PV_idx=np.arange(6*horizon,7*horizon,1)

    real_pow_idx_remaining=np.setdiff1d(real_pow_idx,real_pow_HVAC_idx)
    real_pow_idx_remaining=np.setdiff1d(real_pow_idx_remaining,real_pow_PV_idx)
    
    # #this for loop add constraints related to primal objective == dual objective
    # for i in range(num_homes):
        

            
    #     cost=dual_list[i]['c']
    #     cost_dev=cost[:6*horizon]
        

    #     # m_c_a.addConstr(quicksum(home_devs[i][t]*cost_dev[t] for t in range(len(cost_dev))) +\
    #     #                 quicksum(home_reals[i][t]*price_e[t%horizon] for t in real_pow_idx_remaining) +\
    #     #                 quicksum(home_reals[i][t]*power_HVAC_list[i]*price_e[t%horizon] for t in real_pow_HVAC_idx) +\
    #     #                 quicksum(home_reals[i][t]*-price_e[t%horizon] for t in real_pow_PV_idx ) -\
    #     #                 d_obj_exp_list[i] == 0,name="H_"+str(i+1)+"_objective")  
            
    #     m_c_a.addConstr(quicksum(home_devs[i][t]*cost_dev[t] for t in range(len(cost_dev))) +\
    #                     quicksum(home_reals[i][t]*price_e[t%horizon] for t in real_pow_idx_remaining) +\
    #                     quicksum(home_reals[i][t]*power_HVAC_list[i]*price_e[t%horizon] for t in real_pow_HVAC_idx) +\
    #                     quicksum(home_reals[i][t]*-price_e[t%horizon] for t in real_pow_PV_idx ) -\
    #                     d_obj_exp_list[i] <= epsilon[i],name="H_"+str(i+1)+"_objective") 
                
    #Set objective function of coordination agent
    #deviation_loss=m_c_a.addVars(horizon,lb=-GRB.INFINITY,ub=GRB.INFINITY,name="dev_loss")
    dev_loss_helper=m_c_a.addVars(horizon,lb=0,name="dev_loss_obj")
    
    #obj_term1=[]#holds the deviation from desired power consumption level.
    for j in range(horizon):
        #obj_term1=Q[j]-quicksum(home_reals[i][t*horizon+j] for i in range(num_homes)for t in range(7))
        
        real_pow_idx_remaining_sel=real_pow_idx_remaining[(real_pow_idx_remaining%horizon)==j]
        obj_term1=Q[j]-(quicksum(home_reals[i][t] for i in range(num_homes)for t in real_pow_idx_remaining_sel)+\
                        quicksum(home_reals[i][real_pow_HVAC_idx[j]]*power_HVAC_list[i] for i in range(num_homes)))
                        
        m_c_a.addConstr(dev_loss_helper[j] >= obj_term1)
        m_c_a.addConstr(dev_loss_helper[j] >= -obj_term1)

    #m_c_a.addConstrs((dev_loss_helper[j]==abs_(deviation_loss[j]) for j in range(horizon)),name="tmp")
    
    
    obj_term2=[]#list holding the cost related to home deviation.
    for i in range(num_homes):
        cost=dual_list[i]['c']
        cost_dev=cost[:6*horizon]
        obj_term2.append(quicksum(home_devs[i][t]*cost_dev[t] for t in range(len(cost_dev))))
        
    
    
    
    #m_c_a.setObjective(quicksum(dev_loss_helper[i] for i in range(horizon))+\
    #                   quicksum(obj_term2[i] for i in range(num_homes))+\
    #                   quicksum(epsilon[i]*lambda_gap for i in range(num_homes))    ,GRB.MINIMIZE) 
    
    m_c_a.setObjective(quicksum(dev_loss_helper[i] for i in range(horizon))+\
                       quicksum(obj_term2[i] for i in range(num_homes))    ,GRB.MINIMIZE) 
        
        
    
    #-1 automatic 0 primal 1 dual 2 barrier
    #m_c_a.Params.Method=1
    m_c_a.Params.OptimalityTol=opt_tolerance
    m_c_a.Params.Threads=4
    #m_c_a.Params.Presolve=0
    #m_c_a.Params.NonConvex=2
    m_c_a.Params.MIPGap = MIPGap
    m_c_a.Params.TimeLimit = TimeLimit
    problem_formulation_end_time = datetime.now()
    opt_start_time = datetime.now()
    m_c_a.optimize()
    opt_end_time = datetime.now()#137.639 #110.12322216954293
    
    #keep track of c.a. objective.
    c_a_obj_list=[]
    c_a_obj_list.append(m_c_a.objVal)
    
    
    real_power_list= []
    deviation_power_list= []
    desired_power_list=[]
    home_dev_cost=[]
    #home_weak_duality_epsilon=[]
    
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
        
        
        
        for k in range (horizon):
            P_wm_a[k]=home_reals[i][0*horizon+k].X
            P_oven_a[k]=home_reals[i][1*horizon+k].X
            P_dryer_a[k]=home_reals[i][2*horizon+k].X
            P_hvac_a[k]=home_reals[i][3*horizon+k].X *power_HVAC_list[i]#HVAC variable is relaxed binary. Need to have a multiplication.
            #P_hvac_a[k]=home_reals[i][3*horizon+k].X#HVAC variable is relaxed binary. Need to have a multiplication.
            P_ewh_a[k]=home_reals[i][4*horizon+k].X
            P_ev_a[k]=home_reals[i][5*horizon+k].X
            P_pv_a[k]=home_reals[i][6*horizon+k].X
            
            
            P_wm_d[k]=home_signed_devs[i][0*horizon+k].X
            P_oven_d[k]=home_signed_devs[i][1*horizon+k].X
            P_dryer_d[k]=home_signed_devs[i][2*horizon+k].X
            P_hvac_d[k]=home_signed_devs[i][3*horizon+k].X#deviation variable is unbounded. No need to have a multiplication.
            P_ewh_d[k]=home_signed_devs[i][4*horizon+k].X
            P_ev_d[k]=home_signed_devs[i][5*horizon+k].X
            
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
        
        des_power={'ewh':P_ewh_a-P_ewh_d,
         'ev':P_ev_a-P_ev_d,
         'hvac':P_hvac_a*power_HVAC_list[i]-P_hvac_d,
         'oven':P_oven_a-P_oven_d,
         'wm':P_wm_a-P_wm_d,
         'dryer':P_dryer_a-P_dryer_d}
        
        
        cost=dual_list[i]['c']
        cost_dev=cost[:6*horizon]
        
        
        real_power_list.append(real_power)
        deviation_power_list.append(dev_power)
        desired_power_list.append(des_power)  
        home_dev_cost.append(cost_dev)
    

    power_summary={'real_ca':real_power_list,
                   'dev_ca':deviation_power_list,
                   'real_before_changing_price':real_power_list_before_changing_price,
                   'dev_before_changing_price':dev_power_list_before_changing_price,
                   'price_lb': price,
                   'Q'        : Q,
                   'optimality_gap':"NA",
                   'c_a_obj_list':c_a_obj_list,
                   'optimization_time':opt_end_time-opt_start_time,
                   'problem_formulation_time':problem_formulation_end_time-problem_formulation_start_time,
                   'home_lambda_gap':lambda_gap,
                   'home_dev_cost':home_dev_cost,
                   'opt_tolerance':opt_tolerance,
                   'MIPGap':MIPGap,
                   'calculated_MIPGAP':abs(m_c_a.ObjVal-m_c_a.ObjBound)/abs(m_c_a.ObjVal)}
    
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
    
    