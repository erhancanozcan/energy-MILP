import numpy as np
import pandas as pd
from gurobipy import *


from energy.code.appliances.ewh import ewh
from energy.code.appliances.ev import ev
from energy.code.appliances.hvac import hvac
from energy.code.appliances.PV import pv
from energy.code.appliances.other_appliances import refrigerator
from energy.code.appliances.other_appliances import oven
from energy.code.appliances.other_appliances import wm
from energy.code.appliances.other_appliances import dryer




class Home:
    def __init__(self,time_res=15,s_effect=1):
        """TODO:
                Need to include solar panel!
        """
        
        
        
        """
            Demands
            Home must have demand for appliances.
        TODO:
            Demands will be updated with the help of outside function.
        """
        self.ewh_demand=None#water demand per interval in terms of kg. (array)
        self.ev_demand=None#demand per interval in terms of miles.  (array)
        self.set_temperature=None#desired temperature in terms of celcius per interval.
        self.deadband=None#temperature comfort range (number)
        self.refrigerator_demand=None#on/off status per interval. (array)
        self.oven_demand=None#on/off status per interval.
        self.wm_demand=None#on/off status per interval.
        self.dryer_demand=None#on/off status per interval.

        
        
        """
            Appliances
            Home must have appliances.
        TODO:
            The properties of appliances will be set by an outside function.
        """
        self.ewh=ewh()
        self.ev=ev()
        self.hvac=hvac()
        self.refrigerator=refrigerator()
        self.oven=oven()
        self.wm=wm()
        self.dryer=dryer()
        self.pv=pv(s_effect=s_effect)
        
        
        """
            Some appliances must have instantenous information.
            ev_battery    : electric vehicle charged percentage
            water_amount  : available hot water
            inside_temp   : inside temperature.
            tap_water_temp: depends on heating season
            des_water_temp: desired water temperature depends on home.
            ev_current    : customer preference to charge current.
            outside_temp  : outside temperature for HVAC
            gamma1        :
            gamma2        :
            s_effect      : 1 during winter, -1 during winter.
        TODO:
            1.These features must be initialized.
                Initialize while setting appliances` properties.
                Initial values: They are full or at desired level.
            2.Tap water temperature must change from season to season.
        """
        self.ev_battery=None
        self.water_amount=None
        self.init_temp=None
        self.tap_water_temp=None
        self.des_water_temp=None
        self.ev_current=None
        self.outside_temp=None
        self.gamma1=None
        self.gamma2=None
        #self.s_effect=None
        self.s_effect=s_effect

    
    def generate_desirable_load(self):
        
        self.ewh_demand,self.ewh_desirable_load=self.ewh.load_curve(self.ewh_demand,self.tap_water_temp,self.des_water_temp)
        self.ev_demand,self.ev_desirable_load=self.ev.load_curve(self.ev_demand,self.ev_current)
        self.hvac_desirable_load,self.neg_devs,self.pos_devs=self.hvac.load_curve(self.set_temperature,self.deadband,self.outside_temp,self.init_temp,self.gamma1,self.gamma2)
        self.refrigerator_desirable_load=self.refrigerator.load_curve(self.refrigerator_demand)
        self.oven_desirable_load=self.oven.load_curve(self.oven_demand)
        self.wm_desirable_load=self.wm.load_curve(self.wm_demand)
        self.dryer_desirable_load=self.dryer.load_curve(self.dryer_demand)
        
    def total_desirable_load(self,price,normalize):
        total=self.ewh_desirable_load+self.ev_desirable_load+\
            self.hvac_desirable_load+self.refrigerator_desirable_load+\
            self.oven_desirable_load+self.wm_desirable_load+\
            self.dryer_desirable_load
            
            
        daily_fee=(np.dot(total,price))*(self.wm.time_r/60)#to convert kwh in 15minutes
        """
        TODO
            daily_budget must be a home specific parameter.
            discuss cost_u calculation deviation.
        """
        daily_budget=15
        if self.s_effect==-1:
            daily_budget=10
        
        if daily_fee <= daily_budget:
            #in this case, home does not have any incentive to deviate from desirable load.
            cost_u={'ewh':1000.0,
                    'ev':1000.0,
                    'hvac':1000.0,
                    'oven':1000.0,
                    'wm':1000.0,
                    'dryer':1000.0}
        else:
            correction_amount=abs(daily_fee-daily_budget)/(normalize*100)
            #correction_amount=abs(daily_fee-daily_budget)
            
            total_ewh=np.sum(self.ewh_desirable_load)
            total_ev=np.sum(self.ev_desirable_load)
            total_hvac=np.sum(self.hvac_desirable_load)
            total_refrigerator=np.sum(self.refrigerator_desirable_load)
            total_oven=np.sum(self.oven_desirable_load)
            total_wm=np.sum(self.wm_desirable_load)
            total_dryer=np.sum(self.dryer_desirable_load)
            
            tmp_total=np.sum(total)
            cost_u={'ewh':correction_amount*(total_ewh/tmp_total),
                    'ev':correction_amount*(total_ev/tmp_total),
                    'hvac':correction_amount*(total_hvac/tmp_total),
                    'oven':correction_amount*(total_oven/tmp_total),
                    'wm':correction_amount*(total_wm/tmp_total),
                    'dryer':correction_amount*(total_dryer/tmp_total)}
        
        return total,cost_u,daily_fee
            
            
        
        

    def optimize_mpc(self,cost_u,price):
        """
        Args:
            cost_u: is a dictionary. cost of deviation from desireable curve for each appliance.
            price : shows the price of electricity in the next 24 hours. (1*96 array.)
        """
        #jj=0
        #tmp_name="H"+str(jj+1)+"_P"
        cost_u_ewh=cost_u['ewh']
        cost_u_ev=cost_u['ev']
        cost_u_hvac=cost_u['hvac']
        cost_u_oven=cost_u['oven']
        cost_u_wm=cost_u['wm']
        cost_u_dryer=cost_u['dryer']
        
        horizon=len(self.wm_desirable_load)
        m = Model("home_mpc")
        
        obj_ewh_price=0.0
        obj_ewh_dev=0.0
        obj_ev_price=0.0
        obj_ev_dev=0.0
        obj_hvac_price=0.0
        obj_hvac_dev=0.0
        obj_oven_price=0.0
        obj_oven_dev=0.0
        obj_wm_price=0.0
        obj_wm_dev=0.0
        obj_dryer_price=0.0
        obj_dryer_dev=0.0
        
        """
        POSSIBLE TODO
            Currently, we assume all devices are scheduled in one day. 
            What if we do not schedule one of those appliances.
            optimize_mpc function before March 7, handles this. However, this
            feature is sacrificed in the last update published on March 7.
            You may need to think about this!.
        """
        
        
        
        #necessary parameters
            #max nominal power
        power_wm=self.wm.nominal_power
        power_oven=self.oven.nominal_power
        power_dryer=self.dryer.nominal_power
        power_HVAC=self.hvac.nominal_power
        power_ewh=self.ewh.nominal_power
        power_ev=self.ev.nominal_power
            #max capacity
        capacity_ewh=self.ewh.capacity
        capacity_ev=self.ev.capacity
            #deviations
        x=0.5 #HVAC related todo!
        #     #demand
            
        #     #ewh
            
        #     #ev
        """
        TODO
            Note that some variables are free. Dual constraints must have equality signs!
        """
        sign_of_dual_variable=[]
        sign_of_dual_constraint=[]
        num_variables=0
        num_helper_variables=0
        #decision variables
            #deviation variables
        y_wm=m.addVars(horizon,lb=0,name="wm_dev")
        y_oven=m.addVars(horizon,lb=0,name="oven_dev")
        y_dryer=m.addVars(horizon,lb=0,name="dryer_dev")
        y_HVAC=m.addVars(horizon,lb=0,name="hvac_dev")
        y_ewh=m.addVars(horizon,lb=0,name="ewh_dev")
        y_ev=m.addVars(horizon,lb=0,name="ev_dev")
        num_variables+=horizon*6
        
        
            #power variables
        P_wm=m.addVars(horizon,lb=0,name="wm_real")
        P_oven=m.addVars(horizon,lb=0,name="oven_real")
        P_dryer=m.addVars(horizon,lb=0,name="dryer_real")
        P_HVAC=m.addVars(horizon,lb=0,name="hvac_real")
        P_ewh=m.addVars(horizon,lb=0,name="ewh_real")
        P_ev=m.addVars(horizon,lb=0,name="ev_real")
        P_pv=m.addVars(horizon,lb=0,name="pv_real")
        num_variables+=horizon*7
        sign_of_dual_constraint=[-1]*num_variables
            
            #helper variables
                #ewh related
        x_ewh=m.addVars(horizon+1,lb=0)
        z_ewh=m.addVars(horizon,lb=0)
        #y_ewh=m.addVars(horizon,lb=0)
                #ev related
        x_ev=m.addVars(horizon+1,lb=0)
        i_ev=m.addVars(horizon,lb=0)
                #HVAC related
        s_HVAC_neg=m.addVars(horizon,lb=0)
        s_HVAC_pos=m.addVars(horizon,lb=0)
        T_in=m.addVars(horizon+1,lb=-GRB.INFINITY)
        """
        add T_set as decision variable to govern on/off of HVAC.
        """
        T_set=m.addVars(horizon,lb=-GRB.INFINITY)
                #pv related
        x_pv=m.addVars(horizon+1,lb=0)
        sign_of_dual_constraint.extend([-1]*(6*horizon+2))
        sign_of_dual_constraint.extend([0]*(2*horizon+1))
        sign_of_dual_constraint.extend([-1]*(1*horizon+1))
                #other deviations
        u_wm=m.addVars(horizon,lb=-GRB.INFINITY,name="wm_signed_dev")
        u_oven=m.addVars(horizon,lb=-GRB.INFINITY,name="oven_signed_dev")
        u_dryer=m.addVars(horizon,lb=-GRB.INFINITY,name="dryer_signed_dev")
        u_HVAC=m.addVars(horizon,lb=-GRB.INFINITY,name="hvac_signed_dev")
        u_ewh=m.addVars(horizon,lb=-GRB.INFINITY,name="ewh_signed_dev")
        u_ev=m.addVars(horizon,lb=-GRB.INFINITY,name="ev_signed_dev")
        
        num_helper_variables+=horizon*6
        num_helper_variables+=horizon*5+(horizon+1)*4
        num_variables+=num_helper_variables
        sign_of_dual_constraint.extend([0]*(6*horizon))
        
        #integer programming
        wm_on_off=m.addVars(horizon,vtype=GRB.BINARY,name="wm_on_off")
        wm_start=m.addVars(horizon,vtype=GRB.BINARY,name="wm_start")
        wm_shut=m.addVars(horizon,vtype=GRB.BINARY,name="wm_shut")
        
        oven_on_off=m.addVars(horizon,vtype=GRB.BINARY,name="oven_on_off")
        oven_start=m.addVars(horizon,vtype=GRB.BINARY,name="oven_start")
        oven_shut=m.addVars(horizon,vtype=GRB.BINARY,name="oven_shut")
        
                
        dryer_on_off=m.addVars(horizon,vtype=GRB.BINARY,name="dryer_on_off")
        dryer_start=m.addVars(horizon,vtype=GRB.BINARY,name="dryer_start")
        dryer_shut=m.addVars(horizon,vtype=GRB.BINARY,name="dryer_shut")
        

        
        
        #objective function construction
        #first deviations then consumption related fee.
        
        p_obj=quicksum(y_wm[i]*cost_u_wm for i in range(horizon))+\
                       quicksum(y_oven[i]*cost_u_oven for i in range(horizon))+\
                       quicksum(y_dryer[i]*cost_u_dryer for i in range(horizon))+\
                       quicksum(y_HVAC[i]*cost_u_hvac for i in range(horizon))+\
                       quicksum(y_ewh[i]*cost_u_ewh for i in range(horizon))+\
                       quicksum(y_ev[i]*cost_u_ev for i in range(horizon))+\
                       quicksum(P_wm[i]*price[i] for i in range(horizon))+\
                       quicksum(P_oven[i]*price[i] for i in range(horizon))+\
                       quicksum(P_dryer[i]*price[i] for i in range(horizon))+\
                       quicksum(P_HVAC[i]*power_HVAC*price[i] for i in range(horizon))+\
                       quicksum(P_ewh[i]*price[i] for i in range(horizon))+\
                       quicksum(P_ev[i]*price[i] for i in range(horizon))+\
                       quicksum(P_pv[i]*-price[i] for i in range(horizon))
        
        m.setObjective(p_obj,GRB.MINIMIZE)
        """
        m.setObjective(quicksum(y_wm[i]*cost_u_wm for i in range(horizon))+\
                       quicksum(y_oven[i]*cost_u_oven for i in range(horizon))+\
                       quicksum(y_dryer[i]*cost_u_dryer for i in range(horizon))+\
                       quicksum(y_HVAC[i]*cost_u_hvac for i in range(horizon))+\
                       quicksum(y_ewh[i]*cost_u_ewh for i in range(horizon))+\
                       quicksum(y_ev[i]*cost_u_ev for i in range(horizon))+\
                       quicksum(P_wm[i]*price[i] for i in range(horizon))+\
                       quicksum(P_oven[i]*price[i] for i in range(horizon))+\
                       quicksum(P_dryer[i]*price[i] for i in range(horizon))+\
                       quicksum(P_HVAC[i]*power_HVAC*price[i] for i in range(horizon))+\
                       quicksum(P_ewh[i]*price[i] for i in range(horizon))+\
                       quicksum(P_ev[i]*price[i] for i in range(horizon))+\
                       quicksum(P_pv[i]*-price[i] for i in range(horizon))    , GRB.MINIMIZE)
        """
        modified_c=np.array([])
        modified_c=np.concatenate((modified_c,np.repeat(cost_u_wm,horizon)))
        modified_c=np.concatenate((modified_c,np.repeat(cost_u_oven,horizon)))
        modified_c=np.concatenate((modified_c,np.repeat(cost_u_dryer,horizon)))
        modified_c=np.concatenate((modified_c,np.repeat(cost_u_hvac,horizon)))
        modified_c=np.concatenate((modified_c,np.repeat(cost_u_ewh,horizon)))
        modified_c=np.concatenate((modified_c,np.repeat(cost_u_ev,horizon)))
        modified_c=np.concatenate((modified_c,price))#wm
        modified_c=np.concatenate((modified_c,price))#oven
        modified_c=np.concatenate((modified_c,price))#dryer
        modified_c=np.concatenate((modified_c,power_HVAC*price))#hvac
        modified_c=np.concatenate((modified_c,price))#ewh
        modified_c=np.concatenate((modified_c,price))#ev
        modified_c=np.concatenate((modified_c,-price))#pv
        
        modified_c=np.concatenate((modified_c,np.repeat(0,num_helper_variables)))#cost is zero for other vars.
        
        
        

        


        
        
        
        
        
        
        #constraints
        
        #real power upper bounds
        m.addConstrs((P_wm[i]<=power_wm for i in range(horizon)),name='c_wm_realpow_up')
        m.addConstrs((P_oven[i]<=power_oven for i in range(horizon)),name='c_oven_realpow_up')
        m.addConstrs((P_dryer[i]<=power_dryer for i in range(horizon)),name='c_dryer_realpow_up')
        m.addConstrs((P_HVAC[i]<=1 for i in range(horizon)),name='c_HVAC_realpow_up')
        m.addConstrs((P_ewh[i]<=power_ewh for i in range(horizon)),name='c_ewh_realpow_up')
        m.addConstrs((P_ev[i]<=power_ev for i in range(horizon)),name='c_ev_realpow_up')
        sign_of_dual_variable=[-1]*(6*horizon)
        
        #state variable upper bounbds
        m.addConstrs((x_ewh[i]<=capacity_ewh for i in range(horizon+1)),name='c_ewh_water_up')
        m.addConstrs((x_ev[i]<=capacity_ev for i in range(horizon+1)),name='c_ev_battery_up')
        m.addConstrs((i_ev[i]<=self.ev_current for i in range(horizon)),name='c_ev_current_up')
        sign_of_dual_variable.extend([-1]*(3*horizon+2))


        """
        ________________________WM__________________________
        TODO 
            delay_allowance is a parameter that must be determined for each home by a outside preference function
        """
        operation_index_wm=np.where(self.wm_desirable_load>0)[0]
        operation_index_oven=np.where(self.oven_desirable_load>0)[0]
        operation_index_dryer=np.where(self.dryer_desirable_load>0)[0]
        
        start_index=operation_index_wm[0]
        end_index=operation_index_wm[len(operation_index_wm)-1]
        delay_allowance=1#in terms of hours
        ints_per_hour=int(60/self.wm.time_r)
        start_index=start_index-int(delay_allowance*ints_per_hour)
        end_index=end_index+int(delay_allowance*ints_per_hour)
        K_wm=np.arange(start_index,end_index+1)
        full_index=np.arange(0,horizon)
        K_wm_c=np.setdiff1d(full_index, np.intersect1d(full_index, K_wm))


        #wm constraints
        const_wm_real_pow=m.addConstrs((P_wm[i]-u_wm[i]==self.wm_desirable_load[i]
                                              for i in range(horizon)),name='c_wm_realpow')
        sign_of_dual_variable.extend([0]*(horizon))

        #const_wm_cntrl_low=m.addConstrs((u_wm[i]>=-power
        #                                      for i in K_wm),name='c_wm_cntrl_low')

        #const_wm_cntrl_up=m.addConstrs((u_wm[i]<=power
        #                                      for i in K_wm),name='c_wm_cntrl_up')

        const_wm_cntrl_zero=m.addConstrs((u_wm[i]==0
                                              for i in K_wm_c),name='c_wm_cntrl_zero')
        sign_of_dual_variable.extend([0]*(len(K_wm_c)))

        const_wm_balance=m.addConstr((quicksum(u_wm[i]*1 for i in range(horizon))==0),name='c_wm_balance')
        sign_of_dual_variable.extend([0]*(1))
        


        
        const_wm_abs_value_pos=m.addConstrs((u_wm[i]-y_wm[i]<=0
                                              for i in range(horizon)),name='c_wm_abs_value_pos')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        const_wm_abs_value_neg=m.addConstrs((-u_wm[i]-y_wm[i]<=0
                                              for i in range(horizon)),name='c_wm_abs_value_neg')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        
        #integer programming
        
        wm_on_off_eq_six=m.addConstrs((wm_on_off[i+1]-wm_on_off[i]-wm_start[i+1]+wm_shut[i+1]==0
                                              for i in range(horizon-1)),name='c_wm_on_off_eq_six')
        
        wm_on_off_eq_seven=m.addConstrs((wm_start[i+1]-wm_on_off[i+1]<=0
                                              for i in range(horizon-1)),name='c_wm_on_off_eq_seven')
        
        wm_on_off_eq_eight=m.addConstrs((wm_shut[i+1]+wm_on_off[i+1]<=1
                                              for i in range(horizon-1)),name='c_wm_on_off_eq_eight')
        
        wm_start_once=m.addConstr((quicksum(wm_start[i] for i in range(horizon))<=1
                                              ),name='c_wm_start_once')
        
        wm_shut_once=m.addConstr((quicksum(wm_shut[i] for i in range(horizon))<=1
                                              ),name='c_wm_shut_once')
        
        wm_on_power=m.addConstrs((wm_on_off[i]*self.wm.nominal_power==P_wm[i]
                                              for i in range(horizon)),name='c_wm_on_power')

        
        
        
        
        """
        ________________________Oven__________________________
        TODO 
            delay_allowance is a parameter that must be determined for each home by a outside preference function
        """
        
        start_index=operation_index_oven[0]
        end_index=operation_index_oven[len(operation_index_oven)-1]
        delay_allowance=1#in terms of hours
        ints_per_hour=int(60/self.wm.time_r)
        start_index=start_index-int(delay_allowance*ints_per_hour)
        end_index=end_index+int(delay_allowance*ints_per_hour)
        K_oven=np.arange(start_index,end_index+1)
        full_index=np.arange(0,horizon)
        K_oven_c=np.setdiff1d(full_index, np.intersect1d(full_index, K_oven))



        #oven constraints
        const_oven_real_pow=m.addConstrs((P_oven[i]-u_oven[i]==self.oven_desirable_load[i]
                                              for i in range(horizon)),name='c_oven_realpow')
        sign_of_dual_variable.extend([0]*(horizon))

        #const_oven_cntrl_low=m.addConstrs((u_oven[i]>=-power
        #                                      for i in K_oven),name='c_oven_cntrl_low')

        #const_oven_cntrl_up=m.addConstrs((u_oven[i]<=power
        #                                      for i in K_oven),name='c_oven_cntrl_up')

        const_oven_cntrl_zero=m.addConstrs((u_oven[i]==0
                                              for i in K_oven_c),name='c_oven_cntrl_zero')
        sign_of_dual_variable.extend([0]*(len(K_oven_c)))

        const_oven_balance=m.addConstr((quicksum(u_oven[i]*1 for i in range(horizon))==0),name='c_oven_balance')
        sign_of_dual_variable.extend([0]*(1))
          
        const_oven_abs_value_pos=m.addConstrs((u_oven[i]-y_oven[i]<=0
                                              for i in range(horizon)),name='c_oven_abs_value_pos')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        const_oven_abs_value_neg=m.addConstrs((-u_oven[i]-y_oven[i]<=0
                                              for i in range(horizon)),name='c_oven_abs_value_neg')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        #integer programming
        
        oven_on_off_eq_six=m.addConstrs((oven_on_off[i+1]-oven_on_off[i]-oven_start[i+1]+oven_shut[i+1]==0
                                              for i in range(horizon-1)),name='c_oven_on_off_eq_six')
        
        oven_on_off_eq_seven=m.addConstrs((oven_start[i+1]-oven_on_off[i+1]<=0
                                              for i in range(horizon-1)),name='c_oven_on_off_eq_seven')
        
        oven_on_off_eq_eight=m.addConstrs((oven_shut[i+1]+oven_on_off[i+1]<=1
                                              for i in range(horizon-1)),name='c_oven_on_off_eq_eight')
        
        oven_start_once=m.addConstr((quicksum(oven_start[i] for i in range(horizon))<=1
                                              ),name='c_oven_start_once')
        
        oven_shut_once=m.addConstr((quicksum(oven_shut[i] for i in range(horizon))<=1
                                              ),name='c_oven_shut_once')
        
        oven_on_power=m.addConstrs((oven_on_off[i]*self.oven.nominal_power==P_oven[i]
                                              for i in range(horizon)),name='c_oven_on_power')
        
        
        
        
        """
        ________________________Dryer__________________________
        TODO 
            delay_allowance is a parameter that must be determined for each home by a outside preference function
        """
        start_index=operation_index_dryer[0]
        end_index=operation_index_dryer[len(operation_index_dryer)-1]
        delay_allowance=1#in terms of hours
        ints_per_hour=int(60/self.wm.time_r)
        start_index=start_index-int(delay_allowance*ints_per_hour)
        end_index=end_index+int(delay_allowance*ints_per_hour)
        K_dryer=np.arange(start_index,end_index+1)
        full_index=np.arange(0,horizon)
        K_dryer_c=np.setdiff1d(full_index, np.intersect1d(full_index, K_dryer))



        #wm constraints
        const_dryer_real_pow=m.addConstrs((P_dryer[i]-u_dryer[i]==self.dryer_desirable_load[i]
                                              for i in range(horizon)),name='c_dryer_realpow')
        sign_of_dual_variable.extend([0]*(horizon))


        const_dryer_cntrl_zero=m.addConstrs((u_dryer[i]==0
                                              for i in K_dryer_c),name='c_dryer_cntrl_zero')
        sign_of_dual_variable.extend([0]*(len(K_dryer_c)))

        const_dryer_balance=m.addConstr((quicksum(u_dryer[i]*1 for i in range(horizon))==0),name='c_dryer_balance')
        sign_of_dual_variable.extend([0]*(1))
        
                    
        const_dryer_abs_value_pos=m.addConstrs((u_dryer[i]-y_dryer[i]<=0
                                              for i in range(horizon)),name='c_dryer_abs_value_pos')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        const_dryer_abs_value_neg=m.addConstrs((-u_dryer[i]-y_dryer[i]<=0
                                              for i in range(horizon)),name='c_dryer_abs_value_neg')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        #integer programming
        
        dryer_on_off_eq_six=m.addConstrs((dryer_on_off[i+1]-dryer_on_off[i]-dryer_start[i+1]+dryer_shut[i+1]==0
                                              for i in range(horizon-1)),name='c_dryer_on_off_eq_six')
        
        dryer_on_off_eq_seven=m.addConstrs((dryer_start[i+1]-dryer_on_off[i+1]<=0
                                              for i in range(horizon-1)),name='c_dryer_on_off_eq_seven')
        
        dryer_on_off_eq_eight=m.addConstrs((dryer_shut[i+1]+dryer_on_off[i+1]<=1
                                              for i in range(horizon-1)),name='c_dryer_on_off_eq_eight')
        
        dryer_start_once=m.addConstr((quicksum(dryer_start[i] for i in range(horizon))<=1
                                              ),name='c_dryer_start_once')
        
        dryer_shut_once=m.addConstr((quicksum(dryer_shut[i] for i in range(horizon))<=1
                                              ),name='c_dryer_shut_once')
        
        dryer_on_power=m.addConstrs((dryer_on_off[i]*self.dryer.nominal_power==P_dryer[i]
                                              for i in range(horizon)),name='c_dryer_on_power')
        
        
        """
        ________________________HVAC__________________________
        TODO 
            x is a parameter that must be determined for each home by a outside preference function
        """
        
        
        
        
        #hvac constraints
        const_temp_change=m.addConstrs((T_in[i+1]==T_in[i]+self.gamma1*(self.hvac.T_out[i]-T_in[i])+self.gamma2*P_HVAC[i]*power_HVAC*self.hvac.efficiency*(1000*60*self.hvac.time_r)*self.hvac.s_effect
                                              for i in range(horizon)),name='c_hvac_temp_chng')
        sign_of_dual_variable.extend([0]*(horizon))


        const_temp_cntrl_low=m.addConstrs((-T_in[i+1]<=-(self.set_temperature[i]-self.deadband-s_HVAC_neg[i])
                                              for i in range(horizon)),name='c_temp_low')
        sign_of_dual_variable.extend([-1]*(horizon))

        const_temp_cntrl_up=m.addConstrs((T_in[i+1]<=self.set_temperature[i]+self.deadband+s_HVAC_pos[i]
                                              for i in range(horizon)),name='c_temp_up')
        sign_of_dual_variable.extend([-1]*(horizon))

        neg_x_dev=m.addConstrs((s_HVAC_neg[i]<=self.neg_devs[i]+x
                                              for i in range(horizon)),name='c_neg_x_dev')
        sign_of_dual_variable.extend([-1]*(horizon))

        pos_x_dev=m.addConstrs((s_HVAC_pos[i]<=self.pos_devs[i]+x
                                              for i in range(horizon)),name='c_pos_x_dev')
        sign_of_dual_variable.extend([-1]*(horizon))

        const_HVAC_real_pow=m.addConstrs((P_HVAC[i]*power_HVAC-u_HVAC[i]==self.hvac_desirable_load[i]
                                              for i in range(horizon)),name='c_HVAC_realpow')
        sign_of_dual_variable.extend([0]*(horizon))

        const_HVAC_abs_value_pos=m.addConstrs((u_HVAC[i]-y_HVAC[i]<=0
                                              for i in range(horizon)),name='c_HVAC_abs_value_pos')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        const_HVAC_abs_value_neg=m.addConstrs((-u_HVAC[i]-y_HVAC[i]<=0
                                              for i in range(horizon)),name='c_HVAC_abs_value_neg')
        sign_of_dual_variable.extend([-1]*(horizon))
          
        const_init_temp=m.addConstr(T_in[0]==self.init_temp,name='c_hvac_init_temp')
        sign_of_dual_variable.extend([0]*(1))
        
        if self.hvac.s_effect==1:
            ##additional constraints
            #Equation 8, set M=1000
            const_on_off_1=m.addConstrs((-T_set[i]+self.deadband+T_in[i]<=1000*(1-P_HVAC[i])
                                                  for i in range(horizon)),name='c_HVAC_on_off_eq8')
            sign_of_dual_variable.extend([-1]*(horizon))
            #Equation 9, set M=1000
            const_on_off_2=m.addConstrs((-T_in[i]+T_set[i]-self.deadband<=1000*(P_HVAC[i])
                                                  for i in range(horizon)),name='c_HVAC_on_off_eq9')
            sign_of_dual_variable.extend([-1]*(horizon))
        else:
            #summer season cooling
            #modified Eq8
            const_on_off_1=m.addConstrs((T_set[i]+self.deadband-T_in[i]<=1000*(1-P_HVAC[i])
                                                  for i in range(horizon)),name='c_HVAC_on_off_eq8')
            sign_of_dual_variable.extend([-1]*(horizon))
            
            #modified Eq9
            const_on_off_2=m.addConstrs((T_in[i]-T_set[i]-self.deadband<=1000*(P_HVAC[i])
                                                  for i in range(horizon)),name='c_HVAC_on_off_eq9')
            sign_of_dual_variable.extend([-1]*(horizon))
            
        
        
        
        
        
        """
        ________________________EWH__________________________
        TODO 
            shif_by : must be set by outside function 
            p       : must be set by outside function 
        """
        rho=4186#J/(KG*°C)
        shift_by=0
        p=0.0#safety percentage
        
        
        water_demand=self.ewh_demand
        
        #shift water demand and add safety stock.
        first_of_shifted=np.sum(water_demand[:shift_by+1])

        shifted_demand=np.roll(water_demand, -shift_by)
        shifted_demand[len(shifted_demand)-shift_by:len(shifted_demand)]=0
        shifted_demand[0]=first_of_shifted
        shifted_demand=shifted_demand*(100+p)/100
        #
        """
        POSSIBLE TODO
            water_demand and water_level constraints can be relaxed by adding a slack variable.
            Note that need to adjust both constraints !!!
        """
        
    
        
        
        #constraints
        water_demand=m.addConstrs((-x_ewh[i]<=-shifted_demand[i]
                                              for i in range(horizon)),name='c_water_demand')
        sign_of_dual_variable.extend([-1]*(horizon))

        water_level=m.addConstrs((x_ewh[i+1]==x_ewh[i]+z_ewh[i]-shifted_demand[i]
                                              for i in range(horizon)),name='c_water_level')
        sign_of_dual_variable.extend([0]*(horizon))

        heated_water=m.addConstrs((z_ewh[i]==P_ewh[i]*1000*60*self.ewh.time_r*self.ewh.efficiency/(rho*(self.des_water_temp-self.tap_water_temp))
                                              for i in range(horizon)),name='c_ewh_heated_water')
        sign_of_dual_variable.extend([0]*(horizon))

        const_ewh_real_pow=m.addConstrs((P_ewh[i]-u_ewh[i]==self.ewh_desirable_load[i]
                                              for i in range(horizon)),name='c_ewh_realpow')
        sign_of_dual_variable.extend([0]*(horizon))
        
                    
        const_ewh_abs_value_pos=m.addConstrs((u_ewh[i]<=y_ewh[i]
                                              for i in range(horizon)),name='c_ewh_abs_value_pos')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        const_ewh_abs_value_neg=m.addConstrs((-u_ewh[i]<=y_ewh[i]
                                              for i in range(horizon)),name='c_ewh_abs_value_neg')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        """
        TODO
            need to add this constraint to the formulation report.
        """
        #not to charge can be costlier than charging, to avoid such cases, we can either introduce
        #a constraint on total real ev power, or need to set cost of deviation carefully.
        const_ewh_real_des=m.addConstr((quicksum(P_ewh[i]*1 for i in range(horizon))<=quicksum(self.ewh_desirable_load[i]*1 for i in range(horizon))),name='c_ewh_balance')
        sign_of_dual_variable.extend([-1]*(1))
        
        const_init_water=m.addConstr(x_ewh[0]==self.water_amount,name='c_ewh_init_water')
        sign_of_dual_variable.extend([0]*(1))
     
     
        """
        ________________________EV__________________________
        TODO 
            shif_by : must be set by outside function 
            p       : must be set by outside function 
        """
        shift_by=0
        p=0.0#safety percentage
        
       
        average_energy=0.346#KwH per mile
        average_energy=average_energy*3600 #kw per mile
        ev_demand=self.ev_demand
        #shift ev demand and add safety stock.
        first_of_shifted=np.sum(ev_demand[:shift_by+1])

        shifted_demand=np.roll(ev_demand, -shift_by)
        shifted_demand[len(shifted_demand)-shift_by:len(shifted_demand)]=0
        shifted_demand[0]=first_of_shifted
        shifted_demand=shifted_demand*(100+p)/100
        shifted_demand=shifted_demand*average_energy#miles to Kwh conversion.
        #
        
        #allow charging
        
        car_in_use=np.where(ev_demand>0)[0]
        #K_wm=np.arange(start_index,end_index+1)
        #full_index=np.arange(0,horizon)
        #K_ev_charging=np.setdiff1d(full_index, np.intersect1d(full_index, car_in_use))
        
        

        #deneme_ev=m.addVars(horizon,lb=0,ub=capacity)
        """
        POSSIBLE TODO
            ev_demand and ev_level constraints can be relaxed by adding a slack variable.
            Note that need to adjust both constraints !!!
        """
        
        
        
        
        #constraints
        ev_demand=m.addConstrs((-x_ev[i]<=-shifted_demand[i]
                                             for i in range(horizon)),name='c_ev_demand')
        sign_of_dual_variable.extend([-1]*(horizon))

        ev_level=m.addConstrs((x_ev[i+1]==x_ev[i]+P_ev[i]*(60*self.wm.time_r)-shifted_demand[i]
                                             for i in range(horizon)),name='c_ev_level')
        sign_of_dual_variable.extend([0]*(horizon))

        applied_power=m.addConstrs((P_ev[i]==((240*i_ev[i])/1000)
                                             for i in range(horizon)),name='c_ev_power')
        sign_of_dual_variable.extend([0]*(horizon))

        in_use=m.addConstrs((i_ev[i]==0
                                         for i in car_in_use),name='c_ev_in_use')
        sign_of_dual_variable.extend([0]*(len(car_in_use)))
        
        const_ev_real_pow=m.addConstrs((P_ev[i]-u_ev[i]==self.ev_desirable_load[i]
                                             for i in range(horizon)),name='c_ev_realpow')
        sign_of_dual_variable.extend([0]*(horizon))

              
        const_ev_abs_value_pos=m.addConstrs((u_ev[i]-y_ev[i]<=0
                                              for i in range(horizon)),name='c_ev_abs_value_pos')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        const_ev_abs_value_neg=m.addConstrs((-u_ev[i]-y_ev[i]<=0
                                              for i in range(horizon)),name='c_ev_abs_value_neg')
        sign_of_dual_variable.extend([-1]*(horizon))
        

        
        """
        TODO
            need to add this constraint to the formulation report.
        """
        #not to charge can be costlier than charging, to avoid such cases, we can either introduce
        #a constraint on total real ev power, or need to set cost of deviation carefully.
        const_real_des=m.addConstr((quicksum(P_ev[i]*1 for i in range(horizon))<=quicksum(self.ev_desirable_load[i]*1 for i in range(horizon))),name='c_ev_balance')
        sign_of_dual_variable.extend([-1]*(1))
        
        const_init_ev=m.addConstr(x_ev[0]==self.ev_battery,name='c_ev_init_charged')
        sign_of_dual_variable.extend([0]*(1))
        
        
        """
        ________________________PV__________________________
        TODO 
        ...
        """
        #considers the change in stored energy.
        pv_level=m.addConstrs((x_pv[i+1]==x_pv[i]+self.pv.stored_energy[i]-P_pv[i]*(60*self.wm.time_r/3600)
                                             for i in range(horizon)),name='c_pv_stored_level')
        sign_of_dual_variable.extend([0]*(horizon))
        
        #consume stored energy only when there is electricity demand
        pv_non_negativity=m.addConstrs((P_pv[i]-(P_ewh[i]+P_ev[i]+P_HVAC[i]+P_oven[i]+P_wm[i]+P_dryer[i])<=0
                                             for i in range(horizon)),name='c_pv_non_negativity')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        const_init_temp=m.addConstr(x_pv[0]==self.init_energy,name='c_pv_init_storage')
        sign_of_dual_variable.extend([0]*(1))
        
    
        
        
        #-1 automatic 0 primal 1 dual 2 barrier
        m.Params.Method=0
        m.optimize()
        
        
        P_ewh_a=np.zeros(horizon)
        P_ev_a=np.zeros(horizon)
        P_hvac_a=np.zeros(horizon)
        P_oven_a=np.zeros(horizon)
        P_wm_a=np.zeros(horizon)
        P_dryer_a=np.zeros(horizon)
        P_pv_a=np.zeros(horizon)
        P_refrigerator_a=self.refrigerator_desirable_load
        
        
        #deviations from desirable power level.
        P_ewh_d=np.zeros(horizon)
        P_ev_d=np.zeros(horizon)
        P_hvac_d=np.zeros(horizon)
        P_oven_d=np.zeros(horizon)
        P_wm_d=np.zeros(horizon)
        P_dryer_d=np.zeros(horizon)
        
        for i in range(horizon):
            P_ewh_a[i]=P_ewh[i].X
            P_ev_a[i]=P_ev[i].X
            P_hvac_a[i]=P_HVAC[i].X*power_HVAC
            P_oven_a[i]=P_oven[i].X
            P_wm_a[i]=P_wm[i].X
            P_dryer_a[i]=P_dryer[i].X
            P_pv_a[i]=P_pv[i].X
            
            P_wm_d[i]=u_wm[i].X
            P_oven_d[i]=u_oven[i].X
            P_dryer_d[i]=u_dryer[i].X
            P_hvac_d[i]=u_HVAC[i].X#deviation variable is unbounded. No need to have a multiplication.
            P_ewh_d[i]=u_ewh[i].X
            P_ev_d[i]=u_ev[i].X
            
        real_power={'ewh':P_ewh_a,
                 'ev':P_ev_a,
                 'hvac':P_hvac_a,
                 'oven':P_oven_a,
                 'wm':P_wm_a,
                 'dryer':P_dryer_a,
                 'pv':P_pv_a,
                 'refrigerator':P_refrigerator_a}
        
        dev_power={'ewh':P_ewh_d,
                 'ev':P_ev_d,
                 'hvac':P_hvac_d,
                 'oven':P_oven_d,
                 'wm':P_wm_d,
                 'dryer':P_dryer_d}
        
        
        x_ewh_a=np.zeros(horizon+1)
        T_in_a=np.zeros(horizon+1)
        x_ev_a=np.zeros(horizon+1)
        x_pv_a=np.zeros(horizon+1)
        
        for i in range(horizon+1):
            x_ewh_a[i]=x_ewh[i].X
            T_in_a[i]=T_in[i].X
            x_ev_a[i]=x_ev[i].X
            x_pv_a[i]=x_pv[i].X
            
        states={'ewh':x_ewh_a,
                 'ev':x_ev_a,
                 'pv':x_pv_a,
                 'hvac':T_in_a}
        
        
        #deneme=0
        #for i in range(horizon):
        #    print(deneme_ev[i].X)
        #    deneme+=deneme_ev[i].X
        #print(deneme)
        
        #get dual related info
        RHS=m.RHS#gives the RHS of constraints.
        A=m.getA()#gives constraint matrix
        
        dual={'RHS':RHS,
              'A': A,
              'c':modified_c,
              'dual_var_sign':sign_of_dual_variable,
              'dual_const_sign':sign_of_dual_constraint}

        
        
        return real_power,dev_power,states,dual,m,p_obj
    
    
    def primal_constraints(self,cost_u,price,m):
        """
        Args:
            cost_u: is a dictionary. cost of deviation from desireable curve for each appliance.
            price : shows the price of electricity in the next 24 hours. (1*96 array.)
        """
        #jj=0
        #tmp_name="H"+str(jj+1)+"_P"
        cost_u_ewh=cost_u['ewh']
        cost_u_ev=cost_u['ev']
        cost_u_hvac=cost_u['hvac']
        cost_u_oven=cost_u['oven']
        cost_u_wm=cost_u['wm']
        cost_u_dryer=cost_u['dryer']
        
        horizon=len(self.wm_desirable_load)
        #m = Model("home_mpc")
        
        obj_ewh_price=0.0
        obj_ewh_dev=0.0
        obj_ev_price=0.0
        obj_ev_dev=0.0
        obj_hvac_price=0.0
        obj_hvac_dev=0.0
        obj_oven_price=0.0
        obj_oven_dev=0.0
        obj_wm_price=0.0
        obj_wm_dev=0.0
        obj_dryer_price=0.0
        obj_dryer_dev=0.0
        
        """
        POSSIBLE TODO
            Currently, we assume all devices are scheduled in one day. 
            What if we do not schedule one of those appliances.
            optimize_mpc function before March 7, handles this. However, this
            feature is sacrificed in the last update published on March 7.
            You may need to think about this!.
        """
        
        
        
        #necessary parameters
            #max nominal power
        power_wm=self.wm.nominal_power
        power_oven=self.oven.nominal_power
        power_dryer=self.dryer.nominal_power
        power_HVAC=self.hvac.nominal_power
        power_ewh=self.ewh.nominal_power
        power_ev=self.ev.nominal_power
            #max capacity
        capacity_ewh=self.ewh.capacity
        capacity_ev=self.ev.capacity
            #deviations
        x=0.5 #HVAC related todo!
        #     #demand
            
        #     #ewh
            
        #     #ev
        """
        TODO
            Note that some variables are free. Dual constraints must have equality signs!
        """
        sign_of_dual_variable=[]
        sign_of_dual_constraint=[]
        num_variables=0
        num_helper_variables=0
        #decision variables
            #deviation variables
        y_wm=m.addVars(horizon,lb=0)
        y_oven=m.addVars(horizon,lb=0)
        y_dryer=m.addVars(horizon,lb=0)
        y_HVAC=m.addVars(horizon,lb=0)
        y_ewh=m.addVars(horizon,lb=0)
        y_ev=m.addVars(horizon,lb=0)
        num_variables+=horizon*6
        
        
            #power variables
        P_wm=m.addVars(horizon,lb=0)
        P_oven=m.addVars(horizon,lb=0)
        P_dryer=m.addVars(horizon,lb=0)
        P_HVAC=m.addVars(horizon,lb=0)
        P_ewh=m.addVars(horizon,lb=0)
        P_ev=m.addVars(horizon,lb=0)
        P_pv=m.addVars(horizon,lb=0)
        num_variables+=horizon*7
        sign_of_dual_constraint=[-1]*num_variables
            
            #helper variables
                #ewh related
        x_ewh=m.addVars(horizon+1,lb=0)
        z_ewh=m.addVars(horizon,lb=0)
        #y_ewh=m.addVars(horizon,lb=0)
                #ev related
        x_ev=m.addVars(horizon+1,lb=0)
        i_ev=m.addVars(horizon,lb=0)
                #HVAC related
        s_HVAC_neg=m.addVars(horizon,lb=0)
        s_HVAC_pos=m.addVars(horizon,lb=0)
        T_in=m.addVars(horizon+1,lb=-GRB.INFINITY)
        """
        add T_set as decision variable to govern on/off of HVAC.
        """
        T_set=m.addVars(horizon,lb=-GRB.INFINITY)
                #pv related
        x_pv=m.addVars(horizon+1,lb=0)
        sign_of_dual_constraint.extend([-1]*(6*horizon+2))
        sign_of_dual_constraint.extend([0]*(2*horizon+1))
        sign_of_dual_constraint.extend([-1]*(1*horizon+1))
                #other deviations
        u_wm=m.addVars(horizon,lb=-GRB.INFINITY)
        u_oven=m.addVars(horizon,lb=-GRB.INFINITY)
        u_dryer=m.addVars(horizon,lb=-GRB.INFINITY)
        u_HVAC=m.addVars(horizon,lb=-GRB.INFINITY)
        u_ewh=m.addVars(horizon,lb=-GRB.INFINITY)
        u_ev=m.addVars(horizon,lb=-GRB.INFINITY)
        
        num_helper_variables+=horizon*6
        num_helper_variables+=horizon*5+(horizon+1)*4
        num_variables+=num_helper_variables
        sign_of_dual_constraint.extend([0]*(6*horizon))
        
        
        #objective function construction
        #first deviations then consumption related fee.
        """
        m.setObjective(quicksum(y_wm[i]*cost_u_wm for i in range(horizon))+\
                       quicksum(y_oven[i]*cost_u_oven for i in range(horizon))+\
                       quicksum(y_dryer[i]*cost_u_dryer for i in range(horizon))+\
                       quicksum(y_HVAC[i]*cost_u_hvac for i in range(horizon))+\
                       quicksum(y_ewh[i]*cost_u_ewh for i in range(horizon))+\
                       quicksum(y_ev[i]*cost_u_ev for i in range(horizon))+\
                       quicksum(P_wm[i]*price[i] for i in range(horizon))+\
                       quicksum(P_oven[i]*price[i] for i in range(horizon))+\
                       quicksum(P_dryer[i]*price[i] for i in range(horizon))+\
                       quicksum(P_HVAC[i]*power_HVAC*price[i] for i in range(horizon))+\
                       quicksum(P_ewh[i]*price[i] for i in range(horizon))+\
                       quicksum(P_ev[i]*price[i] for i in range(horizon))+\
                       quicksum(P_pv[i]*-price[i] for i in range(horizon))    , GRB.MINIMIZE)
        """
        
        p_obj=quicksum(y_wm[i]*cost_u_wm for i in range(horizon))+\
                       quicksum(y_oven[i]*cost_u_oven for i in range(horizon))+\
                       quicksum(y_dryer[i]*cost_u_dryer for i in range(horizon))+\
                       quicksum(y_HVAC[i]*cost_u_hvac for i in range(horizon))+\
                       quicksum(y_ewh[i]*cost_u_ewh for i in range(horizon))+\
                       quicksum(y_ev[i]*cost_u_ev for i in range(horizon))+\
                       quicksum(P_wm[i]*price[i] for i in range(horizon))+\
                       quicksum(P_oven[i]*price[i] for i in range(horizon))+\
                       quicksum(P_dryer[i]*price[i] for i in range(horizon))+\
                       quicksum(P_HVAC[i]*power_HVAC*price[i] for i in range(horizon))+\
                       quicksum(P_ewh[i]*price[i] for i in range(horizon))+\
                       quicksum(P_ev[i]*price[i] for i in range(horizon))+\
                       quicksum(P_pv[i]*-price[i] for i in range(horizon)) 
        
        modified_c=np.array([])
        modified_c=np.concatenate((modified_c,np.repeat(cost_u_wm,horizon)))
        modified_c=np.concatenate((modified_c,np.repeat(cost_u_oven,horizon)))
        modified_c=np.concatenate((modified_c,np.repeat(cost_u_dryer,horizon)))
        modified_c=np.concatenate((modified_c,np.repeat(cost_u_hvac,horizon)))
        modified_c=np.concatenate((modified_c,np.repeat(cost_u_ewh,horizon)))
        modified_c=np.concatenate((modified_c,np.repeat(cost_u_ev,horizon)))
        modified_c=np.concatenate((modified_c,price))#wm
        modified_c=np.concatenate((modified_c,price))#oven
        modified_c=np.concatenate((modified_c,price))#dryer
        modified_c=np.concatenate((modified_c,power_HVAC*price))#hvac
        modified_c=np.concatenate((modified_c,price))#ewh
        modified_c=np.concatenate((modified_c,price))#ev
        modified_c=np.concatenate((modified_c,-price))#pv
        
        modified_c=np.concatenate((modified_c,np.repeat(0,num_helper_variables)))#cost is zero for other vars.
        
        
        

        


        
        
        
        
        
        
        #constraints
        
        #real power upper bounds
        m.addConstrs((P_wm[i]<=power_wm for i in range(horizon)),name='c_wm_realpow_up')
        m.addConstrs((P_oven[i]<=power_oven for i in range(horizon)),name='c_oven_realpow_up')
        m.addConstrs((P_dryer[i]<=power_dryer for i in range(horizon)),name='c_dryer_realpow_up')
        m.addConstrs((P_HVAC[i]<=1 for i in range(horizon)),name='c_HVAC_realpow_up')
        m.addConstrs((P_ewh[i]<=power_ewh for i in range(horizon)),name='c_ewh_realpow_up')
        m.addConstrs((P_ev[i]<=power_ev for i in range(horizon)),name='c_ev_realpow_up')
        sign_of_dual_variable=[-1]*(6*horizon)
        
        #state variable upper bounbds
        m.addConstrs((x_ewh[i]<=capacity_ewh for i in range(horizon+1)),name='c_ewh_water_up')
        m.addConstrs((x_ev[i]<=capacity_ev for i in range(horizon+1)),name='c_ev_battery_up')
        m.addConstrs((i_ev[i]<=self.ev_current for i in range(horizon)),name='c_ev_current_up')
        sign_of_dual_variable.extend([-1]*(3*horizon+2))


        """
        ________________________WM__________________________
        TODO 
            delay_allowance is a parameter that must be determined for each home by a outside preference function
        """
        operation_index_wm=np.where(self.wm_desirable_load>0)[0]
        operation_index_oven=np.where(self.oven_desirable_load>0)[0]
        operation_index_dryer=np.where(self.dryer_desirable_load>0)[0]
        
        start_index=operation_index_wm[0]
        end_index=operation_index_wm[len(operation_index_wm)-1]
        delay_allowance=1#in terms of hours
        ints_per_hour=int(60/self.wm.time_r)
        start_index=start_index-int(delay_allowance*ints_per_hour)
        end_index=end_index+int(delay_allowance*ints_per_hour)
        K_wm=np.arange(start_index,end_index+1)
        full_index=np.arange(0,horizon)
        K_wm_c=np.setdiff1d(full_index, np.intersect1d(full_index, K_wm))


        #wm constraints
        const_wm_real_pow=m.addConstrs((P_wm[i]-u_wm[i]==self.wm_desirable_load[i]
                                              for i in range(horizon)),name='c_wm_realpow')
        sign_of_dual_variable.extend([0]*(horizon))

        #const_wm_cntrl_low=m.addConstrs((u_wm[i]>=-power
        #                                      for i in K_wm),name='c_wm_cntrl_low')

        #const_wm_cntrl_up=m.addConstrs((u_wm[i]<=power
        #                                      for i in K_wm),name='c_wm_cntrl_up')

        const_wm_cntrl_zero=m.addConstrs((u_wm[i]==0
                                              for i in K_wm_c),name='c_wm_cntrl_zero')
        sign_of_dual_variable.extend([0]*(len(K_wm_c)))

        const_wm_balance=m.addConstr((quicksum(u_wm[i]*1 for i in range(horizon))==0),name='c_wm_balance')
        sign_of_dual_variable.extend([0]*(1))
        


        
        const_wm_abs_value_pos=m.addConstrs((u_wm[i]-y_wm[i]<=0
                                              for i in range(horizon)),name='c_wm_abs_value_pos')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        const_wm_abs_value_neg=m.addConstrs((-u_wm[i]-y_wm[i]<=0
                                              for i in range(horizon)),name='c_wm_abs_value_neg')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        """
        ________________________Oven__________________________
        TODO 
            delay_allowance is a parameter that must be determined for each home by a outside preference function
        """
        
        start_index=operation_index_oven[0]
        end_index=operation_index_oven[len(operation_index_oven)-1]
        delay_allowance=1#in terms of hours
        ints_per_hour=int(60/self.wm.time_r)
        start_index=start_index-int(delay_allowance*ints_per_hour)
        end_index=end_index+int(delay_allowance*ints_per_hour)
        K_oven=np.arange(start_index,end_index+1)
        full_index=np.arange(0,horizon)
        K_oven_c=np.setdiff1d(full_index, np.intersect1d(full_index, K_oven))



        #oven constraints
        const_oven_real_pow=m.addConstrs((P_oven[i]-u_oven[i]==self.oven_desirable_load[i]
                                              for i in range(horizon)),name='c_oven_realpow')
        sign_of_dual_variable.extend([0]*(horizon))

        #const_oven_cntrl_low=m.addConstrs((u_oven[i]>=-power
        #                                      for i in K_oven),name='c_oven_cntrl_low')

        #const_oven_cntrl_up=m.addConstrs((u_oven[i]<=power
        #                                      for i in K_oven),name='c_oven_cntrl_up')

        const_oven_cntrl_zero=m.addConstrs((u_oven[i]==0
                                              for i in K_oven_c),name='c_oven_cntrl_zero')
        sign_of_dual_variable.extend([0]*(len(K_oven_c)))

        const_oven_balance=m.addConstr((quicksum(u_oven[i]*1 for i in range(horizon))==0),name='c_oven_balance')
        sign_of_dual_variable.extend([0]*(1))
          
        const_oven_abs_value_pos=m.addConstrs((u_oven[i]-y_oven[i]<=0
                                              for i in range(horizon)),name='c_oven_abs_value_pos')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        const_oven_abs_value_neg=m.addConstrs((-u_oven[i]-y_oven[i]<=0
                                              for i in range(horizon)),name='c_oven_abs_value_neg')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        
        
        
        """
        ________________________Dryer__________________________
        TODO 
            delay_allowance is a parameter that must be determined for each home by a outside preference function
        """
        start_index=operation_index_dryer[0]
        end_index=operation_index_dryer[len(operation_index_dryer)-1]
        delay_allowance=1#in terms of hours
        ints_per_hour=int(60/self.wm.time_r)
        start_index=start_index-int(delay_allowance*ints_per_hour)
        end_index=end_index+int(delay_allowance*ints_per_hour)
        K_dryer=np.arange(start_index,end_index+1)
        full_index=np.arange(0,horizon)
        K_dryer_c=np.setdiff1d(full_index, np.intersect1d(full_index, K_dryer))



        #wm constraints
        const_dryer_real_pow=m.addConstrs((P_dryer[i]-u_dryer[i]==self.dryer_desirable_load[i]
                                              for i in range(horizon)),name='c_dryer_realpow')
        sign_of_dual_variable.extend([0]*(horizon))


        const_dryer_cntrl_zero=m.addConstrs((u_dryer[i]==0
                                              for i in K_dryer_c),name='c_dryer_cntrl_zero')
        sign_of_dual_variable.extend([0]*(len(K_dryer_c)))

        const_dryer_balance=m.addConstr((quicksum(u_dryer[i]*1 for i in range(horizon))==0),name='c_dryer_balance')
        sign_of_dual_variable.extend([0]*(1))
        
                    
        const_dryer_abs_value_pos=m.addConstrs((u_dryer[i]-y_dryer[i]<=0
                                              for i in range(horizon)),name='c_dryer_abs_value_pos')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        const_dryer_abs_value_neg=m.addConstrs((-u_dryer[i]-y_dryer[i]<=0
                                              for i in range(horizon)),name='c_dryer_abs_value_neg')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        
        """
        ________________________HVAC__________________________
        TODO 
            x is a parameter that must be determined for each home by a outside preference function
        """
        
        
        
        
        #hvac constraints
        const_temp_change=m.addConstrs((T_in[i+1]==T_in[i]+self.gamma1*(self.hvac.T_out[i]-T_in[i])+self.gamma2*P_HVAC[i]*power_HVAC*self.hvac.efficiency*(1000*60*self.hvac.time_r)*self.hvac.s_effect
                                              for i in range(horizon)),name='c_hvac_temp_chng')
        sign_of_dual_variable.extend([0]*(horizon))


        const_temp_cntrl_low=m.addConstrs((-T_in[i+1]<=-(self.set_temperature[i]-self.deadband-s_HVAC_neg[i])
                                              for i in range(horizon)),name='c_temp_low')
        sign_of_dual_variable.extend([-1]*(horizon))

        const_temp_cntrl_up=m.addConstrs((T_in[i+1]<=self.set_temperature[i]+self.deadband+s_HVAC_pos[i]
                                              for i in range(horizon)),name='c_temp_up')
        sign_of_dual_variable.extend([-1]*(horizon))

        neg_x_dev=m.addConstrs((s_HVAC_neg[i]<=self.neg_devs[i]+x
                                              for i in range(horizon)),name='c_neg_x_dev')
        sign_of_dual_variable.extend([-1]*(horizon))

        pos_x_dev=m.addConstrs((s_HVAC_pos[i]<=self.pos_devs[i]+x
                                              for i in range(horizon)),name='c_pos_x_dev')
        sign_of_dual_variable.extend([-1]*(horizon))

        const_HVAC_real_pow=m.addConstrs((P_HVAC[i]*power_HVAC-u_HVAC[i]==self.hvac_desirable_load[i]
                                              for i in range(horizon)),name='c_HVAC_realpow')
        sign_of_dual_variable.extend([0]*(horizon))

        const_HVAC_abs_value_pos=m.addConstrs((u_HVAC[i]-y_HVAC[i]<=0
                                              for i in range(horizon)),name='c_HVAC_abs_value_pos')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        const_HVAC_abs_value_neg=m.addConstrs((-u_HVAC[i]-y_HVAC[i]<=0
                                              for i in range(horizon)),name='c_HVAC_abs_value_neg')
        sign_of_dual_variable.extend([-1]*(horizon))
          
        const_init_temp=m.addConstr(T_in[0]==self.init_temp,name='c_hvac_init_temp')
        sign_of_dual_variable.extend([0]*(1))
        
        ##additional constraints
        #Equation 8, set M=1000
        const_on_off_1=m.addConstrs((-T_set[i]+self.deadband+T_in[i]<=1000*(1-P_HVAC[i])
                                              for i in range(horizon)),name='c_HVAC_on_off_eq8')
        sign_of_dual_variable.extend([-1]*(horizon))
        #Equation 9, set M=1000
        const_on_off_2=m.addConstrs((-T_in[i]+T_set[i]-self.deadband<=1000*(P_HVAC[i])
                                              for i in range(horizon)),name='c_HVAC_on_off_eq9')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        
        
        
        """
        ________________________EWH__________________________
        TODO 
            shif_by : must be set by outside function 
            p       : must be set by outside function 
        """
        rho=4186#J/(KG*°C)
        shift_by=0
        p=0.0#safety percentage
        
        
        water_demand=self.ewh_demand
        
        #shift water demand and add safety stock.
        first_of_shifted=np.sum(water_demand[:shift_by+1])

        shifted_demand=np.roll(water_demand, -shift_by)
        shifted_demand[len(shifted_demand)-shift_by:len(shifted_demand)]=0
        shifted_demand[0]=first_of_shifted
        shifted_demand=shifted_demand*(100+p)/100
        #
        """
        POSSIBLE TODO
            water_demand and water_level constraints can be relaxed by adding a slack variable.
            Note that need to adjust both constraints !!!
        """
        
    
        
        
        #constraints
        water_demand=m.addConstrs((-x_ewh[i]<=-shifted_demand[i]
                                              for i in range(horizon)),name='c_water_demand')
        sign_of_dual_variable.extend([-1]*(horizon))

        water_level=m.addConstrs((x_ewh[i+1]==x_ewh[i]+z_ewh[i]-shifted_demand[i]
                                              for i in range(horizon)),name='c_water_level')
        sign_of_dual_variable.extend([0]*(horizon))

        heated_water=m.addConstrs((z_ewh[i]==P_ewh[i]*1000*60*self.ewh.time_r*self.ewh.efficiency/(rho*(self.des_water_temp-self.tap_water_temp))
                                              for i in range(horizon)),name='c_ewh_heated_water')
        sign_of_dual_variable.extend([0]*(horizon))

        const_ewh_real_pow=m.addConstrs((P_ewh[i]-u_ewh[i]==self.ewh_desirable_load[i]
                                              for i in range(horizon)),name='c_ewh_realpow')
        sign_of_dual_variable.extend([0]*(horizon))
        
                    
        const_ewh_abs_value_pos=m.addConstrs((u_ewh[i]<=y_ewh[i]
                                              for i in range(horizon)),name='c_ewh_abs_value_pos')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        const_ewh_abs_value_neg=m.addConstrs((-u_ewh[i]<=y_ewh[i]
                                              for i in range(horizon)),name='c_ewh_abs_value_neg')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        """
        TODO
            need to add this constraint to the formulation report.
        """
        #not to charge can be costlier than charging, to avoid such cases, we can either introduce
        #a constraint on total real ev power, or need to set cost of deviation carefully.
        const_ewh_real_des=m.addConstr((quicksum(P_ewh[i]*1 for i in range(horizon))<=quicksum(self.ewh_desirable_load[i]*1 for i in range(horizon))),name='c_ewh_balance')
        sign_of_dual_variable.extend([-1]*(1))
        
        const_init_water=m.addConstr(x_ewh[0]==self.water_amount,name='c_ewh_init_water')
        sign_of_dual_variable.extend([0]*(1))
     
     
        """
        ________________________EV__________________________
        TODO 
            shif_by : must be set by outside function 
            p       : must be set by outside function 
        """
        shift_by=0
        p=0.0#safety percentage
        
       
        average_energy=0.346#KwH per mile
        average_energy=average_energy*3600 #kw per mile
        ev_demand=self.ev_demand
        #shift ev demand and add safety stock.
        first_of_shifted=np.sum(ev_demand[:shift_by+1])

        shifted_demand=np.roll(ev_demand, -shift_by)
        shifted_demand[len(shifted_demand)-shift_by:len(shifted_demand)]=0
        shifted_demand[0]=first_of_shifted
        shifted_demand=shifted_demand*(100+p)/100
        shifted_demand=shifted_demand*average_energy#miles to Kwh conversion.
        #
        
        #allow charging
        
        car_in_use=np.where(ev_demand>0)[0]
        #K_wm=np.arange(start_index,end_index+1)
        #full_index=np.arange(0,horizon)
        #K_ev_charging=np.setdiff1d(full_index, np.intersect1d(full_index, car_in_use))
        
        

        #deneme_ev=m.addVars(horizon,lb=0,ub=capacity)
        """
        POSSIBLE TODO
            ev_demand and ev_level constraints can be relaxed by adding a slack variable.
            Note that need to adjust both constraints !!!
        """
        
        
        
        
        #constraints
        ev_demand=m.addConstrs((-x_ev[i]<=-shifted_demand[i]
                                             for i in range(horizon)),name='c_ev_demand')
        sign_of_dual_variable.extend([-1]*(horizon))

        ev_level=m.addConstrs((x_ev[i+1]==x_ev[i]+P_ev[i]*(60*self.wm.time_r)-shifted_demand[i]
                                             for i in range(horizon)),name='c_ev_level')
        sign_of_dual_variable.extend([0]*(horizon))

        applied_power=m.addConstrs((P_ev[i]==((240*i_ev[i])/1000)
                                             for i in range(horizon)),name='c_ev_power')
        sign_of_dual_variable.extend([0]*(horizon))

        in_use=m.addConstrs((i_ev[i]==0
                                         for i in car_in_use),name='c_ev_in_use')
        sign_of_dual_variable.extend([0]*(len(car_in_use)))
        
        const_ev_real_pow=m.addConstrs((P_ev[i]-u_ev[i]==self.ev_desirable_load[i]
                                             for i in range(horizon)),name='c_ev_realpow')
        sign_of_dual_variable.extend([0]*(horizon))

              
        const_ev_abs_value_pos=m.addConstrs((u_ev[i]-y_ev[i]<=0
                                              for i in range(horizon)),name='c_ev_abs_value_pos')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        const_ev_abs_value_neg=m.addConstrs((-u_ev[i]-y_ev[i]<=0
                                              for i in range(horizon)),name='c_ev_abs_value_neg')
        sign_of_dual_variable.extend([-1]*(horizon))
        

        
        """
        TODO
            need to add this constraint to the formulation report.
        """
        #not to charge can be costlier than charging, to avoid such cases, we can either introduce
        #a constraint on total real ev power, or need to set cost of deviation carefully.
        const_real_des=m.addConstr((quicksum(P_ev[i]*1 for i in range(horizon))<=quicksum(self.ev_desirable_load[i]*1 for i in range(horizon))),name='c_ev_balance')
        sign_of_dual_variable.extend([-1]*(1))
        
        const_init_ev=m.addConstr(x_ev[0]==self.ev_battery,name='c_ev_init_charged')
        sign_of_dual_variable.extend([0]*(1))
        
        
        """
        ________________________PV__________________________
        TODO 
        ...
        """
        #considers the change in stored energy.
        pv_level=m.addConstrs((x_pv[i+1]==x_pv[i]+self.pv.stored_energy[i]-P_pv[i]*(60*self.wm.time_r/3600)
                                             for i in range(horizon)),name='c_pv_stored_level')
        sign_of_dual_variable.extend([0]*(horizon))
        
        #consume stored energy only when there is electricity demand
        pv_non_negativity=m.addConstrs((P_pv[i]-(P_ewh[i]+P_ev[i]+P_HVAC[i]+P_oven[i]+P_wm[i]+P_dryer[i])<=0
                                             for i in range(horizon)),name='c_pv_non_negativity')
        sign_of_dual_variable.extend([-1]*(horizon))
        
        const_init_temp=m.addConstr(x_pv[0]==self.init_energy,name='c_pv_init_storage')
        sign_of_dual_variable.extend([0]*(1))
        
    
        
        
        #-1 automatic 0 primal 1 dual 2 barrier
        #m.Params.Method=0
        #m.optimize()
        
        
        m.update()
        #get dual related info
        RHS=m.RHS#gives the RHS of constraints.
        A=m.getA()#gives constraint matrix
        
        dual={'RHS':RHS,
              'A': A,
              'c':modified_c,
              'dual_var_sign':sign_of_dual_variable,
              'dual_const_sign':sign_of_dual_constraint}

        
        
        return dual,m,p_obj
        
        
        
        
    

        
        
        
        
    

        
        
        
        
    

        
        
        
        
    
