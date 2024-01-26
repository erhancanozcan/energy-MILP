#change in real power with respect to electricity price.
import matplotlib.pyplot as plt



import os
import numpy as np
script_location="/Users/can/Documents/GitHub"
os.chdir(script_location)


from energy.code.home import Home
from energy.code.common.demand import initialize_demand
from energy.code.common.appliance import initialize_appliance_property
from energy.code.common.power_analysis import home_power_plot

home=Home()

home=initialize_demand(home)
home=initialize_appliance_property(home)



home.generate_desirable_load()


horizon=len(home.wm_desirable_load)
price_mean=np.arange(0.35,50.35,5.0)


num_trial_per_price=1

total_real_power_list=[]
total_desirable_power_list=[]
cost_u_list=[]  

def init_dict(price_mean):
    
    tmp={}
    for i in price_mean:
        tmp[i]=[]
        
    return tmp

    
total_real_power_dic=init_dict(price_mean)
total_desirable_power_dic=init_dict(price_mean)

for mean in price_mean:
    for i in range (num_trial_per_price):
        
        #generate prices
        price=abs(np.random.normal(mean,0.1,size=horizon))#0.33$ per KwH
        total,cost_u,daily_fee_desirable=home.total_desirable_load(price,mean)
        cost_u_list.append(cost_u)
        
        real_power,states=home.optimize_mpc(cost_u,price)
        #real_power,states=home.dual_optimize_mpc(cost_u,price)
        _,_,total_real_power,total_desirable_power=home_power_plot(home,real_power,price,plot_type="total",plot=False)
        
        total_real_power_dic[mean].append(total_real_power)
        total_desirable_power_dic[mean].append(total_desirable_power)

fig, ax = plt.subplots()
ax.boxplot(total_real_power_dic.values())
#ax.boxplot(total_real_power_dic.values(),showmeans=True)
#ax.set_xticklabels(total_real_power_dic.keys())
ax.set_xticklabels(price_mean.round(2))
ax.set_title("Box Plot of Real Power with Respect to Price")
ax.set_xlabel("Mean of Electricity Price")
fig


#%%
home_power_plot(home,real_power,price,plot_type="ewh",plot=True)







