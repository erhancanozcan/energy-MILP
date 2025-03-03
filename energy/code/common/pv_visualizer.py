import numpy as np




def p(x):
    return -(x**2) + 96*x - 1728

def calc_coeff(x, t):
    return x / np.sum(p(t))

t = np.arange(24,72,1)


coeff_zero = calc_coeff(0,t)
coeff_one = calc_coeff(5,t)
coeff_two = calc_coeff(10,t)
coeff_three = calc_coeff(20,t)
coeff_four = calc_coeff(30,t)
#coeff_five = calc_coeff(40,t)
#coeff_six = calc_coeff(45,t)



import matplotlib.pyplot as plt


fig, ax = plt.subplots()
ax.plot(t, p(t)*coeff_zero,label="coeff_0")
ax.plot(t, p(t)*coeff_one,label="coeff_1")
ax.plot(t, p(t)*coeff_two,label="coeff_2")
ax.plot(t, p(t)*coeff_three,label="coeff_3")
ax.plot(t, p(t)*coeff_four,label="coeff_4")
#ax.plot(t, p(t_two)/120,label="four")
#ax.plot(t, p(t_two)/150,label="five")
#ax.plot(np.arange(len(Q)), data.NonACConsumption.values + tmp_sum,label="TotalConsumption After Optimization")
#ax.plot(np.arange(len(Q)), data.NonACConsumption.values,label="NonACConsumption")
#ax.plot(np.arange(len(Q)), data.Generation.values,label="PV Generation")
#ax.plot(np.arange(len(Q)), data.NonACConsumption.values + tmp_sum - data.Generation.values,label="NonAC+OptimizedAC-PV Generation")
ax.legend(loc="upper right")
#ax.set_ylim((-7500, 32500))
#ax.set_title('Q_parameter is '+str(PowerToBuy))