import matplotlib.pyplot as plt
from main import *

plt.figure(figsize=(16, 16))
ctrs = np.zeros((200, 10))
for i in range(0, 200):
    mpc.update_market()
    ctrs[i, :] = mpc.ctr
    
plt.plot(ctrs[:, ])
plt.show()

np.mean(ctrs[:, ])


########
# costs versus bids
plt.plot(past_bids[0, :], past_costs[0, :], '*')


#######
# our cost and the target budget
cumsum_cost = np.cumsum(running_total_cost)

plt.plot(cumsum_cost)
plt.plot(y_target)
plt.show()

#######
# slope plot
plt.plot(slope_array)
plt.show()

#####
# bstar and ctr plot
plt.plot(ctr_array)
plt.plot(bstar_array)
plt.show()

#####
# inverse cpc
plt.plot(cpc_inv)
plt.show()


#### collected

# TODO:
# 3 slopes
#cpcinv+new_bids+
#cost versus u

fig, axs = plt.subplots(4, 3)

cumsum_cost = np.cumsum(running_total_cost)

axs[0, 0].plot(cumsum_cost)
axs[0, 0].plot(y_target)
axs[0, 0].set_title('cost evolution')
axs[0, 1].plot(slope_array_mean)
axs[0, 1].set_title('slope values')
axs[0, 2].plot(clicks_array)
axs[0, 2].set_title('clicks')
axs[1, 0].plot(bstar_array)
axs[1, 0].set_title('b_star')
axs[1, 1].plot(ctr_array)
axs[1, 1].set_title('ctr')
axs[1, 2].plot(invcpc_array)
axs[1, 2].set_title('cpc_inv')
axs[2, 0].plot(imps_array)
axs[2, 0].set_title('imps')
axs[2, 1].plot(bid_array)
axs[2, 1].set_ylim(0, 0.03)
axs[2, 1].set_title('bids')
axs[2, 2].plot(cost_array)
axs[2, 2].set_title('cost')
axs[3, 0].plot(alpha_array)
axs[3, 0].set_title('alpha')
axs[3, 1].plot(beta_array)
axs[3, 1].set_title('beta')
axs[3, 2].set_title('bid uncertainties')
axs[3, 2].plot(bu_array)
fig.show()

plt.plot(y_target)
plt.plot(cumsum_cost)
plt.show()

# cost daily pred
selected_day = 20
cost_daily_pred_20 = cost_daily_pred[selected_day]
days = list(range(selected_day, selected_day+N))
plt.plot(y_target, 'r')
plt.plot(cumsum_cost, 'k')
plt.plot(
    days,
    np.transpose(cost_daily_pred_20),
    alpha=.25,
    color='k',
    linewidth=0.5,
    linestyle='dashed'
)
plt.show()


# mean and variance objective plots
plt.subplot(2, 1, 1)
plt.plot(mean_terms)
plt.ylabel('mean objective')

plt.subplot(2, 1, 2)
plt.plot(variance_terms)
plt.ylabel('variance objective')

plt.show()

# grad plots
# dim(slope_array) = 86x(50x3)
# dim(intercept_array) = 86x(50x3)
slope_daily_20 = slope_array[80]
plt.hist(slope_daily_20[:, 0], bins=15, alpha=0.5, histtype='bar', ec='black')
plt.show()

intercept_daily_20 = intercept_array[selected_day]
u_20 = u_values[selected_day]

cost_lin = slope_daily_20 * np.transpose(u_20)

plt.plot(np.transpose(slope_daily_20[0, :]))
plt.show()


#### cost single plot
plt.plot(cumsum_cost)
plt.plot(y_target)
plt.show()