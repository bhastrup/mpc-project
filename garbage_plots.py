import matplotlib.pyplot as plt
#from main import *

plt.figure(figsize=(16, 16))
ctrs = np.zeros((200, 10))
for i in range(0, 200):
    mpc.update_market()
    ctrs[i, :] = mpc.ctr
    
plt.plot(ctrs[:, ])
plt.show()

np.mean(ctrs[:, ])


########
# costs versus bids, Ad slot 0
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
axs[0, 0].plot(y_target[:len(cumsum_cost)])
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
axs[2, 1].set_title('bids')
axs[2, 2].plot(cost_array)
axs[2, 2].set_title('cost')
axs[3, 0].plot(alpha_array)
axs[3, 0].set_title('alpha')
axs[3, 1].plot(beta_array)
axs[3, 1].set_title('beta')
axs[3, 2].set_title('bid uncertainties')
#axs[3, 2].plot(bu_array)
fig.show()

plt.plot(y_target[:len(cumsum_cost)])
plt.plot(cumsum_cost)
plt.show()

# cost daily pred
selected_day = 12
cumsum_cost = np.cumsum(running_total_cost)
cost_daily_pred_selected_day = cost_daily_pred[selected_day] @ I_upper

c_shape = cost_daily_pred_selected_day.shape

cost_daily_pred_selected_day_shift = np.append(
    np.ones((c_shape[0], 1)) * cumsum_cost[selected_day],
    cost_daily_pred_selected_day + np.ones(c_shape)*cumsum_cost[selected_day],
    axis=1
)

y_ref_daily_seleted = np.append(cumsum_cost[selected_day], y_ref_array[selected_day])

days = list(range(selected_day, selected_day+N+1))
plt.figure(figsize=(14,10))
plt.plot(selected_day, cumsum_cost[selected_day], 'o')
plt.plot(y_target[0:len(cumsum_cost)], 'r', linewidth=3)
plt.plot(cumsum_cost, 'b', linewidth=3)
plt.plot(
    days,
    np.transpose(cost_daily_pred_selected_day_shift),
    alpha=.25,
    color='green',
    linewidth=0.5
)
plt.plot(days,y_ref_daily_seleted, '*-', c='k', linewidth=3)
plt.xlim([11, 20]); plt.ylim([6900, 15000])
plt.show()

# Cost linearization


# Bid sequences
for plot_day in range(0,T-N):
    plt.figure()
    for plot_index in range(n_slots):
        plt.plot(np.arange(0,N), u_values[plot_day][plot_index,:])




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


#
plt.plot(ad_opportunities_rate_array)
plt.show()