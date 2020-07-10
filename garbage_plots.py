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
axs[0, 1].plot(slope_array)
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
axs[2, 1].set_ylim(0, 0.05)
axs[2, 1].set_title('bids')
axs[2, 2].plot(cost_array)
axs[2, 2].set_title('cost')
axs[3, 0].plot(alpha_array)
axs[3, 0].set_title('alpha')
axs[3, 1].plot(beta_array)
axs[3, 1].set_title('beta')
axs[3, 2].set_title('bid uncertainties')
axs[3, 2].plot(bu_array)

plt.show()



