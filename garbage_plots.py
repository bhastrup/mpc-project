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

plt.plot(past_bids[0, :], past_costs[0, :], '*')


#######
# our cost and the target budget
cumsum_cost = np.cumsum(running_total_cost)

plt.plot(cumsum_cost)
plt.plot(y_target)
plt.show()