import numpy as np
from mpc import MPC

# Define parameters
N = 10  # Time horizon.
T = 100  # Campaign length.
n = 1000  # number of time steps.
t = np.linspace(0., T, n)  # vector of times.

ctr_mu = 0.01  # mean ctr value.
n_slots = 10  # number of ad slots

# construct class
mpc = MPC(ctr_mu)

# run simulation
for i in range(0,T-N):
    ti = t[i]
    tf = tf[i+N]

    mpc.update_market(
        ti,
        tf,
        N
    )

    cost, imps, clicks = mpc.simulate_data()

    # Update CPC from Karlsson method
    
    # Calculate matrix


    # Update nominal bid
    mpc.set_bid_multiplier(u)