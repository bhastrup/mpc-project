
N = 10 # Time horizon
T = 100 # Campaign length
ctr_mu = 0.01
mpc = MPC(
    N = N,
    T = T,
    ctr_mu = ctr_mu
)

for i in range(0,T-N):

    mpc.update_market()
    cost, imps, clicks = mpc.simulate_data()
