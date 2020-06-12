import numpy as np

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

    # Update alpha and beta (from Karlsson page 30, equation (24)+(25), cpc_inv given by (26)) 
    alpha, beta = mpc.update_cpc_variables(cost, clicks)

    # Sample cpc_inv from gamma posterior, cpc_inv ~ Gamma(α(k), β(k))
    cpc_inv = mpc.draw_cpc_inv(alpha, beta)

    # Find expression for cost as linear function of u: dCost/du=a, if cost is given by Cost=a*u+b.
    # We use weighted linear Bayesian regression (newest observations most important)
    a, b = mpc.cost_linearization() # outputs a^omega, b^omega for omega=1,...,n_omega. (for each adslot of course)

    # Construct A (trivial)
    A = np.eye(2)

    # Contruct B from gradients of x=[clicks, cost] w.r.t. input
    grad_cost = a # 
    grad_clicks = cpc_inv * a
    B = np.array([grad_clicks, grad_cost]) # maybe this is just a B^omega

    # Calculate matrix

    # Solve convex optimization problem using CVXPY

    # Update nominal bid
    mpc.set_bid_multiplier(u)