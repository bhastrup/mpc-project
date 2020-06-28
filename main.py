import numpy as np
from mpc import MPC
from params import *

# construct MPC class
mpc = MPC(
    ctr_mu,
    n_slots,
    ad_opportunities_params,
    ad_opportunities_rate_initial,
    b_star_params,
    b_star_initial,
    ctr_params,
    ctr_initial
)

# Run the simulation
for i in range(0, T - N):

    mpc.update_market()

    ad_data = mpc.simulate_data()

    cost = ad_data["cost"]
    imps = ad_data["imps"]
    clicks = ad_data["clicks"]

    # Update alpha and beta cf. Karlsson p.30, Eq. [24] and [25]
    cpc_variables = mpc.update_cpc_variables(
        lam_cpc_vars,
        alpha_vec[i],
        beta_vec[i],
        cost,
        clicks
    )

    alpha = cpc_variables["alpha"]
    beta = cpc_variables["beta"]
    alpha_vec.append(alpha)
    beta_vec.append(beta)

    # Sample cpc_inv from gamma posterior, cpc_inv ~ Gamma(α(k), β(k))
    cpc_inv = mpc.draw_cpc_inv(
        alpha_0,
        alpha,
        beta_0,
        beta
    )

    # Find expression for cost as linear function of u: dCost/du=a, if cost is given by Cost=a*u+b.
    # We use weighted linear Bayesian regression (newest observations most important)
    cost_params = mpc.cost_linearization(
        cost,
        u
    )  # outputs a^omega, b^omega for omega=1,...,n_omega. (for each adslot of course)

    a = cost_params["a"]
    b = cost_params["b"]

    # Construct A (trivial)
    A = np.eye(2)

    # Contruct B from gradients of x=[clicks, cost] w.r.t. input
    grad_cost = a # 
    grad_clicks = cpc_inv * a
    B = np.array([grad_clicks, grad_cost]) # maybe this is just a B^omega

    # Calculate matrix

    # Specify rho, penalty of violating soft output constraint
    
    # Solve convex optimization problem using CVXPY

    # Update nominal bid
    mpc.set_bid_multiplier(u)