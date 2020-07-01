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
    ctr_initial,
    cov,
    bid_price_initial,
    bid_uncertainty_initial
)

# Run the simulation
for i in range(0, T - N):

    # 1. Evolve market parameters: ad_opportunities_rate, true ctr, and b*
    mpc.update_market()

    # 2. Simulate action data + ad serving
    ad_data = mpc.simulate_data()

    cost = ad_data["cost"]
    imps = ad_data["imps"]
    clicks = ad_data["clicks"]

    # 3. Update alpha and beta cf. Karlsson p.30, Eq. [24] and [25] and set bid_uncertainty
    cpc_variables = mpc.update_cpc_variables(
        lam_cpc_vars,
        alpha,
        beta,
        cost,
        clicks
    )

    alpha = cpc_variables["alpha"]
    beta = cpc_variables["beta"]

    mpc.set_bid_uncertainty(alpha)

    # 4. Sample cpc_inv from gamma posterior, cpc_inv ~ Gamma(α(k), β(k))
    cpc_inv = mpc.draw_cpc_inv(alpha, beta)

    # 5. Linearization of cost using Bayesian regression
    cost_params = mpc.cost_linearization(
        cost,
        weights
    )

    # outputs a^omega, b^omega for omega=1,...,n_omega. (for each adslot of course)
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
    mpc.set_bid_price(u)