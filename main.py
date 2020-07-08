import numpy as np
from params import *
from mpc import MPC

from cvxpy_diagonalizing import *
import cvxpy as cp

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

# 0. Initialize campaign without MPC informed bidding
for i in range(1000):

    ad_data = mpc.simulate_data()
    cost = ad_data["cost"]

    # Update historic cost data
    past_costs = mpc.update_history(past_costs, cost)
    past_bids = mpc.update_history(past_bids, mpc.bid_price)

    # Update to new dum bid_price
    u = mpc.ctr*(1 + 0.1*np.random.randn(mpc.n_slots))
    mpc.set_bid_price(u)


# Run the simulation
T = 30
k = 0
for k in range(T - N):

    # 1. Evolve market parameters: ad_opportunities_rate, true ctr, and b*
    mpc.update_market()

    # 2. Simulate action data + ad serving
    ad_data = mpc.simulate_data()

    cost = ad_data["cost"]
    imps = ad_data["imps"]
    clicks = ad_data["clicks"]

    past_costs = mpc.update_history(past_costs, cost)
    past_bids = mpc.update_history(past_bids, mpc.bid_price)

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
    cpc_inv = mpc.draw_cpc_inv(alpha, beta, n_samples)

    # 5. Linearization of cost using weighted Bayesian regression using last 10 obs
    cost_params = mpc.cost_linearization(
        costs=past_costs,
        bids=past_bids,
        weights=weights,
        n_days_cost=n_days_cost,
        n_samples=n_samples
    )

    # Extract slope and intercept, both dim = n_samples x n_slots

    # cost slopes, a^omega
    A_mat_all = np.array(cost_params["a"])
    A_mat = np.transpose(A_mat_all[:, :n_samples])

    # cost intercepts, b^omega
    b_all = np.array(cost_params["b"])
    b = np.transpose(b_all[:, :n_samples])

    # Construct A (trivial)
    A = np.eye(2)

    # Calculate reference trajectory
    y_ref = np.linspace(mpc.cost, y_target[k+N], N+1)[1:]  # dim = N
    y_ref = np.outer(np.ones(n_samples), y_ref)  # dim = n_samples x N

    # Initialize MPC optimizer
    U = cp.Variable((mpc.n_slots, N))

    dev_list = []
    dev_mat = (((A_mat @ U) + (b @ I_intercept)) @ I_upper) - y_ref

    for n in range(N):
        dev_list.append(
            ((((A_mat @ U) + (b @ I_intercept)) @ I_upper) - y_ref) * day_mat[:, n]
        )

    sum_dev_list = sum(q_vec[i] * dev_list[i] for i in range(N-1))

    objective = cp.Minimize(
        cp.sum_squares(dev_mat * Q_mat)
    )

    # Set constraints
    u_star = cost_params['u_star']
    u_lower_bound = np.outer(u_star, np.ones(N))
    u_upper_bounder = 2 * u_lower_bound

    constraints = [-u_lower_bound <= U, U <= u_upper_bounder]

    # Construct the problem
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve(max_iter=50000)

    # The optimal value for U is stored in `U.value`.
    print(U.value)

    # Contruct B from gradients of x=[clicks, cost] w.r.t. input
    grad_cost = a
    grad_clicks = cpc_inv * np.outer(a, np.ones(n_samples))
    B = np.array([grad_clicks, grad_cost]) # maybe this is just a B^omega


    # Update nominal bid
    mpc.set_bid_price(U.value[:, 0])
