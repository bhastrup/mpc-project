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
for i in range(100):

    ad_data = mpc.simulate_data()
    cost = ad_data["cost"]

    # Update historic cost data
    past_costs = mpc.update_history(past_costs, cost)
    past_bids = mpc.update_history(past_bids, mpc.bid_price)

    # Update to new dum bid_price
    u = mpc.ctr*(1 + 0.1*np.random.randn(mpc.n_slots))
    mpc.set_bid_price(u)


# initialize arrays for historical data
running_total_cost = []
cost_array = []
mpc_cost_array = []
slope_array = []
slope_array_mean = []
intercept_array = []
ctr_array = []
bstar_array = []
ad_opportunities_rate_array = []
invcpc_array = []
clicks_array = []
imps_array = []
alpha_array = []
beta_array = []
bid_array = []
bid_pred = []
ustar_array = []
bu_array = []

u_values = []
u_star_values = []

cost_daily_pred = []
click_daily_pred = []
mean_terms = []
variance_terms = []
y_ref_array = []


for k in range(T - N):

    # 1. Evolve market parameters: ad_opportunities_rate, true ctr, and b_star
    market_params = mpc.update_market()
    ctr_array.append(market_params['ctr'])
    bstar_array.append(market_params['b_star'])
    ad_opportunities_rate_array.append(market_params['ad_opportunities_rate'])

    # 2. Simulate action data + ad serving
    ad_data = mpc.simulate_data()

    cost = ad_data["cost"]
    imps = ad_data["imps"]
    clicks = ad_data["clicks"]

    if k > 0:
        mpc.update_spend(cost)
        cost_array.append(cost)
        running_total_cost.append(sum(cost))
        imps_array.append(imps)
        clicks_array.append(clicks)
    else:
        running_total_cost.append(sum(cost) * 0)
        clicks_array.append(clicks * 0)
        imps_array.append(imps * 0)

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
    alpha_array.append(alpha)

    beta = cpc_variables["beta"]
    beta_array.append(beta)

    bu = mpc.set_bid_uncertainty(alpha)
    bu_array.append(bu)

    # 4. Sample cpc_inv from gamma posterior, cpc_inv ~ Gamma(α(k), β(k))
    cpc_inv = np.transpose(mpc.draw_cpc_inv(alpha, beta, n_samples))
    invcpc_array.append(np.mean(cpc_inv, axis=0))

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
    A_mat_all = np.array(cost_params['a'])
    A_mat = np.transpose(A_mat_all[:, :n_samples])
    slope_array.append(A_mat)
    slope_array_mean.append(np.mean(A_mat_all, axis=1))

    # cost intercepts, b^omega
    b_all = np.array(cost_params['b'])
    b = np.transpose(b_all[:, :n_samples])
    intercept_array.append(b)

    # Construct A (trivial)
    A = np.eye(2)

    # Calculate reference trajectory
    mpc_cost_array.append(mpc.cost)
    y_ref = np.linspace(mpc.cost, y_target[k+N], N+1)[1:]  # dim = N
    y_ref = np.outer(np.ones(n_samples), y_ref)  # dim = n_samples x N
    y_ref_array.append(y_ref[0,:])

    # Initialize MPC optimizer
    U = cp.Variable((mpc.n_slots, N))

    cost_daily = A_mat @ U + b @ I_intercept  # dim = n_samples x N

    # Construct mean objective
    click_daily = (cpc_inv * A_mat) @ U + (cpc_inv * b) @ I_intercept

    # Construct variance objective
    dev_list = []
    cost_accum = cost_daily @ I_upper + mpc.cost * np.ones((n_samples, N))
    dev_mat = cost_accum - y_ref

    for n in range(N):
        dev_list.append(
            dev_mat[:, n]
        )

    sum_dev_list = sum(q_vec[i] * dev_list[i] for i in range(N-1))

    # Solve MPC problem
    objective = cp.Minimize(
        - (alpha_mv / n_samples) * cp.sum(click_daily)
        + (1-alpha_mv) * cp.sum_squares(dev_mat @ Q_mat) / n_samples
    )

    # for testing magnitude of mean and variance objectives
    # term1 = - alpha_mv / n_samples * sum(click_daily)
    # term2 = (1-alpha_mv) * np.sum((dev_mat @ Q_mat)**2) / n_samples

    # Set constraints
    u_star = cost_params['u_star']
    u_lower_bound = np.outer(u_star, np.ones(N))
    u_upper_bounder = 2 * u_lower_bound

    constraints = [
        -u_lower_bound + 10**(-3) <= U,
        U <= u_upper_bounder
    ]

    # Construct the problem
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve(max_iter=100000)

    # The optimal value for U is stored in `U.value`.
    u_traj = U.value
    U_traj = np.zeros((n_slots, N))

    # Daily cost prediction
    for i in range(len(u_star)):
        U_traj[i, :] = u_traj[i, :] + u_star[i]

    u_values.append(U_traj)
    u_star_values.append(u_star)

    # Store historical mean and variance terms
    cost_daily_pred.append(
        (A_mat @ u_traj + b @ I_intercept)
    )

    click_daily_pred = (cpc_inv * A_mat) @ u_traj + (cpc_inv * b) @ I_intercept  # mean objective

    dev_list_pred = []
    cost_accum_pred = cost_daily_pred @ I_upper + mpc.cost * np.ones((n_samples, N))
    dev_mat_pred = cost_accum_pred - y_ref

    for n in range(N):
        dev_list_pred.append(
            dev_mat_pred[:, n]
        )

    sum_dev_list = sum(q_vec[i] * dev_list_pred[i] for i in range(N-1))

    mean_terms.append(- (alpha_mv / n_samples) * sum(sum(click_daily_pred)))
    variance_terms.append((1-alpha_mv) * np.sum((dev_mat_pred @ Q_mat)**2) / n_samples)

    # Calculate new bid
    new_bid = U.value[:, 0] + u_star

    bid_pred.append(U.value)
    ustar_array.append(u_star)
    bid_array.append(new_bid)

    # Update nominal bid
    mpc.set_bid_price(new_bid)




