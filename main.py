import numpy as np
from params import *
from mpc import MPC
from control_room import ControlRoom

import cvxpy as cp
from cvxpy.atoms.elementwise.abs import abs as cp_abs
import random
random.seed(3)

# construct MPC class
mpc = MPC(
    ctr_mu=ctr_mu,
    n_slots=n_slots,
    ad_opportunities_params=ad_opportunities_params,
    ad_opportunities_rate_initial=ad_opportunities_rate_initial,
    b_star_params=b_star_params,
    b_star_initial=b_star_initial,
    ctr_params=ctr_params,
    ctr_initial=ctr_initial,
    cov=cov,
    bid_price_initial=bid_price_initial,
    bid_uncertainty_initial=bid_uncertainty_initial
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

N_fix = N
for k in range(T-1):

    if k > T-N_fix:
        print("Decrease N")
        N -= 1
        I_intercept = I_intercept[:,0:N]
        I_upper = I_upper[0:N, 0:N]
        Q_mat = np.diag(np.linspace(1, 3, N) / np.sum(np.linspace(1, 3, N)))

    # Are we approacing the terminal date and need to decrease N?

    # 1. Evolve market parameters: ad_opportunities_rate, true ctr, and b_star
    market_params = mpc.update_market()

    # 1.b Save market parameters
    ctr_array.append(market_params['ctr'])
    bstar_array.append(market_params['b_star'])
    ad_opportunities_rate_array.append(market_params['ad_opportunities_rate'])

    # 2. Simulate action data + ad serving
    ad_data = mpc.simulate_data()

    # 2.b, Unfold, update and save cost/bid regression data
    cost = ad_data["cost"]
    imps = ad_data["imps"]
    clicks = ad_data["clicks"]
    past_costs = mpc.update_history(past_costs, cost)
    past_bids = mpc.update_history(past_bids, mpc.bid_price)
    past_costs_array.append(past_costs)
    past_bids_array.append(past_bids)
    
    # 2.c, Save cost, imp and click data
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

    # 3. Update alpha and beta cf. Karlsson p.30, Eq. [24] and [25] and set bid_uncertainty
    cpc_variables = mpc.update_cpc_variables(
        lam_cpc_vars,
        alpha,
        beta,
        cost,
        clicks
    )

    # 3.b, Unfold the CPC variables
    alpha = cpc_variables["alpha"]
    alpha_array.append(alpha)
    beta = cpc_variables["beta"]
    beta_array.append(beta)

    # 4, Set bid uncertainty using u_u = 1/(sqrt(alpha))
    bu = mpc.set_bid_uncertainty(alpha)

    # 4.b, Save bid uncertainty
    bid_uncertainty_array.append(bu)

    # 5. Sample cpc_inv from gamma posterior, cpc_inv ~ Gamma(α(k), β(k))
    cpc_inv = np.transpose(mpc.draw_cpc_inv(alpha, beta, n_samples))
    invcpc_array.append(np.mean(cpc_inv, axis=0))

    # 6. Linearization of cost using weighted Bayesian regression using last 14 obs
    cost_params = mpc.cost_linearization(
        costs=past_costs,
        bids=past_bids,
        weights=weights,
        n_days_cost=n_days_cost,
        n_samples=n_samples
    )

    # 6.b, Extract slope and intercept, both dim = n_samples x n_slots
    # cost slopes, a^omega
    A_mat_all = np.array(cost_params['a'])
    A_mat = np.transpose(A_mat_all[:, :n_samples])
    slope_array.append(A_mat)
    slope_array_mean.append(np.mean(A_mat_all, axis=1))

    # cost intercepts, b^omega
    b_all = np.array(cost_params['b'])
    b = np.transpose(b_all[:, :n_samples])
    intercept_array.append(b)

    # 6.c, Save data
    alpha_new_array.append(cost_params['alpha'])
    beta_new_array.append(cost_params['beta'])
    alpha_mean_array.append(cost_params['alpha_means'])
    beta_mean_array.append(cost_params['beta_means'])


    # 7. Calculate reference trajectory
    mpc_cost_array.append(mpc.cost)
    y_ref = np.linspace(mpc.cost, y_target[k+N], N+1)[1:]  # dim = N
    y_ref = np.outer(np.ones(n_samples), y_ref)  # dim = n_samples x N
    y_ref_array.append(y_ref[0, :])

    # Initialize MPC optimizer
    U = cp.Variable((mpc.n_slots, N))

    # 8. Construct mean objective
    click_daily = (cpc_inv * A_mat) @ U + (cpc_inv * b) @ I_intercept

    # 9. Construct variance objective
    cost_daily = A_mat @ U + b @ I_intercept  # dim = n_samples x N
    cost_accum = cost_daily @ I_upper + mpc.cost * np.ones((n_samples, N))
    dev_mat = cost_accum - y_ref

    # 10. Set constraints
    u_star = cost_params['u_star']
    u_lower_bound = - np.outer(u_star, np.ones(N)) + 1*10**(-3)
    u_upper_bound = 0.1*np.ones((n_slots, N)) - np.outer(u_star, np.ones(N))

    u_old = mpc.bid_price - u_star
    U_lag = cp.atoms.affine.hstack.hstack([np.outer(u_old, np.ones((1, 1))), U[:, :-1]])
    #cp.atoms.affine.diff.diff(U, k=1, axis=0).T

    constraints = [
        u_lower_bound <= U,
        U <= u_upper_bound,
        cp_abs(U-U_lag) <= 0.005
        #-0.1 <= U-U_lag,
        #U-U_lag <= 0.1
    ]

    # 10. Calculate MPC objective
    objective = cp.Minimize(
        - (alpha_mv / n_samples) * cp.sum(click_daily)
        + (1-alpha_mv) * (cp.sum_squares(dev_mat @ Q_mat) + cp.sum_squares(U-U_lag) ) / n_samples 
    )
    # Construct the problem
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve(max_iter=200000)

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
        A_mat @ u_traj + b @ I_intercept
    )

    click_daily_pred = (cpc_inv * A_mat) @ u_traj + (cpc_inv * b) @ I_intercept  # mean objective

    cost_accum_pred = (A_mat @ u_traj + b @ I_intercept) @ I_upper + mpc.cost * np.ones((n_samples, N))
    dev_mat_pred = cost_accum_pred - y_ref

    mean_terms.append( (alpha_mv / n_samples) * sum(sum(click_daily_pred)))
    variance_terms.append((1-alpha_mv) * np.sum((dev_mat_pred @ Q_mat)**2) / n_samples**2)

    # Calculate new bid
    new_bid = U.value[:, 0] + u_star

    # append bid values
    bid_pred.append(U.value)
    ustar_array.append(u_star)
    bid_array.append(new_bid)
    u_tilde = cost_params['u_tildes']
    u_tilde_array.append(u_tilde)

    # Update nominal bid
    mpc.set_bid_price(new_bid)

print(sum(mean_terms))
print(sum(variance_terms))


# construct Control room
cr = ControlRoom(
    N=N_fix,
    running_total_cost=running_total_cost,
    y_target=y_target,
    slope_array_mean=slope_array_mean,
    clicks_array=clicks_array,
    bstar_array=bstar_array,
    ctr_array=ctr_array,
    invcpc_array=invcpc_array,
    imps_array=imps_array,
    bid_array=bid_array,
    cost_array=cost_array,
    alpha_array=alpha_array,
    beta_array=beta_array,
    bid_uncertainty_array=bid_uncertainty_array,
    mean_terms=mean_terms,
    variance_terms=variance_terms,
    cost_daily_pred=cost_daily_pred,
    y_ref_array=y_ref_array
)

# display control room
cr.show_control_room()

# display evolution of mean and variance terms
cr.mean_vs_variance_obj()

# display cost trajectories
cr.cost_trajectory()

# Build upper triangular matrix of ones
I_upper = np.zeros((N_fix, N_fix))
upper_triangle_indices = np.triu_indices(N_fix)
I_upper[upper_triangle_indices] = 1  # dim = N x N

# display cost trajectories with prediction_horizon
cr.prediction_horizon(
    selected_day=selected_day,
    I_upper=I_upper,
    y_target=y_target,
)

# display plots related to linearization
cr.linearization_plots(
    selected_day=selected_day,
    slope_array=slope_array,
    intercept_array=intercept_array,
    costs=past_costs_array,
    bids=past_bids_array,
    n_days_cost=n_days_cost,
    u_tilde=u_tilde_array,
    u_star=u_star_values,
    alpha_new_array=alpha_new_array,
    beta_new_array=beta_new_array,
    alpha_mean_array=alpha_mean_array,
    beta_mean_array=beta_mean_array
)

