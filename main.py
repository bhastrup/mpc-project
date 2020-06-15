import numpy as np
from mpc import MPC

# Define MPC parameters
N = 10  # Time horizon.
T = 100  # Campaign length.
n = 1000  # number of time steps.
t = np.linspace(0., T, n)  # vector of times.

# Define ad parameters
ctr_mu = 0.01  # mean ctr value.
n_slots = 10  # number of ad slots
lam_cpc_vars = 0.9  # forgetting factor related to cpc var update

# gamma (cpc) distribution parameters
alpha_0 = 1
beta_0 = 1
alpha_vec = []
beta_vec = []
alpha_vec.append(alpha_0)
beta_vec.append(beta_0)

# construct class
mpc = MPC(
    ctr_mu,
    n_slots
)

# run simulation
for i in range(0, T - N):
    ti = t[i]
    tf = tf[i + N]

    mpc.update_market(
        ti,
        tf,
        N
    )

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