import numpy as np

# Define MPC parameters
N = 10  # Time horizon.
T = 100  # Campaign length.
n = 1000  # number of time steps.
t = np.linspace(0., T, n)  # vector of times.

# Define general ad parameters
n_slots = 10  # number of ad slots
cov = 0.8 # Covariance between b_star and ctr

# Define click-through-rate parameters
ctr_mu = 0.01  # mean ctr value.
ctr_initial = ctr_mu + 0.2*ctr_mu*np.random.randn(n_slots)
ctr_lambda = 0.0025
ctr_delta = 0.004
ctr_p = 0.5
ctr_ub = 3*ctr_mu
ctr_lb = 1/3*ctr_mu

# Setup CTR dictionary
ctr_params = {
    "mu": ctr_mu,
    "lamba": ctr_lambda,
    "delta": ctr_delta,
    "p": ctr_p,
    "upper_bound": ctr_ub,
    "lower_bound": ctr_lb
}

# Define parameters related to opportunities
ad_opportunities_mu = 50
ad_opportunities_rate_initial = ad_opportunities_mu + 10*np.random.randn(n_slots)
ad_opportunities_lambda = 0.1
ad_opportunities_delta = 0.1
ad_opportunities_p = 0.5
ad_opportunities_ub = 5 * ad_opportunities_mu
ad_opportunities_lb = 0.5 * ad_opportunities_mu
ad_opportunities_phi = 2

# Setup the ad opportunity dictionary
ad_opportunities_params = {
    "mu": ad_opportunities_mu,
    "lamba": ad_opportunities_lambda,
    "delta": ad_opportunities_delta,
    "p": ad_opportunities_p,
    "upper_bound": ad_opportunities_ub,
    "lower_bound": ad_opportunities_lb,
    "phi": ad_opportunities_phi
}

# Define parameters related to highest competitive bid
b_star_mu = ctr_mu
b_star_initial = b_star_mu + 0.2*b_star_mu*np.random.randn(n_slots)
b_star_lambda = 0.1
b_star_delta = 0.1
b_star_p = 0.5
b_star_ub = 5*b_star_mu
b_star_lb = 0.2*b_star_mu

# Setup the b_star dictionary
b_star_params = {
    "mu": b_star_mu,
    "lamba": b_star_lambda,
    "delta": b_star_delta,
    "p": b_star_p,
    "upper_bound": b_star_ub,
    "lower_bound": b_star_lb
}

bid_price_initial = ctr_mu

# Define cost-per-click parameters
lam_cpc_vars = 0.9  # forgetting factor related to CPC var update

# gamma (CPC) distribution parameters
alpha_0 = 1
beta_0 = 1
alpha_vec = []
beta_vec = []
alpha_vec.append(alpha_0)
beta_vec.append(beta_0)
