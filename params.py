import numpy as np

# Define MPC parameters
N = 5  # Time horizon.
T = 25  # Campaign length.
n = 1000  # number of time steps.
t = np.linspace(0., T, n)  # vector of times.

# Define general ad parameters
n_slots = 4  # number of ad slots
cov = 0.75  # Covariance between b_star and ctr

# Define click-through-rate parameters
ctr_mu = 0.01  # mean ctr value.
ctr_initial = ctr_mu + 0.2*ctr_mu*np.random.randn(n_slots)
ctr_lambda = 0.0025
ctr_delta = 0.004
ctr_p = 0.5
ctr_ub = 4*ctr_mu
ctr_lb = 1/4*ctr_mu

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
ad_opportunities_mu = 15000
ad_opportunities_rate_initial = np.repeat(ad_opportunities_mu, n_slots)
ad_opportunities_lambda = 0.0001
ad_opportunities_delta = 2
ad_opportunities_p = 0.5
ad_opportunities_ub = 5 * ad_opportunities_mu
ad_opportunities_lb = 0.5 * ad_opportunities_mu
ad_opportunities_phi = .0001

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
b_star_initial = ctr_initial
b_star_lambda = 0.025
b_star_delta = 0.004
b_star_p = 0.5
b_star_ub = 4*b_star_mu
b_star_lb = 1/4*b_star_mu

# Setup the b_star dictionary
b_star_params = {
    "mu": b_star_mu,
    "lamba": b_star_lambda,
    "delta": b_star_delta,
    "p": b_star_p,
    "upper_bound": b_star_ub,
    "lower_bound": b_star_lb
}

bid_price_initial = ctr_initial
bid_uncertainty_initial = 0.5*np.ones(n_slots)  # 50% Heisenberg randomization

# Define cost-per-click parameters
lam_cpc_vars = 0.5  # forgetting factor related to CPC var update
n_samples = 50
# gamma (CPC) distribution parameters
alpha = 1
beta = 1

# Define weight array for cost linearization
n_days_cost = 14
decaying_rate = 0.95
weights = [decaying_rate ** i for i in range(n_days_cost)]

# Initialize historic bids and costs for cost_linearization
past_costs = np.zeros((n_slots, n_days_cost))
past_bids = np.zeros((n_slots, n_days_cost))


# Campaign budget
budget = 7500
y_target = np.linspace(0, budget, T+1)


# Build unit matrix for broadcasting of b
I_intercept = np.ones((n_slots, N))  # dim = n_slots x N

# Build upper triangular matrix of ones
I_upper = np.zeros((N, N))
upper_triangle_indices = np.triu_indices(N)
I_upper[upper_triangle_indices] = 1  # dim = N x N

# Define Q matrix
q_vec = np.linspace(1, 3, N) / np.sum(np.linspace(1, 3, N))
Q_mat = np.diag(q_vec)

# Initialization for the MPC optimization
day_mat = np.eye(N)

# Mean variance constant
alpha_mv = 0.05

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
invcpc_all_array = []
clicks_array = []
imps_array = []
alpha_array = []
beta_array = []
bid_array = []
bid_pred = []
ustar_array = []
u_tilde_array = []
bid_uncertainty_array = []

u_values = []
u_star_values = []
past_costs_array = []
past_bids_array = []

cost_daily_pred = []
click_daily_pred = []
mean_terms = []
variance_terms = []
y_ref_array = []

alpha_new_array = []
beta_new_array = []
alpha_mean_array = []
beta_mean_array = []

# Set day for predictions
selected_day = 10
