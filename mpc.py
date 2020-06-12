
import numpy as np
from scipy import stats

# Auction parameters
n_slots = 10 # int const
ad_opportunities_rate_mean = 100 # int const
ad_opportunities_rate # n_slots dimensional random walk around ad_opportunities_rate_mean
b_star # highest competitive bid price random around ctr_mu. dim=n_slot

# Ad serving
ctr_mu # float const
ctr # n_slots dimensional random walk around ctr_mu

# Modeling estimates
alpha # float vector, adaptive measure for amount of clicks
beta # float vector, adaptive measure for amount of cost

def sde_walk(
    x_old: np.ndarray,
    mu: float,
    lamba: float,
    up_lim: float,
    low_lim: float
) -> np.ndarray:


    return updated_random_walk


def update_market() -> None:
    """
    Evolves the underlying market parameters.
    :param param_name
    """
    # TODO: Update expected number of impression opportunies for each adslot
    self.ad_opportunities_rate = sde_walk(self.ad_opportunities_rate)

    # TODO: Update expectation value of competitor bids for each adslot
    self.b_star = sde_walk(self.bstar)

    self.ctr += stats.norm.rvs(loc=0, scale=0.1*self.ctr_mu, size=self.n_slots)
    self.cpc += 1

    self.ctr += 1
    self.cpc += 1

    # Simulate CTR's around group mean
    sigma = 0.5*mu
    #CTR = np.random.normal(mu,sigma,   size=n_slots, random)
    CTR = stats.norm.rvs(loc=self.ctr_mu, scale=sigma, size=n_slots)
    
    if np.mean(CTR<0) > 0:
        print("Increase mu or decrease sigma such that CTR is always positive")
    CTR = np.abs(CTR)
    return None

def simulate_data() -> Dict: # Tuple(np.ndarray, np.ndarray, np.ndarray)
    """
    Observe cost, impressions and click from the auction and ad serving.
    """
    
    # TODO: Draw number of ad opportunities from poisson distribution with mean given by mean-reverting sde

    # 
    # Simulate impressions
    imps = 
    np.abs(np.floor(stats.norm.rvs(loc=250,scale=10, size=self.n_slots)))


    # build dict
    #cost, imps, clicks
    return ad_data