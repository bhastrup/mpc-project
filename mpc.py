
import numpy as np
import stats

n_slots
def update_market() -> None:
    """
    Evolves the underlying market parameters.
    :param param_name
    """
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
    
    # Simulate impressions
    I = np.abs(np.floor(stats.norm.rvs(loc=250,scale=10, size=(n_slots,n_days), random_state=1)))


    # build dict
    #cost, imps, clicks
    return ad_data