import numpy as np
from scipy import stats
from typing import Dict
from typing import List

class MPC():
    def __init__(
            self,
            ctr_mu: float

    ) -> None:
        self.ctr_mu = ctr_mu

    def wiener_process(
            self,
            dt: float,
            n: int
    ):
        """
        Function for creating Wiener process
        :param dt: time step
        :param n: number of time steps
        :return w: the Wiener process
        """
        dw = np.sqrt(dt) * np.random.randn(n)
        w = np.cumsum(dw)

        return w

    def sde_walk(
            self,
            ti: float,
            tf: float,
            n: int,
            x_old: np.ndarray,
            mu: float,
            lamba: float,
            delta: float,
            p: float
    ) -> np.ndarray:
        """
        Function for simulating stochastic differential equation
        :param ti: initial time
        :param tf: final time
        :param n: number of time steps
        :param x_old: current x value
        :param mu: asymptotic mean
        :param lamba: decay/growth rate
        :param delta: size of noise
        :param p: x exponent
        """

        # compute time steps
        dt = (tf-ti) / n
        # compute drift term
        drift_term = - lamba*(x_old-mu)

        # compute diffusion term
        w = self.wiener_process(dt, n)  # obtain Wiener process
        diffusion_term = delta*(x_old**p)*w

        updated_random_walk = drift_term + diffusion_term

        return updated_random_walk


    def update_market(
            self,
            ti,
            tf,
            n,
            n_slots
    ) -> None:
        """
        Evolves the underlying market parameters.
        :param ti: initial time
        :param tf: final time
        :param n: number of time steps
        :param n_slots: number of ad slots
        """
        # TODO: Update expected number of impression opportunities for each adslot
        mu_ad_opportunities = 50
        lamba_ad_opportunities = 0.99
        delta_ad_opportunities = 0.01
        p_ad_opportunities = 0.5

        self.ad_opportunities_rate = self.sde_walk(
            ti,
            tf,
            n,
            self.ad_opportunities_rate,
            mu_ad_opportunities,
            lamba_ad_opportunities,
            delta_ad_opportunities,
            p_ad_opportunities
        )

        # TODO: Update expectation value of competitor bids for each adslot
        mu_b_star = 5.
        lamba_b_star = 0.99
        delta_b_star = 0.02
        p_b_star = 0.5

        self.b_star = self.sde_walk(
            ti,
            tf,
            n,
            self.b_star,
            mu_b_star,
            lamba_b_star,
            delta_b_star,
            p_b_star
        )

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
        ad_data = {
            'cost': cost,
            'imps': imps,
            'clicks': clicks
        }

        return ad_data