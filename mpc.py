import numpy as np
from scipy import stats
from scipy.stats import gamma
from misc import StanModel_cache

from typing import Dict, List

class MPC():
    def __init__(
            self,
            ctr_mu: float,
            n_slots: int,
            ad_opportunities_params: Dict,
            ad_opportunities_rate_initial: np.ndarray,
            b_star_params: Dict,
            b_star_initial: np.ndarray,
            ctr_params: Dict,
            ctr_initial: np.ndarray
    ) -> None:
        self.ctr_mu = ctr_mu
        self.n_slots = n_slots
        self.ad_opportunities_params = ad_opportunities_params
        self.ad_opportunities_rate = ad_opportunities_rate_initial
        self.b_star_params = b_star_params
        self.b_star = b_star_initial
        self.ctr_params = ctr_params
        self.ctr = ctr_initial

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
            x_old: np.ndarray,
            mu: float,
            lamba: float,
            delta: float,
            p: float,
            upper_bound: float,
            lower_bound: float
    ) -> np.ndarray:
        """
        Function for propagating time varying parameters according to a stochastic difference equation
        :param x_old: current x value
        :param mu: asymptotic mean
        :param lamba: decay/growth rate
        :param delta: size of noise
        :param p: x exponent
        :param upper_bound: reflection boundary
        :param lower_bound: reflection boundary
        """

        dim_x = len(x_old)

        # Compute drift term
        drift_term = - lamba*(x_old - mu)

        # Compute diffusion term
        diffusion_term = delta*(x_old**p)*np.random.randn(dim_x)

        updated_random_walk = x_old + drift_term + diffusion_term
        
        # Reflect output in lower bound
        lb_diff = updated_random_walk - lower_bound
        updated_random_walk[lb_diff < 0] = updated_random_walk[lb_diff < 0] - 2*lb_diff[lb_diff < 0]

        ub_diff = updated_random_walk - upper_bound
        updated_random_walk[ub_diff > 0] = updated_random_walk[ub_diff > 0] - 2*ub_diff[ub_diff > 0]
        
        # https://benjaminmoll.com/wp-content/uploads/2019/07/Lecture4_2149.pdf
        return updated_random_walk


    def update_market(self) -> None:
        """
        Evolves the underlying market parameters.
        :param ti: initial time
        :param tf: final time
        :param n: number of time steps
        """
        # Update expected number of impression opportunities for each adslot
        self.ad_opportunities_rate = self.sde_walk(
            self.ad_opportunities_rate,
            self.ad_opportunities_params["mu"],
            self.ad_opportunities_params["lamba"],
            self.ad_opportunities_params["delta"],
            self.ad_opportunities_params["p"],
            self.ad_opportunities_params["upper_bound"],
            self.ad_opportunities_params["lower_bound"]
        )

        # Update expectation value of competitor bids for each adslot
        self.b_star = self.sde_walk(
            self.b_star,
            self.b_star_params["mu"],
            self.b_star_params["lamba"],
            self.b_star_params["delta"],
            self.b_star_params["p"],
            self.b_star_params["upper_bound"],
            self.b_star_params["lower_bound"]
        )

        # Update CTR of each adslot
        self.ctr = self.sde_walk(
            self.ctr,
            self.ctr_params["mu"],
            self.ctr_params["lamba"],
            self.ctr_params["delta"],
            self.ctr_params["p"],
            self.ctr_params["upper_bound"],
            self.ctr_params["lower_bound"]
        )

        return None


    def nb_samples(self, mu: np.ndarray, dispersion: np.ndarray, size: int) -> np.ndarray:
        """
        Computes samples for the negative binomial distribution with mu/phi parameterization

        Args:
            mu: numpy array, beware of broadcasting!
            dispersion: numpy array, beware of broadcasting!
            size: number of samples

        Returns:
            Numpy array.
        """

        return np.random.poisson(np.random.gamma(shape=dispersion, scale=mu / dispersion, size=size))


    def simulate_data(self) -> Dict: # Tuple(np.ndarray, np.ndarray, np.ndarray)
        """
        Observe cost, impressions and click from the auction and ad serving.
        """

        # Draw number of ad opportunities from poisson distribution with mean given by mean-reverting sde
        imps = self.nb_samples(
            self.ad_opportunities_rate,
            self.ad_opportunities_params["phi"]*self.ad_opportunities_rate,
            self.n_slots
        )
        # No need to draw competitors bid, just use his random walk. self.b_star is given

        # Heisenberg bidding
        #our_bid = heisenberg_bidding(u)  # ()

        # TODO: Simulate impressions. Did we win the aucion?
        #imps = np.sum(our_bid > self.b_star)  # for each adslot of course

        # build ad data dict
        ad_data = {
        #    'cost': cost,
            'imps': imps
        #    'clicks': clicks
        }

        return ad_data

    def update_cpc_variables(
            self,
            lam_cpc_vars: float,
            alpha_old: float,
            beta_old: float,
            cost: float,
            clicks: float
    ) -> Dict:
        """
        :param lam_cpc_vars: forgetting factor
        :param alpha_old: shape parameter
        :param beta_old: scale parameter
        :param cost: observed cost
        :param clicks: observed number of clicks
        :return cpc_variables: clicks and cost collected
        """

        alpha = lam_cpc_vars*alpha_old + clicks
        beta = lam_cpc_vars*beta_old + cost

        cpc_variables = {
            "alpha": alpha,
            "beta": beta
        }

        return cpc_variables

    def draw_cpc_inv(
            self,
            alpha_0,
            alpha,
            beta_0,
            beta
    ) -> float:

        cpc_inv = gamma.rvs(
            alpha_0 + alpha,
            beta_0 + beta,
            self.n_slots
        )

        return cpc_inv

    def cost_linearization(
            self,
            cost,
            u
    ) -> Dict:

        # Stan initialization
        iterations = 5000
        warms = 1000
        chains = 4

        # create dict for Stan
        stan_data = {
            "N": self.n_slots,
            "cost": cost,
            "bid": u
        }

        # define Stan file or use cached model
        stanfile = 'stanfiles/cost_linearization.stan'
        model = StanModel_cache(model_file=stanfile)

        # run Stan model
        fit = model.sampling(
            data=stan_data,
            iter=warms + iterations,
            chains=chains,
            warmup=warms
        )

        # obtain parameter est
        params = fit.extract()
        a = params['a']
        b = params['b']

        # collect parameters in dict
        cost_params = {"a": a, "b": b}

        return cost_params
