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
            ctr_initial: np.ndarray,
            cov: float,
            bid_price_initial: np.ndarray,
            bid_uncertainty_initial: np.ndarray
    ) -> None:
        self.ctr_mu = ctr_mu
        self.n_slots = n_slots
        self.ad_opportunities_params = ad_opportunities_params
        self.ad_opportunities_rate = ad_opportunities_rate_initial
        self.b_star_params = b_star_params
        self.b_star = b_star_initial
        self.ctr_params = ctr_params
        self.ctr = ctr_initial
        self.cov = cov
        self.bid_price = bid_price_initial
        self.bid_uncertainty = bid_uncertainty_initial

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
            lower_bound: float,
            dw: np.ndarray
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

        # Obtain dimensions of x
        dim_x = len(x_old)

        # Compute drift term
        drift_term = - lamba*(x_old - mu)

        # Compute diffusion term
        diffusion_term = delta*(x_old**p)*dw

        # Update the random walk
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
        """
        # Update expected number of impression opportunities for each adslot
        self.ad_opportunities_rate = self.sde_walk(
            self.ad_opportunities_rate,
            self.ad_opportunities_params["mu"],
            self.ad_opportunities_params["lamba"],
            self.ad_opportunities_params["delta"],
            self.ad_opportunities_params["p"],
            self.ad_opportunities_params["upper_bound"],
            self.ad_opportunities_params["lower_bound"],
            np.random.randn(self.n_slots)
        )
        #print("ad_opportunities_rate")
        #print(self.ad_opportunities_rate)
        # Specify correlation between ctr and b_star
        mean = np.zeros(2)
        cov_matrix = np.array(
            [[1, self.cov],
            [self.cov, 1]]
        )
        # Draw correlated Wiener realization
        dw = np.random.multivariate_normal(
            mean=mean,
            cov=cov_matrix,
            size=self.n_slots
        )
        # Update expectation value of competitor bids for each adslot
        self.b_star = self.sde_walk(
            self.b_star,
            self.b_star_params["mu"],
            self.b_star_params["lamba"],
            self.b_star_params["delta"],
            self.b_star_params["p"],
            self.b_star_params["upper_bound"],
            self.b_star_params["lower_bound"],
            dw[:,0]
        )
        #print("b_star")
        #print(self.b_star)
        # Update CTR of each adslot
        self.ctr = self.sde_walk(
            self.ctr,
            self.ctr_params["mu"],
            self.ctr_params["lamba"],
            self.ctr_params["delta"],
            self.ctr_params["p"],
            self.ctr_params["upper_bound"],
            self.ctr_params["lower_bound"],
            dw[:,1]
        )
        #print("ctr")
        #print(self.ctr)

        return None


    def nb_samples(
            self,
            mu: np.ndarray,
            dispersion: np.ndarray
    ) -> np.ndarray:
        """
        Computes samples for the negative binomial distribution with mu/phi parameterization:
            https://mc-stan.org/docs/2_19/functions-reference/nbalt.html

        :param mu: mean value
        :param dispersion: excess variance relative to poisson distribution
        :param size: number of samples
        """

        eps = 0.0000001

        nb_samples = np.random.poisson(
            np.random.gamma(
                shape=dispersion,
                scale=mu / (dispersion + eps)
            )
        )

        return nb_samples


    def heisenberg_bidding(
            self,
            bid_price: np.ndarray,
            bid_uncertainty: np.ndarray,
            ad_opportunities: np.ndarray
    ) -> List:
        """
        Randomizes bids to smoothen plant gain related to impressions won vs bid price
        param: bid_price: nominal bid price
        param: bid_uncertainty: bid price uncertainty
        """

        randomized_bids = []

        for i in range(self.n_slots):
            randomized_bids.append(
                np.random.gamma(
                    shape=1/bid_uncertainty[i]**2,
                    scale=bid_price[i]*bid_uncertainty[i]**2,
                    size=ad_opportunities[i]
                ).tolist()
            )

        return randomized_bids

    def simulate_data(self) -> Dict:  # Tuple(np.ndarray, np.ndarray, np.ndarray)
        """
        Observe cost, impressions and click from the auction and ad serving.
        """

        # Draw number of ad opportunities from neg_binom distribution with mean given by mean-reverting sde
        ad_opportunities = self.nb_samples(
            mu=self.ad_opportunities_rate,
            dispersion=self.ad_opportunities_params["phi"]*self.ad_opportunities_rate
        )

        # Heisenberg bidding, Karlsson page 26
        realized_bid = self.heisenberg_bidding(
            self.bid_price,
            self.bid_uncertainty,
            ad_opportunities
        )

        # No need to draw competitors bid, just use their random walk. self.b_star is given

        # Calculate impressions won
        imps = np.asarray(
            [np.sum(np.asarray(realized_bid[i]) > self.b_star[i]) for i in range(self.n_slots)]
        )

        print("imps")
        print(imps)
        # Calculate cost
        cost = imps * self.b_star
        print("cost")
        print(cost)
        # Simulate clicks
        mu_clicks = imps * self.ctr
        disp_clicks = 1.0 * mu_clicks
        print("disp_clicks")
        print(disp_clicks)

        clicks = self.nb_samples(
            mu=mu_clicks,
            dispersion=disp_clicks
        )
        ad_data = {
            'cost': cost,
            'imps': imps,
            'clicks': clicks
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
            alpha,
            beta
    ) -> float:

        cpc_inv = np.random.gamma(
            shape=alpha,
            scale=beta,
            size=self.n_slots
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


    def set_bid_price(self, u: np.ndarray) -> None:
        """
        Updates the bid price that was calculated using Model Predictive Control
        param u: Entire control sequence of bid prices
        """

        self.bid_price = u[:,0]

        # TODO: Define some constraints that prevents setting a dangerously high bid
        
        return None


    def set_bid_uncertainty(self, alpha: float) -> None:
        """
        Updates the bid price uncertainty, according to Karlsson page 32, equation (28)
        param alpha: defined in Karlsson page 30, equation (24)
        """

        self.bid_uncertainty = alpha**(-1/2)
        
        return None
