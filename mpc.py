import numpy as np
import random
from scipy import stats
from scipy.stats import gamma
from misc import StanModel_cache

from typing import Dict, List


class MPC:
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
        self.cost = 0
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
        self.eps = 0.0000001

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
        :param dw: increment
        :return updated_random_walk: the random walk
        """

        # Obtain dimensions of x
        dim_x = len(x_old)

        # Compute drift term
        drift_term = - lamba * (x_old - mu)

        # Compute diffusion term
        diffusion_term = delta * (x_old ** p) * dw

        # Update the random walk
        updated_random_walk = x_old + drift_term + diffusion_term

        # Reflect output in lower bound
        lb_diff = updated_random_walk - lower_bound
        updated_random_walk[lb_diff < 0] = updated_random_walk[lb_diff < 0] - 2 * lb_diff[lb_diff < 0]

        ub_diff = updated_random_walk - upper_bound
        updated_random_walk[ub_diff > 0] = updated_random_walk[ub_diff > 0] - 2 * ub_diff[ub_diff > 0]

        # https://benjaminmoll.com/wp-content/uploads/2019/07/Lecture4_2149.pdf
        return updated_random_walk

    def update_market(self) -> None:
        """
        Evolves the underlying market parameters.
        :return market_params: b_star and ctr values
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
        # print("ad_opportunities_rate")
        # print(self.ad_opportunities_rate)
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
            dw[:, 0]
        )
        # print("b_star")
        # print(self.b_star)
        # Update CTR of each adslot
        self.ctr = self.sde_walk(
            self.ctr,
            self.ctr_params["mu"],
            self.ctr_params["lamba"],
            self.ctr_params["delta"],
            self.ctr_params["p"],
            self.ctr_params["upper_bound"],
            self.ctr_params["lower_bound"],
            dw[:, 1]
        )

        markets_params = {
            'b_star': self.b_star,
            'ctr': self.ctr
        }

        return markets_params

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
        :return nb_samples: the obtained samples
        """

        nb_samples = np.random.poisson(
            np.random.gamma(
                shape=dispersion,
                scale=mu / (dispersion + self.eps)
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
        :param bid_price: nominal bid price
        :param bid_uncertainty: bid price uncertainty
        :param ad_opportunities: ad opportunities
        :return randomized_bids: randomized according to heisenberg bidding
        """

        randomized_bids = []

        for i in range(self.n_slots):
            randomized_bids.append(
                np.random.gamma(
                    shape=1 / bid_uncertainty[i] ** 2,
                    scale=bid_price[i] * bid_uncertainty[i] ** 2 + self.eps,
                    size=ad_opportunities[i]
                ).tolist()
            )

        return randomized_bids

    def simulate_data(self) -> Dict:  # Tuple(np.ndarray, np.ndarray, np.ndarray)
        """
        Observe cost, impressions and click from the auction and ad serving.
        :return ad_data: cost, imps and clicks
        """

        # Draw number of ad opportunities from neg_binom distribution with mean given by mean-reverting sde
        ad_opportunities = self.nb_samples(
            mu=self.ad_opportunities_rate,
            dispersion=self.ad_opportunities_params["phi"] * self.ad_opportunities_rate
        )

        # Heisenberg bidding, Karlsson page 26
        realized_bid = self.heisenberg_bidding(
            self.bid_price,
            self.bid_uncertainty,
            ad_opportunities
        )

        # Obtain competitors bid b_star
        realized_b_star = self.heisenberg_bidding(
            self.b_star,
            np.array([0.5, 0.5, 0.5]),
            ad_opportunities
        )
        # Calculate impressions won
        imps = np.asarray(
            [np.sum(np.asarray(realized_bid[i]) > np.asarray(realized_b_star[i])) for i in range(self.n_slots)]
        )

        # Calculate cost
        cost = np.asarray(
            [np.sum((np.asarray(realized_bid[i]) > np.asarray(realized_b_star[i]))*realized_b_star[i]) for i in range(self.n_slots)]
        )

        # Simulate clicks
        mu_clicks = imps * self.ctr
        disp_clicks = 2.0 * mu_clicks

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
        """update alpha and beta parameters
        :param lam_cpc_vars: forgetting factor
        :param alpha_old: shape parameter
        :param beta_old: scale parameter
        :param cost: observed cost
        :param clicks: observed number of clicks
        :return cpc_variables: clicks and cost collected
        """

        alpha = lam_cpc_vars * alpha_old + clicks
        beta = lam_cpc_vars * beta_old + cost

        cpc_variables = {
            "alpha": alpha,
            "beta": beta
        }

        return cpc_variables

    def draw_cpc_inv(
            self,
            alpha: np.ndarray,
            beta: np.ndarray,
            sample_size: int
    ) -> float:

        cpc_inv = np.zeros((self.n_slots, sample_size))

        for i in range(self.n_slots):
            cpc_inv[i, :] = np.random.gamma(
                shape=alpha[i],
                scale=1 / beta[i],
                size=sample_size
            )

        return cpc_inv

    def cost_linearization(
            self,
            costs: np.ndarray,
            bids: np.ndarray,
            weights: List,
            n_days_cost: int,
            n_samples: int
    ) -> Dict:
        """
        Find expression for cost as linear function of u:
        dCost/du=a, if cost is given by Cost=a*u+b.
        :param costs: array of last 14 days cost
        :param bids: array of last 14 days bids
        :param weights: list for weighting data points
        :param n_days_cost: days used
        :param n_samples: number of samples used
        :return cost_params: a, b and u_star parameters
        """

        a_params = []
        b_params = []

        u_star = []

        # define Stan file path
        stanfile = 'stanfiles/cost_linearization.stan'

        for slots_i in range(self.n_slots):
            u_tilde = bids[slots_i, :] - np.mean(bids[slots_i, :])
            u_star.append(np.mean(bids[slots_i, :]))

            # create dict for Stan
            stan_data = {
                "n_days": n_days_cost,
                "cost": costs[slots_i, :],
                "u": u_tilde,
                "weights": weights
            }

            # Compile or used the cached Stan model
            model = StanModel_cache(model_file=stanfile)

            # run Stan model
            fit = model.sampling(
                data=stan_data,
                chains=2,
                iter=n_samples * 40,
            )

            # Obbtain parameter estimates
            params = fit.extract()
            a = params['a']
            b = params['b']

            a_params.append(a)
            b_params.append(b)

        # Collect parameters in dict
        cost_params = {
            "a": a_params,
            "b": b_params,
            "u_star": u_star
        }

        return cost_params

    def dummy_cost_linearization(
            self,
            bids: np.ndarray,
            costs: np.ndarray
    ) -> Dict:
        """
        Cost linearization function for testing
        Find expression for cost as linear function of u:
        dCost/du=a, if cost is given by Cost=a*u+b.
        :param bids: bids in ad slots
        :param costs: costs in ad slots
        :return cost_param: intercept a and slope b collected
        """

        a_params = []
        b_params = []

        for i in range(self.n_slots):
            a, b, r_value, p_value, std_err = stats.linregress(bids[i, :], costs[i, :])

            a_params.append(a)
            b_params.append(b)

        # Collect parameters in dict
        cost_params = {"a": np.array(a_params), "b": b_params}

        return cost_params

    def set_bid_price(self, u: np.ndarray) -> None:
        """
        Updates the bid price that was calculated using Model Predictive Control
        :param u: First element of MPC calculated sequence of bid prices
        """

        self.bid_price = u

        for i in range(len(self.bid_price)):
            if self.bid_price[i] > 0.03:
                self.bid_price[i] = random.uniform(0.01, 0.03)


        # TODO: Define some constraints that prevents setting a dangerously high bid

        return None

    def set_bid_uncertainty(self, alpha: np.ndarray) -> None:
        """
        Updates the bid price uncertainty, according to Karlsson page 32, equation (28)
        param alpha: defined in Karlsson page 30, equation (24)
        """

        self.bid_uncertainty = alpha ** (-1 / 2)

        return self.bid_uncertainty

    def update_history(self, old_array: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Shifts all columns to the right, and puts x in the first column
        """

        new_array = np.roll(old_array, 1, axis=1)
        new_array[:, 0] = x

        return new_array
