import numpy as np
import matplotlib.pyplot as plt


class ControlRoom:
    def __init__(
            self,
            running_total_cost: np.ndarray,
            y_target: np.ndarray,
            slope_array_mean: np.ndarray,
            clicks_array: np.ndarray,
            bstar_array: np.ndarray,
            ctr_array: np.ndarray,
            invcpc_array: np.ndarray,
            imps_array: np.ndarray,
            bid_array: np.ndarray,
            cost_array: np.ndarray,
            alpha_array: np.ndarray,
            beta_array: np.ndarray,
            bid_uncertainty_array: np.ndarray,
            mean_terms: np.ndarray,
            variance_terms: np.ndarray

    ) -> None:
        self.running_total_cost = running_total_cost
        self.y_target = y_target
        self.slope_array_mean = slope_array_mean
        self.clicks_array = clicks_array
        self.bstar_array = bstar_array
        self.ctr_array = ctr_array
        self.invcpc_array = invcpc_array
        self.imps_array = imps_array
        self.bid_array = bid_array
        self.cost_array = cost_array
        self.alpha_array = alpha_array
        self.beta_array = beta_array
        self.bid_uncertainty_array = bid_uncertainty_array
        self.mean_terms = mean_terms
        self.variance_terms = variance_terms

    def show_control_room(self) -> None:
        """
        display show_control_room
        """

        # show the control room
        fig, axs = plt.subplots(4, 3, sharex=True)
        cumsum_cost = np.cumsum(self.running_total_cost)  # accumulated sum of cost

        axs[0, 0].plot(cumsum_cost)
        axs[0, 0].plot(self.y_target[:len(cumsum_cost)])
        axs[0, 0].set_title('Cost evolution')
        axs[0, 1].plot(self.slope_array_mean)
        axs[0, 1].set_title('Cost gradients')
        axs[0, 2].plot(self.clicks_array)
        axs[0, 2].set_title('Clicks')
        axs[1, 0].plot(self.bstar_array)
        axs[1, 0].set_title(r'$b^*$')
        axs[1, 1].plot(self.ctr_array)
        axs[1, 1].set_title('CTR')
        axs[1, 2].plot(self.invcpc_array)
        axs[1, 2].set_title(r'$CPC^{-1}$')
        axs[2, 0].plot(self.imps_array)
        axs[2, 0].set_title('Impressions')
        axs[2, 1].plot(self.bid_array)
        axs[2, 1].set_title('Bids')
        axs[2, 2].plot(self.cost_array)
        axs[2, 2].set_title('Cost')
        axs[3, 0].plot(self.alpha_array)
        axs[3, 0].set_title(r'$\alpha$')
        axs[3, 1].plot(self.beta_array)
        axs[3, 1].set_title(r'$\beta$')
        axs[3, 2].set_title('Bid uncertainties')
        axs[3, 2].plot(self.bid_uncertainty_array)

        axs[3, 0].set_xlabel('Days')
        axs[3, 1].set_xlabel('Days')
        axs[3, 2].set_xlabel('Days')

        fig.show()

        return None

    def mean_vs_variance_obj(self) -> None:
        """
        display evolution of mean and variance objectives
        """

        plt.subplot(2, 1, 1)
        plt.title('Mean objective')
        plt.plot(self.mean_terms)

        plt.subplot(2, 1, 2)
        plt.title('Variance objective')
        plt.plot(self.variance_terms)

        plt.show()

        return None






