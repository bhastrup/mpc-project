import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

class ControlRoom:
    def __init__(
            self,
            N: int,
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
            variance_terms: np.ndarray,
            cost_daily_pred: np.ndarray

    ) -> None:
        self.N = N
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
        self.cost_daily_pred = cost_daily_pred

    def show_control_room(self) -> None:
        """
        display show_control_room
        """

        # show the control room
        fig, axs = plt.subplots(4, 3, sharex=True)
        self.cumsum_cost = np.cumsum(self.running_total_cost)  # accumulated sum of cost

        axs[0, 0].plot(self.cumsum_cost)
        axs[0, 0].plot(self.y_target[:len(self.cumsum_cost)])
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

        axs[3, 0].set_xlabel('Time [days]')
        axs[3, 1].set_xlabel('Time [days]')
        axs[3, 2].set_xlabel('Time [days]')

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

    def cost_trajectory(self) -> None:
        """
        Display cost trajectory together with target trajectory
        """
        plt.plot(self.cumsum_cost, label='Accumulated cost')
        plt.plot(self.y_target[:len(self.cumsum_cost)], label='Target cost')
        plt.title('Cost trajectories')
        plt.legend()
        plt.xlabel('Time [Days]')
        plt.show()

        return None

    def prediction_horizon(
            self,
            selected_day: int,
            I_upper: np.ndarray,
            y_target: np.ndarray
    ) -> None:
        """
        Display cost trajectories with predicted cost
        """

        cost_daily_pred_selected_day = self.cost_daily_pred[selected_day] @ I_upper
        c_shape = cost_daily_pred_selected_day.shape

        cost_daily_pred_selected_day_shift = np.append(
            np.ones((c_shape[0], 1)) * self.cumsum_cost[selected_day],
            cost_daily_pred_selected_day + np.ones(c_shape) * self.cumsum_cost[selected_day],
            axis=1
        )

        days = list(range(selected_day, selected_day + self.N + 1))

        fig = plt.figure(figsize=(14, 10))
        ax = plt.axes()

        ax.plot(selected_day, self.cumsum_cost[selected_day], 'o')
        ax.plot(y_target[0:len(self.cumsum_cost)], 'r', linewidth=3, label='Target cost')
        ax.plot(self.cumsum_cost, 'b', linewidth=3, label='Accumulated cost')
        ax.plot(
            days,
            np.transpose(cost_daily_pred_selected_day_shift),
            alpha=.25,
            color='green',
            linewidth=0.5
        )
        axins = zoomed_inset_axes(ax, 1, loc=1)  # zoom = .2
        axins.plot(selected_day, self.cumsum_cost[selected_day], 'o')
        axins.plot(y_target[0:len(self.cumsum_cost)], 'r', linewidth=3, label='Target cost')
        axins.plot(self.cumsum_cost, 'b', linewidth=3, label='Accumulated cost')
        axins.plot(
            days,
            np.transpose(cost_daily_pred_selected_day_shift),
            alpha=.25,
            color='green',
            linewidth=0.5
        )
        axins.set_xlim(selected_day, 10)
        axins.set_ylim(6900, 8000)
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        mark_inset(ax, axins, loc1=2, loc2=2, fc="none", ec="0.5")
        plt.draw()

        plt.xlabel('Time [Days]')

        plt.show()

        return None

    def linearization_plots(
            self,
            selected_day: int,
            slope_array: np.array,
            intercept_array: np.ndarray,
            costs: np.ndarray,
            bids: np.ndarray,
            n_days_cost: np.ndarray,
            u_tilde: np.ndarray
    ) -> None:
        """
        dCost/du=a, if cost is given by Cost=a*u+b
        """

        # Define the data for selected day
        past_costs = costs[selected_day]
        past_bids = bids[selected_day]
        u_tildes = u_tilde[selected_day]
        slopes = slope_array[selected_day]
        intercepts = intercept_array[selected_day]

        # Use only the costs from one ad slots
        past_costs_one_adslot = past_costs[0, :]
        past_bids_one_adslot = past_bids[0, :]
        u_tilde_one_adslot = u_tildes[0]
        slopes_one_adslot = slopes[:, 0]
        intercepts_one_adslot = intercepts[:, 0]

        # the cost linearization
        cost_pred = np.transpose(np.outer(slopes_one_adslot, past_bids_one_adslot)) + intercepts_one_adslot

        # show cost versus bid plot
        plt.scatter(
            past_bids_one_adslot,
            past_costs_one_adslot,
            s=np.linspace(100, 5, n_days_cost),
            edgecolors='k'
        )

        plt.plot(
            past_bids_one_adslot,
            cost_pred
        )

        plt.xlabel('Bids')
        plt.ylabel('Cost')
        plt.show()

        # Define slopes and intercepts for the selected day and a week forward
        slopes_selected_days = slope_array[selected_day:selected_day+self.N]
        intercepts_selected_days = intercept_array[selected_day:selected_day+self.N]

        # Use one of the ad slots for visualization
        #slopes_adslot1 = slopes_selected_days[]
        #intercepts_adslot1 = intercepts_selected_days[]

        return None









