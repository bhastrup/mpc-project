import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns

#sns.set()  # Nice plot aesthetic

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
            cost_daily_pred: np.ndarray,
            y_ref_array: np.ndarray

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
        self.y_ref_array = y_ref_array

    def show_control_room(self) -> None:
        """
        display show_control_room
        """

        labels = ['ad slot 1', 'ad slot 2']

        # show the control room
        fig, axs = plt.subplots(5, 2, sharex=True)
        self.cumsum_cost = np.cumsum(self.running_total_cost)  # accumulated sum of cost

        #axs[0, 0].plot(self.cumsum_cost)
        #axs[0, 0].plot(self.y_target[:len(self.cumsum_cost)])
        #axs[0, 0].set_title('Cost evolution')
        axs[0, 0].plot(self.slope_array_mean)
        axs[0, 0].set_title('Cost gradients')
        axs[0, 1].plot(self.clicks_array)
        axs[0, 1].set_title('Clicks')
        axs[1, 0].plot(self.bstar_array)
        axs[1, 0].set_title(r'$b^*$')
        axs[1, 1].plot(self.ctr_array)
        axs[1, 1].set_title('CTR')
        axs[2, 0].plot(self.invcpc_array)
        axs[2, 0].set_title(r'$CPC^{-1}$')
        axs[2, 1].plot(self.imps_array)
        axs[2, 1].set_title('Impressions')
        axs[3, 0].plot(self.bid_array)
        axs[3, 0].set_title('Bids')
        #axs[2, 2].plot(self.cost_array)
        #axs[2, 2].set_title('Cost')
        axs[3, 1].plot(self.alpha_array)
        axs[3, 1].set_title(r'$\alpha$')
        axs[4, 0].plot(self.beta_array)
        axs[4, 0].set_title(r'$\beta$')
        axs[4, 1].set_title('Bid uncertainties')
        axs[4, 1].plot(self.bid_uncertainty_array)

        axs[4, 0].set_xlabel('Time [days]')
        axs[4, 1].set_xlabel('Time [days]')

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
            y_target: np.ndarray,
    ) -> None:
        """
        Display cost trajectories with predicted cost
        """
        y_ref_daily_seleted = np.append(
            self.cumsum_cost[selected_day],
            self.y_ref_array[selected_day]
        )

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
        ax.plot(
            y_target[0:len(self.cumsum_cost)],
            'r', linewidth=3,
            label='Target cost'
        )
        ax.plot(
            self.cumsum_cost,
            'b',
            linewidth=3,
            label='Accumulated cost'
        )
        cost_daily_pred_selected_day_shift_T = np.transpose(cost_daily_pred_selected_day_shift)
        preds = ax.plot(
            days,
            cost_daily_pred_selected_day_shift_T,
            alpha=.25,
            color='green',
            linewidth=0.5,
            label="Cost prediction"
        )
        plt.setp(preds[1:], label="_")
        ax.plot(
            days,
            y_ref_daily_seleted,
            '--',
            c='b',
            linewidth=3,
            label=r'$y_{ref}$'
        )

        ax.set_xlabel('Time',  fontsize=30)
        ax.set_ylabel('Cost',  fontsize=30)
        ax.set_ylim([0, 7500])
        ax.xaxis.set_tick_params(labelsize=30)
        ax.yaxis.set_tick_params(labelsize=30)
        ax.legend(loc=4, prop={'size': 20})

        axins = zoomed_inset_axes(ax, 2.25, loc=2)  # zoom = .2
        axins.plot(selected_day, self.cumsum_cost[selected_day], 'o')
        axins.plot(y_target[0:len(self.cumsum_cost)], 'r', linewidth=3, label='Target cost')
        axins.plot(self.cumsum_cost, 'b', linewidth=3, label='Accumulated cost')
        axins.plot(
            days,
            y_ref_daily_seleted,
            '--',
            c='b',
            linewidth=3,
            label=r'$y_{ref}$'
        )
        axins.plot(
            days,
            np.transpose(cost_daily_pred_selected_day_shift),
            alpha=.25,
            color='green',
            linewidth=0.5
        )
        axins.set_xlim(selected_day-0.25, 15)
        axins.set_ylim(2000, 4000)
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5")
        plt.draw()

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
            u_tilde: np.ndarray,
            u_star: np.ndarray,
            alpha_new_array: np.ndarray,
            beta_new_array: np.ndarray,
            alpha_mean_array: np.ndarray,
            beta_mean_array: np.ndarray
    ) -> None:
        """
        dCost/du=a, if cost is given by Cost=a*u+b
        """

        # Define the data for selected day
        past_costs = costs[selected_day]
        past_bids = bids[selected_day]
        u_tildes = u_tilde[selected_day]
        u_stars = u_star[selected_day]
        slopes = slope_array[selected_day]
        intercepts = intercept_array[selected_day]
        alphas = alpha_new_array[selected_day]
        betas = beta_new_array[selected_day]
        alpha_mean = alpha_mean_array[selected_day]
        beta_mean = beta_mean_array[selected_day]

        # Use only the costs from one ad slots
        past_costs_one_adslot = past_costs[0, :]
        past_bids_one_adslot = past_bids[0, :]
        u_tilde_one_adslot = u_tildes[0]
        u_star_one_adslot = u_stars[0]
        slopes_one_adslot = slopes[:, 0]
        intercepts_one_adslot = intercepts[:, 0]
        alpha_one_adslot = alphas[0]
        beta_one_adslot = betas[0]
        alpha_mean_one_adslot = alpha_mean[0]
        beta_mean_one_adslot = beta_mean[0]

        # Plot a subset of sampled regression lines
        for _ in range(20):
            alpha_p = np.random.choice(alpha_one_adslot)
            beta_p = np.random.choice(beta_one_adslot)
            plt.plot(
                u_tilde_one_adslot + u_star_one_adslot,
                alpha_p * u_tilde_one_adslot + beta_p,
                color='green',
                alpha=0.3
            )

        plt.scatter(
            u_tilde_one_adslot+u_star_one_adslot,
            past_costs_one_adslot,
            #s=np.linspace(100, 5, n_days_cost),
            edgecolors='k'
        )

        # Plot mean regression line
        plt.plot(
            u_tilde_one_adslot+u_star_one_adslot,
            alpha_mean_one_adslot * u_tilde_one_adslot + beta_mean_one_adslot
        )

        plt.xlabel('Bids')
        plt.ylabel('Cost')
        plt.show()

        # create associated trace plot
        self.plot_trace(alpha_one_adslot)
        #print(alpha_one_adslot.shape())

        return None

    def plot_trace(self, param, param_name='cost gradients'):
        """
        Plot the trace and posterior of a parameter.
        """

        # Summary statistics
        mean = np.mean(param)
        median = np.median(param)
        cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)

        # Plotting
        plt.subplot(2, 1, 1)
        plt.plot(param)
        plt.xlabel('samples')
        plt.ylabel(param_name)
        plt.axhline(mean, color='r', lw=2, linestyle='--')
        plt.axhline(median, color='c', lw=2, linestyle='--')
        plt.axhline(cred_min, linestyle=':', color='k', alpha=0.2)
        plt.axhline(cred_max, linestyle=':', color='k', alpha=0.2)
        #plt.title('Trace and Posterior Distribution for {}'.format(param_name))

        plt.subplot(2, 1, 2)
        plt.hist(param, 30, density=True);
        sns.kdeplot(param, shade=True)
        plt.xlabel(param_name)
        plt.ylabel('density')
        plt.axvline(mean, color='r', lw=2, linestyle='--', label='mean')
        plt.axvline(median, color='c', lw=2, linestyle='--', label='median')
        plt.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
        plt.axvline(cred_max, linestyle=':', color='k', alpha=0.2)

        plt.gcf().tight_layout()
        plt.legend()
        plt.show()

        return None

mean_obj = [3619.64744255879,
7488.117187778079,
11251.132718620216,
26478.42921496186,
18462.8047206992,
22016.548385091213
            ]

var_obj = [89124.47292671933,
83765.70587338542,
62033.327655908586,
238457.924364393,
57899.251160905376,
272563.5084949905
]

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

mean_obj = np.abs(mean_obj)

fig, ax = plt.subplots()
ax.scatter(mean_obj, var_obj)

for i, txt in enumerate(alphas):
    ax.annotate(txt, (mean_obj[i], var_obj[i]))
plt.show()