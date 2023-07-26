import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

_RCPARAMS_LATEX_SINGLE_COLUMN = {
    'text.usetex': True,

    'axes.labelsize': 14,
    'legend.fancybox': True,  # Rounded legend box
    'legend.framealpha': 0.8,
}

# This is the right width (in inches) for a 'letter' page LaTeX document that imports the geometry package with default parameters.
_PAGE_WIDTH_INCHES = 6.775
_GOLDEN_RATIO = (5 ** 0.5 - 1) / 2
RCPARAMS_LATEX_DOUBLE_COLUMN = {**_RCPARAMS_LATEX_SINGLE_COLUMN,
                                }  # 'figure.figsize': (_PAGE_WIDTH_INCHES / 2, _GOLDEN_RATIO * _PAGE_WIDTH_INCHES / 2)}


def figure_1(): #the data here is from experiments and not from synthetic trajectories
    df = pd.read_csv("./Data/experimental_traj_df.csv")
    with matplotlib.rc_context(rc=RCPARAMS_LATEX_DOUBLE_COLUMN):
        fig, ax = plt.subplots(frameon=False, figsize=(6, 6))
        image = plt.imread("./Data/experimental_snapshot.png")
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_xlim([7300, 9300])
        ax.set_ylim([2600, 1100])

        sf = 8750.5 / 113.983  # scale factor
        height_5um = 1300
        ax.plot(8820 + np.array([0, 1]) * sf * 5, [height_5um, height_5um], 'w')
        ax.plot(8820 + np.array([0, 0]) * sf * 5, [height_5um - 20, height_5um + 20], 'w')
        ax.plot(8820 + np.array([1, 1]) * sf * 5, [height_5um - 20, height_5um + 20], 'w')
        ax.text(8820 + 0.17 * sf * 5, height_5um - 45, r"$5\mu \rm{m}$", color="w", fontsize=20)

        colors = sns.color_palette("colorblind")[3:0:-1]
        cmap = ListedColormap(colors)
        bounds = [0, 68, 294, 320]
        norm = BoundaryNorm(bounds, cmap.N)

        ind1 = 684
        ind2 = 2943

        xx = df.x.values * sf
        yy = df.y.values * sf
        xx = np.hstack([xx, [8773, 8791]])  # add finishing touches to make the trajectory end on the particle
        yy = np.hstack([yy, [1535, 1518]])
        ax.plot(xx[100:ind1], yy[100:ind1], '-', color=colors[0], linewidth=2, zorder=100)
        ax.plot(xx[ind1:ind2], yy[ind1:ind2], '-', color=colors[1], linewidth=2, zorder=102)
        ax.plot(xx[ind2:], yy[ind2:], '-', color=colors[2], linewidth=2, zorder=101)
        # Hack to get a Mappable for the colorbar
        sc = ax.scatter(np.linspace(0, 1), np.linspace(0, 1), c=np.linspace(0, 1), cmap=cmap, norm=norm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        fig.colorbar(sc, orientation="horizontal", spacing="proportional", cax=cax)
        cax.set_xlabel(r'$$\rm{Time\,(sec)}$$', fontsize=14)
        cax.xaxis.set_label_coords(.5, -.75)

        fig.savefig("./Figures/experimental_traj.pdf", bbox_inches='tight', pad_inches=0)


def figure_2():
    df = pd.read_csv("./Data/example_traj_df.csv")
    with matplotlib.rc_context(rc=RCPARAMS_LATEX_DOUBLE_COLUMN):
        x = df.x.values
        y = df.y.values
        s = df.state.values
        free = np.vstack([x, y])
        stuck = np.vstack([x, y])
        for i in range(1, len(s)):
            if s[i] != s[i - 1]:
                pass
            else:
                if s[i - 1] == 0:
                    stuck[:, i] = np.nan
                else:
                    free[:, i] = np.nan

        fig, ax = plt.subplots()
        p1 = ax.plot(free[0], free[1], '-', label=r"$$\rm{Free}$$", color=sns.color_palette("colorblind")[0], linewidth=2)
        p2 = ax.plot(stuck[0], stuck[1], '-', label=r"$$\rm{Tethered}$$", color=sns.color_palette("colorblind")[3], linewidth=2)
        ax.set_xlim([-18, 52])
        ax.set_ylim([-58, 12])
        ax.set_aspect("equal")
        ax.legend(fontsize=15)

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        fig.tight_layout()

        fig.savefig("./Figures/Example_traj.pdf", bbox_inches='tight')


def figure_3():
    dt_list = [10., 1., 0.5, 10., 10., 10., 10.]
    tau_list = [100., 100., 100., 50., 20., 50., 50.]
    text_list = [r"$$1$$", r"$$2$$", r"$$3$$", r"$$4,6,7$$", r"$$5$$", r"$$6$$", r"$$7$$"]
    with matplotlib.rc_context(rc=RCPARAMS_LATEX_DOUBLE_COLUMN):
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.set_xscale("log")
        ax.set_yscale("log")
        p1 = ax.plot([.001, 200], [1, 1], "k:", linewidth=1)[0]
        p2 = ax.plot([.001, 200], [.001, 200], "k-", linewidth=1)[0]
        p3 = ax.fill_between([.001, 200], [.001, 200], [.001, .001], color="yellow", alpha=0.1)
        p4 = ax.fill_between([.001, 200], [1, 1], [.001, .001], color="red", alpha=0.1)

        for i in range(5):
            ax.plot(tau_list[i], dt_list[i], '.', markersize=8, color="k", label=f"{i + 1}")
            if i != 3:
                ax.annotate(text_list[i], (tau_list[i], dt_list[i] * 1.1), size=10)
            else:
                ax.annotate(text_list[i], (tau_list[i] / 1.275, dt_list[i] * 1.1), size=10)
        ax.set_xlim([0.2, 150])
        ax.set_ylim([0.2, 150])
        ax.set_xlabel(r"$${\rm min}{(\tau_0,\tau_1)}$$")
        ax.set_ylabel(r"$$\Delta t$$", rotation=0)

        ax.text(.5, 30, r"$$\rm{Multiple\,transitions\,in}\,\Delta t$$", size=13.5)
        ax.text(5, 0.4, r"$$\rm{Oversampling}$$", size=13.5)

        fig.tight_layout()
        fig.savefig("./Figures/Times_triangle.pdf", bbox_inches='tight')


def figure_5():
    params_list_arr = np.load("./Data/indifference_to_init_conditions_params_list_arr_regime_0.npy", allow_pickle=True)
    converged_arr = np.load("./Data/indifference_to_init_conditions_converged_arr_regime_0.npy", allow_pickle=True)
    df = pd.read_csv("./Data/indifference_to_init_conditions_df_regime_0.csv")
    X_arr = df[["x", "y"]].values
    N_init_conditions = len(params_list_arr)
    T_stick, T_unstick, D, A = 100., 100., 1., 1.
    params_arr = np.array([T_stick, T_unstick, D, A])
    dt, T = 10., 10000.

    converged_arr = converged_arr.astype(bool)
    max_iters = np.max([len(params_list) for params_list in params_list_arr])
    theta_mat = np.zeros([N_init_conditions, max_iters, 4])
    for i in range(N_init_conditions):
        params_list = params_list_arr[i]  # nx4
        last_params = params_list[-1]
        rows_to_fill = max_iters - len(params_list)
        if rows_to_fill > 0:
            theta_mat[i] = np.vstack(
                [params_list, np.array([last_params for _ in range(max_iters - len(params_list))])])
        else:
            theta_mat[i] = params_list

    optimal_params = np.load("./Data/indifference_to_init_conditions_optimal_params_regime_0.npy")

    params_text = [r'${\hat{\tau}_0}/{\tau_0}$',
                   r'${\hat{\tau}_1}/{\tau_1}$',
                   r'${\hat{D}}/{D}$',
                   r'${\hat{A}}/{A}$']
    with matplotlib.rc_context(rc=RCPARAMS_LATEX_DOUBLE_COLUMN):
        fig, axes = plt.subplots(1, 4, figsize=(16, 3))
        for n in range(4):
            ax = axes[n]

            for iter in theta_mat[converged_arr, :, n]:
                ax.plot(iter / params_arr[n], "b", alpha=0.05)

            ax.plot([0, max_iters - 1], np.array([1., 1.]) * (optimal_params[n] / params_arr[n]), 'r--', linewidth=1)
            ax.set_title(params_text[n], fontsize=15)
            ax.set_yscale("log")
            ax.set_ylim([1 / 50, 50])
            ax.grid(True)
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.set_xticks(np.arange(0, max_iters, step=2, dtype=int))
            ax.set_xlabel(r"$$\rm{Iterations}$$", size=16)
            if n > 0:
                ax.yaxis.set_ticklabels([])
                ax.yaxis.set_ticks_position('none')

        fig.tight_layout()
        fig.savefig("./Figures/indifference_to_init_conditions.pdf", bbox_inches='tight')


def figure_6():
    df = pd.read_csv("./Data/test_accuracy_df_full.csv")
    with matplotlib.rc_context(rc=RCPARAMS_LATEX_DOUBLE_COLUMN):
        fig, axes = plt.subplots(1, 4, figsize=(16, 3))
        ax = axes[0]
        plt.axes(ax)
        sns.kdeplot(df.groupby("regime").get_group(0)["T0"], clip=[0, 1000], label=r"$$\Delta t = 10$$")
        sns.kdeplot(df.groupby("regime").get_group(1)["T0"], clip=[0, 1000], label=r"$$\Delta t = 1$$")
        sns.kdeplot(df.groupby("regime").get_group(2)["T0"], clip=[0, 1000], label=r"$$\Delta t = 0.5$$")
        ylim = ax.get_ylim()
        ax.plot([100, 100], 2 * np.array(ylim), "k--", linewidth=1.5)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$\tau_0$", fontsize=16)
        ax.set_xlim([0, 220])
        ax.get_yaxis().set_visible(False)
        ax.tick_params(axis='x', which='major', labelsize=14)
        leg = ax.legend(loc="upper left", fontsize=15)

        ax = axes[1]
        plt.axes(ax)
        sns.kdeplot(df.groupby("regime").get_group(0)["T1"], clip=[0, 1000], label=1)
        sns.kdeplot(df.groupby("regime").get_group(1)["T1"], clip=[0, 1000], label=2)
        sns.kdeplot(df.groupby("regime").get_group(2)["T1"], clip=[0, 1000], label=3)
        ylim = ax.get_ylim()
        ax.plot([100, 100], 2 * np.array(ylim), "k--", linewidth=1.5)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$\tau_1$", fontsize=16)
        ax.set_xlim([0, 220])
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel(r"$\tau_1$")
        ax.tick_params(axis='x', which='major', labelsize=14)

        ax = axes[2]
        plt.axes(ax)
        sns.kdeplot(df.groupby("regime").get_group(0)["D"], clip=[0, 1000], label=1)
        sns.kdeplot(df.groupby("regime").get_group(1)["D"], clip=[0, 1000], label=2)
        sns.kdeplot(df.groupby("regime").get_group(2)["D"], clip=[0, 1000], label=3)
        ylim = ax.get_ylim()
        ax.plot([1, 1], 2 * np.array(ylim), "k--", linewidth=1.5)
        ax.set_xlim([0.75, 1.25])
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$D$", fontsize=16)
        ax.get_yaxis().set_visible(False)
        ax.tick_params(axis='x', which='major', labelsize=14)

        ax = axes[3]
        plt.axes(ax)
        sns.kdeplot(df.groupby("regime").get_group(0)["A"], clip=[0, 1000], label=1)
        sns.kdeplot(df.groupby("regime").get_group(1)["A"], clip=[0, 1000], label=2)
        sns.kdeplot(df.groupby("regime").get_group(2)["A"], clip=[0, 1000], label=3)
        ax.set_xlim([0.75, 1.25])
        ylim = ax.get_ylim()
        ax.plot([1, 1], 2 * np.array(ylim), "k--", linewidth=1.5)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$A$", fontsize=16)
        ax.get_yaxis().set_visible(False)
        ax.tick_params(axis='x', which='major', labelsize=14)

        fig.tight_layout()
        fig.savefig("./Figures/regimes_1_to_3.pdf", bbox_inches='tight')


def figure_7():
    params_by_iter, params_std_by_iter, max_L_by_iter, softmax_by_iter, S_list, X_tether_list, convergence_flag = np.load(
        "./Data/K_most_likely_alg.npy", allow_pickle=True)
    S_list_alg = S_list
    params_by_iter, params_std_by_iter, max_L_by_iter, softmax_by_iter, S_list, X_tether_list, convergence_flag = np.load(
        "./Data/K_most_likely_oracle.npy", allow_pickle=True)
    S_list_oracle = S_list
    true_df = pd.read_csv("./Data/k_most_likely_orig_df.csv")
    colors = [sns.color_palette("colorblind")[0], sns.color_palette("colorblind")[3]]
    cmap_name = 'my_list'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)

    N = len(true_df)
    ylim = 10
    with matplotlib.rc_context(rc=RCPARAMS_LATEX_DOUBLE_COLUMN):
        fig, ax = plt.subplots(3, gridspec_kw={'height_ratios': [1, 10, 10]})
        ax[0].imshow(true_df.state.values.reshape((1, N)), cmap=cmap, interpolation="none", aspect="auto")
        ax[0].grid(True, axis="x")
        ax[0].xaxis.set_ticklabels([])
        ax[0].xaxis.set_ticks_position('none')
        ax[0].set_yticks([])
        ax[0].set_ylabel("True")

        for j in range(1, 3):
            ax[j].imshow([S_list_oracle, S_list_alg][j - 1], cmap=cmap, interpolation="none", aspect="auto")
            ax[j].set_yticks(np.arange(ylim) + 1)
            ax[j].grid(True, axis="x")
            for i in range(ylim):
                ax[j].plot([0, N - 1], [i + 0.5, i + 0.5], 'k', linewidth=1)

            ax[j].set_ylim([-0.5, ylim - .5])
            ax[j].invert_yaxis()
            ax[j].yaxis.set_ticklabels([])
            ax[j].yaxis.set_ticks_position('none')

            for y in [1, 4, 7, 10]:
                if y < 10:
                    ax[j].text(-6, y - 0.75, str(y))
                else:
                    ax[j].text(-9.5, y - 0.75, str(y))
            if j == 1:
                ylabel_string = r"$$\rm{Oracle\,paramaeters}$$"
            else:
                ylabel_string = r"$$\rm{Estimated\,paramaeters}$$"
            ax[j].set_ylabel(ylabel_string, labelpad=10)
        ax[1].xaxis.set_ticklabels([])
        ax[1].xaxis.set_ticks_position('none')
        ax[2].set_xlabel(r"$$\rm{Time}$$")
        fig.supylabel(r"$$\rm{Likelihood\,ranking}$$", fontsize=16)

        legend_elements = [Patch(facecolor=colors[0], edgecolor='k',
                                 label=r'$$\rm{Free}$$'),
                           Patch(facecolor=colors[1], edgecolor='k',
                                 label=r'$$\rm{Tethered}$$')
                           ]

        # Create the figure
        leg = ax[1].legend(handles=legend_elements, loc="upper left", fontsize=15)
        fig.tight_layout()
        fig.savefig("./Figures/K_most_likely.pdf", bbox_inches='tight')
