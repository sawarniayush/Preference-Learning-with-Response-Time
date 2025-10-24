import yaml
import copy
from ast import literal_eval as make_tuple
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl


file_path = os.path.realpath(__file__)
dir_path = os.path.dirname(file_path)
dir_path = os.path.join(dir_path, "run_SSM_estimation")
path_data = os.path.join(dir_path, "run_SSM_estimation.yaml")
ylims = [(0, 1), (0, 1), (0, 1), (-1, 1), (-1, 1)]
cmap_name = "plasma"

figure_size = (10, 10)
figure_size_legend = (0.5, 10)


def main():
    # Enable LaTeX rendering in Matplotlib
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # https://stackoverflow.com/questions/23824687/text-does-not-work-in-a-matplotlib-label
    # https://stackoverflow.com/questions/65426069/use-of-mathbb-in-matplotlib
    # for \text and \mathbb command
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amsfonts}'

    with open(path_data, 'r') as file:
        data = yaml.safe_load(file)
    methods = [make_tuple(x) for x in data["methods"]]
    budgets = [float(x) for x in data["budgets"]]
    DDM_barrier_from_0s = [float(x) for x in data["DDM_barrier_from_0s"]]
    scale_zs = [float(x) for x in data["scale_zs"]]
    print("methods=", methods)
    print("budgets=", budgets)
    print("DDM_barrier_from_0s=", DDM_barrier_from_0s)
    print("scale_zs=", scale_zs)

    assert len(budgets) == 1
    budget = budgets[0]

    results_ = data["DDMBarrierFrom0_scalez_budget_method_2_error"]
    results = {}
    for (k, v) in results_.items():
        results[make_tuple(k)] = v

    data_plots = []
    for method in methods:
        print("\nmethod=", method)
        data_plot = np.zeros((len(scale_zs), len(DDM_barrier_from_0s)))  # y,x
        for (scale_z_idx, scale_z) in enumerate(scale_zs):
            for (DDM_barrier_from_0_idx, DDM_barrier_from_0) in enumerate(DDM_barrier_from_0s):
                key = (DDM_barrier_from_0, scale_z, budget, method)
                error_elim = make_tuple(results[key]["error_elim"])
                error_elim_mean = error_elim[0]
                error_elim_std = error_elim[1]
                error_elim_count = error_elim[2]
                error_elim_stderr = error_elim[3]
                assert abs(error_elim_stderr-error_elim_std /
                           (error_elim_count**0.5)) <= 1e-10

                data_plot[scale_z_idx,
                          DDM_barrier_from_0_idx] = error_elim_mean
        data_plots.append(copy.deepcopy(data_plot))

    legends = [False, False, False]

    for (method_idx, method) in enumerate(methods):
        data_plot = data_plots[method_idx]
        print("\n", method, ": ", np.min(data_plot), np.max(data_plot))
        assert ylims[method_idx][0] <= np.min(data_plot)
        assert ylims[method_idx][1] >= np.max(data_plot)

        X, Y = np.meshgrid(range(len(DDM_barrier_from_0s)),
                           range(len(scale_zs)))  # x,y

        fig, ax = plt.subplots(figsize=figure_size)
        cax = ax.pcolormesh(X, Y, data_plot, cmap=cmap_name, shading='auto',
                            vmin=ylims[method_idx][0], vmax=ylims[method_idx][1])

        # https://stackoverflow.com/questions/27092991/white-lines-in-matplotlibs-pcolor
        cax.set_edgecolor('face')
        if legends[method_idx]:
            fig.colorbar(cax)

        # Set ticks at the positions corresponding to the selected indices
        x_tick_indices = []
        for (i, v) in enumerate(DDM_barrier_from_0s):
            if i % 2 == 0:  # every Nth tick
                x_tick_indices.append(i)
        print("x_tick_indices=", x_tick_indices)
        ax.set_xticks(x_tick_indices)
        ax.set_xticklabels(
            [f"{DDM_barrier_from_0s[i]:.1f}" for i in x_tick_indices], fontsize=35, rotation=-90)

        y_tick_indices = []
        y_tick_labels = []
        for (i, v) in enumerate(scale_zs):
            if v < 1:
                if v in [0.1, 0.4, 0.7]:
                    y_tick_indices.append(i)
                    y_tick_labels.append(f"{v:.1f}")
            elif v <= 101:
                if i % 2 == 1:  # every Nth tick
                    y_tick_indices.append(i)
                    y_tick_labels.append(f"{v:.0f}")
            else:
                if i % 2 == 1:  # every Nth tick
                    y_tick_indices.append(i)
                    y_tick_labels.append(f"{v:.0f}")

        print("y_tick_indices=", y_tick_indices)
        print("y_tick_labels=", [f"{scale_zs[i]:.1f}" for i in y_tick_indices])
        ax.set_yticks(y_tick_indices)
        ax.set_yticklabels(y_tick_labels, fontsize=35)

        ax.invert_yaxis()

        ax.set_xlabel(r"$\text{Barrier } a$", fontsize=45)
        ax.xaxis.set_label_coords(0.5, -0.12)
        ax.set_ylabel(
            r"$\text{Arm scaling factor } c_{\mathcal{Z}}$",
            fontsize=45)
        ax.yaxis.set_label_coords(-0.14, 0.5)

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        path = os.path.join(dir_path, "run_SSM_estimation_" +
                            '_'.join(method)+'.pdf')
        plt.savefig(path, format='pdf', bbox_inches='tight')
        plt.close(fig)

    # Save only the colorbar
    # https://stackoverflow.com/questions/16595138/standalone-colorbar
    a = np.array([[0, 1]])
    fig = plt.figure(figsize=figure_size_legend)
    img = plt.imshow(a, cmap=cmap_name)
    plt.gca().set_visible(False)

    # Create a separate colorbar with custom positioning
    cax = plt.axes([0.1, 0.2, 0.8, 0.6])
    cbar = plt.colorbar(img, orientation="vertical",
                        cax=cax)
    plt.tick_params(labelsize=30)

    # Remove the frame (axis lines) from the colorbar, but keep ticks and labels
    cbar.outline.set_visible(False)  # Hide the colorbar outline

    colorbar_path = os.path.join(dir_path, "run_SSM_estimation_colorbar.pdf")
    plt.savefig(colorbar_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
