import matplotlib
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt

import numpy as np
import yaml
import os
import math
from scipy import stats
import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.patches as mpatches

problem_name = "foodrisk"
# problem_name = "Clithero"
# problem_name = "Krajbich"


# algName_2_latex = {
#     "Wagenmakers07Eq5_trans": r"$(\lambda_{\text{trans}},\widehat{\theta}_{\text{CH,logit}})$",
#     "GLM_weakPref": r"$(\lambda_{\text{weak}},\widehat{\theta}_{\text{CH}})$",
#     "GLM_trans": r"$(\lambda_{\text{trans}},\widehat{\theta}_{\text{CH}})$",
#     "LM_trans": r"$(\lambda_{\text{trans}},\widehat{\theta}_{\text{CH,DT}})$",
#     "LM_trans_noSubt": r"$(\lambda_{\text{trans}},\widehat{\theta}_{\text{CH,}\mathbb{RT}})$",
#     "Chiong24Lemma1_trans": r"$(\lambda_{\text{trans}},\widehat{\theta}_{\text{CH,DT,logit}})$",
# }

##Here's exactly what to replace in your algName_2_latex dictionary:
algName_2_latex = {
    "Wagenmakers07Eq5_trans": r"$(\lambda_{trans},\hat{\theta}_{CH,logit})$",
    "GLM_weakPref": r"$(\lambda_{weak},\hat{\theta}_{CH})$",
    "GLM_trans": r"$(\lambda_{trans},\hat{\theta}_{CH})$",
    "LM_trans": r"$(\lambda_{trans},\hat{\theta}_{CH,DT})$",
    "LM_trans_noSubt": r"$(\lambda_{trans},\hat{\theta}_{CH,RT})$",
    "Chiong24Lemma1_trans": r"$(\lambda_{trans},\hat{\theta}_{CH,DT,logit})$",
    "LMOrtho_trans": r"$(\lambda_{trans},\hat{\theta}_{CH,DT,ortho})$"
}

algName_color_hatches = [
    ("Wagenmakers07Eq5_trans", '#f0f0f0', ''),
    ("GLM_weakPref", '#bdbdbd', ''),
    ("GLM_trans", "#636363", ''),
    ("LM_trans", '#e6550d', ''),
    ("LM_trans_noSubt", "#fdae6b", ''),
    ("Chiong24Lemma1_trans", "#fee6ce", ''),
    ("LMOrtho_trans", "#fd8d3c", ''),              # Orange between the two LM variants
]

problemName_2_budgets = {
    "Clithero": [100, 300],
    "Krajbich": [200, 300],
    "foodrisk": [500, 1000],
}


def main(plot_all_budgets):
    global problem_name

    file_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(file_path)
    result_filepath = os.path.join(
        dir_path, "run_"+problem_name, "processed_result_parallel.yaml")
    plot_path = os.path.join(dir_path, "run_"+problem_name)

    box_plot_width = 0.4
    violin_bar_width = 0.8
    num_y_ticks = 6
    figure_size = (5, 5)
    if plot_all_budgets:
        figure_size = (20, 5)

    # Enable LaTeX rendering in Matplotlib
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    # # https://stackoverflow.com/questions/23824687/text-does-not-work-in-a-matplotlib-label
    # # https://stackoverflow.com/questions/65426069/use-of-mathbb-in-matplotlib
    # # for \text and \mathbb command
    # mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amsfonts}'

    algName_2_eta = {}  # eta is uniquely determined by problem_name
    budgets = []
    algName_budget_2_data = {}
    with open(result_filepath, 'r') as file:
        data = yaml.safe_load(file)
        for (alg_name, c, h) in algName_color_hatches:
            assert len(data[alg_name].keys()) == 1
            eta = list(data[alg_name].keys())[0]
            if alg_name not in algName_2_eta:
                algName_2_eta[alg_name] = eta
            else:
                assert algName_2_eta[alg_name] == eta

            if plot_all_budgets:
                for budget in data[alg_name][eta].keys():
                    budgets.append(budget)
                    assert (alg_name, budget) not in algName_budget_2_data
                    algName_budget_2_data[(alg_name, budget)
                                          ] = data[alg_name][eta][budget]
            else:
                for budget in problemName_2_budgets[problem_name]:
                    budgets.append(budget)
                    assert (alg_name, budget) not in algName_budget_2_data
                    algName_budget_2_data[(alg_name, budget)
                                          ] = data[alg_name][eta][budget]

    print("algName_2_eta=", algName_2_eta)
    budgets = sorted(list(set(budgets)))
    groups = budgets
    categories = [x[0] for x in algName_color_hatches]
    print("groups=", groups)
    print("categories=", categories)
    category_colors = [x[1]
                       for x in algName_color_hatches]  # parallel to groups
    # hatches = [x[2] for x in algName_color_hatches]  # parallel to groups
    ##group_labels = [r'${:.0f}$'.format(x) for x in groups]
    group_labels = ['{:.0f}'.format(x) for x in groups]
    category_labels = [algName_2_latex[x] for x in categories]

    data = []
    for budget in budgets:
        for (alg_name, c, h) in algName_color_hatches:
            result = algName_budget_2_data[(alg_name, budget)]
            mistake_at_budget_means = []
            for (subject_idx, v) in result.items():
                mistake_at_budget_mean = v['mistake_at_budget_mean']
                mistake_at_budget_means.append(mistake_at_budget_mean)
            data.append(mistake_at_budget_means)

    num_groups = len(groups)
    num_categories = len(categories)
    print("num_groups=", num_groups)  # 2 groups
    print("num_categories=", num_categories)  # 2 groups
    assert num_groups*num_categories == len(data)

    # Generate positions dynamically
    positions = []
    for g in range(num_groups):
        start = g * (num_categories + 1) + 1
        positions.extend(range(start, start + num_categories))
    print("positions=", positions)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figure_size)
    ax.grid(axis='y', color='gray', linestyle='-', linewidth=0.5)
    # https://stackoverflow.com/questions/1726391/matplotlib-draw-grid-lines-behind-other-graph-elements
    ax.set_axisbelow(True)

    # Create the violin plot
    violin_parts = ax.violinplot(
        data, positions=positions, showmeans=False, showmedians=False, showextrema=False, widths=violin_bar_width)

    # Color the violins according to the categories and set border
    for i, vp in enumerate(violin_parts['bodies']):
        vp.set_facecolor(category_colors[i % num_categories])
        vp.set_edgecolor('black')  # Border color for the violin plot
        vp.set_linewidth(1)      # Border width for the violin plot
        vp.set_alpha(1)

    # Create the boxplot with customizations
    box = ax.boxplot(data, positions=positions,
                     patch_artist=True, widths=box_plot_width,
                     flierprops=dict(
                         marker='o', color='black', markersize=5),
                     )

    # Set boxplot box color, border color, and border width
    for i in range(len(box['boxes'])):
        box['boxes'][i].set_facecolor('none')   # Box color
        box['boxes'][i].set_edgecolor('black')  # Border color
        # box['boxes'][i].set_linewidth(2)        # Border width
        box['medians'][i].set_color('black')    # Median line color
        box['boxes'][i].set_alpha(1)  # Optional: Set alpha for transparency

    # Create custom legend handles
    # https://stackoverflow.com/questions/4534480/get-legend-as-a-separate-picture-in-matplotlib
    custom_handles = []

    def f(label, color): return mpatches.Patch(
        facecolor=color, edgecolor="black", linewidth=2, label=label)
    for i, (alg_name, color, _) in enumerate(algName_color_hatches):
        custom_handles.append(f(category_labels[i], color))
    legend = ax.legend(custom_handles, category_labels,
                       loc='upper right', fontsize=20, frameon=False,
                       handleheight=2.0, handlelength=2.0)

    # Set x-tick labels and positions
    xticks = []
    for g in range(num_groups):
        tmp1 = positions[int(g * num_categories)]
        tmp2 = positions[int((g+1) * num_categories)-1]
        xticks.append((tmp1+tmp2)/2)
    print("xticks=", xticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels(group_labels)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    ax.set_ylim([-0.05, 1.05])  # below 0 for outliers

    y_ticks = np.linspace(0, 1, num_y_ticks)
    ax.set_yticks(y_ticks)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel("Budget (sec)", fontsize=20)
    # ax.set_ylabel(
    #     r"$\text{Error probability }\mathbb{P}\left[\widehat{z}\neq z^*\right]$",
    #     fontsize=20)
    ax.set_ylabel(
        r"Error probability P[$\hat{z} \neq z^*$]",
    fontsize=20)

    path = os.path.join(plot_path, problem_name + '_' +
                        str(len(budgets))+'_legend.pdf')
    plt.savefig(path, format='pdf', bbox_inches='tight')

    legend.remove()
    path = os.path.join(plot_path, problem_name + '_' +
                        str(len(budgets))+'_NoLegend.pdf')
    plt.savefig(path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    # Create a new figure just for the legend
    if plot_all_budgets:
        ncol = len(algName_color_hatches)
    else:
        ncol = 2
    fig_legend, ax_legend = plt.subplots(figsize=figure_size)
    fig_legend.legend(custom_handles, category_labels, loc='center',
                      fontsize=30, frameon=False, ncol=ncol,
                      handleheight=2.0, handlelength=2.0)
    ax_legend.axis('off')  # Hide the axis
    path = os.path.join(plot_path, problem_name + '_legend'+str(ncol)+'.pdf')
    plt.savefig(path, format='pdf', bbox_inches='tight')
    plt.close(fig_legend)


if __name__ == "__main__":
    main(plot_all_budgets=False)
    main(plot_all_budgets=True)
