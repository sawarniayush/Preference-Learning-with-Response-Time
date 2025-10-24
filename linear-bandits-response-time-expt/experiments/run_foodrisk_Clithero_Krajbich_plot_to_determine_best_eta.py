import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
import math
from scipy import stats
import matplotlib.ticker as ticker
import matplotlib as mpl

problem_name = "Clithero"
# problem_name = "Krajbich"

algName_2_latex = {
    "Wagenmakers07Eq5_trans": r"$(\lambda_{\text{trans}},\widehat{\theta}_{\text{CH,logit}})$",
    "GLM_weakPref": r"$(\lambda_{\text{weak}},\widehat{\theta}_{\text{CH}})$",
    "GLM_trans": r"$(\lambda_{\text{trans}},\widehat{\theta}_{\text{CH}})$",
    "LM_trans": r"$(\lambda_{\text{trans}},\widehat{\theta}_{\text{CH,DT}})$",
    "LM_trans_noSubt": r"$(\lambda_{\text{trans}},\widehat{\theta}_{\text{CH,}\mathbb{RT}})$",
    "Chiong24Lemma1_trans": r"$(\lambda_{\text{trans}},\widehat{\theta}_{\text{CH,DT,logit}})$",
}

algName_color_hatches = [
    ("Wagenmakers07Eq5_trans", '#f0f0f0', ''),
    ("GLM_weakPref", '#bdbdbd', ''),
    ("GLM_trans", "#636363", ''),
    ("LM_trans", '#e6550d', ''),
    ("LM_trans_noSubt", "#fdae6b", ''),
    ("Chiong24Lemma1_trans", "#fee6ce", ''),
]


def main():
    file_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(file_path)
    result_filepath = os.path.join(
        dir_path, "run_"+problem_name+"_tune_eta", "alg_name_2_eta_2_mistakeAtBudgetMeans.yaml")
    plot_path = os.path.join(dir_path, "run_"+problem_name+"_tune_eta")

    figure_size = (10, 6)
    box_plot_width = 0.3
    violin_bar_width = 0.9
    num_y_ticks = 6

    # Enable LaTeX rendering in Matplotlib
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # https://stackoverflow.com/questions/23824687/text-does-not-work-in-a-matplotlib-label
    # https://stackoverflow.com/questions/65426069/use-of-mathbb-in-matplotlib
    # for \text and \mathbb command
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amsfonts}'

    alg_name_2_eta_2_mistakeAtBudgetMeans = None
    etas = None
    with open(result_filepath, 'r') as file:
        alg_name_2_eta_2_mistakeAtBudgetMeans = yaml.safe_load(file)
        for (alg_name, c, h) in algName_color_hatches:
            tmp = alg_name_2_eta_2_mistakeAtBudgetMeans[alg_name].keys()
            etas_ = sorted([int(x) for x in tmp])
            if etas is None:
                etas = etas_
            else:
                assert etas == etas_
    print("etas=", etas)

    for (alg_name, color, h) in algName_color_hatches:
        group_labels = [r'${:.0f}$'.format(value) for value in etas]
        category_labels = [alg_name]
        category_colors = [color]

        data = []
        for eta in etas:
            tmp = alg_name_2_eta_2_mistakeAtBudgetMeans[alg_name][eta]
            # print(len(tmp))
            data.append(tmp)

        num_groups = len(group_labels)
        num_categories = len(category_labels)
        print("num_groups=", num_groups)  # 2 groups
        print("num_categories=", num_categories)  # 2 groups
        assert num_groups*num_categories == len(data)

        # Generate positions dynamically
        positions = []
        for g in range(num_groups):
            start = g * num_categories + 1
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
            # Optional: Set alpha for transparency
            box['boxes'][i].set_alpha(1)

        # Set x-tick labels and positions
        xticks = []
        for g in range(num_groups):
            tmp1 = positions[int(g * num_categories)]
            tmp2 = positions[int((g+1) * num_categories)-1]
            xticks.append((tmp1+tmp2)/2)
        print("xticks=", xticks)
        ax.set_xticks(xticks)
        ax.set_xticklabels(group_labels)

        # ax.set_ylim(bottom=0)  # Set y-axis to start from 0 (optional)
        ax.set_ylim([-0.05, 1.05])  # below 0 for outliers

        # y_min, y_max = min(np.min(d)
        #                    for d in data), max(np.max(d) for d in data)
        # y_ticks = np.linspace(y_min, y_max, num_y_ticks)
        y_ticks = np.linspace(0, 1, num_y_ticks)
        ax.set_yticks(y_ticks)

        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlabel(r'$\mathrm{Elimination\ parameter\ }\eta$', fontsize=25)
        plt.ylabel(
            r"$\text{Error probability }\mathbb{P}\left[\widehat{z}\neq z^*\right]$",
            fontsize=25)

        # Show the plot
        path = os.path.join(plot_path, problem_name +
                            '_eta_' + alg_name + '.pdf')
        plt.savefig(path, format='pdf', bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    main()
