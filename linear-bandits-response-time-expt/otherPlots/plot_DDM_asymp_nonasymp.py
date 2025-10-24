import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.patches as mpatches


y_ticks_asymp = None
y_ticks_nonasymp = None
figure_size = (10, 6)
color_LM = "#ef8a62"
color_GLM = "#999999"
color_diff = "#000000"


def main():
    # Enable LaTeX rendering in Matplotlib
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # https://stackoverflow.com/questions/23824687/text-does-not-work-in-a-matplotlib-label
    # https://stackoverflow.com/questions/65426069/use-of-mathbb-in-matplotlib
    # for \text and \mathbb command
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amsfonts,amssymb}'

    as_ = [0.8, 1.8]
    assert len(as_) in [1, 2]

    vs = np.arange(-6, 6, 1e-3)

    a_2_terms_LM_nonAsym = {}
    a_2_terms_GLM_nonAsym = {}
    a_2_terms_LM_asym = {}
    a_2_terms_GLM_asym = {}

    a_2_terms_LM_nonAsym_ignoreConst = {}
    a_2_terms_GLM_nonAsym_ignoreConst = {}

    a_2_ERTs = {}
    a_2_VRTs = {}
    a_2_ECHs = {}
    a_2_VCHs = {}

    for a in as_:
        # Prepare arrays to store results
        ERTs = np.zeros(len(vs))
        ECHs = np.zeros(len(vs))
        VRTs = np.zeros(len(vs))
        VCHs = np.zeros(len(vs))
        ECHs_01 = np.zeros(len(vs))
        VCHs_01 = np.zeros(len(vs))

        for i, v in enumerate(vs):
            # Eq.6 in Wagenmakers, E. J., Van Der Maas, H. L., & Grasman, R. P. (2007). An EZ-diffusion model for response time and accuracy. Psychonomic bulletin & review, 14(1), 3-22. Note that our `a` here = `z=a/2` in that paper.
            if v != 0:
                tmp1 = -4 * a * v * np.exp(-2 * a * v) - np.exp(-4 * a * v) + 1
                tmp2 = (np.exp(-2 * a * v) + 1)**2
                VRTs[i] = a / v**3 * tmp1 / tmp2
            else:
                VRTs[i] = 2 * a**4 / 3

            # Eq.A.17 in Palmer, J., Huk, A. C., & Shadlen, M. N. (2005). The effect of stimulus strength on the speed and accuracy of a perceptual decision. Journal of vision, 5(5), 1-1.
            if v != 0:
                ERTs[i] = a / v * np.tanh(a * v)
            else:
                ERTs[i] = a**2

            # https://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
            if v > 0:
                tmp = np.exp(-2 * v * a)
                ECHs_01[i] = 1 / (1 + tmp)
            else:
                tmp = np.exp(2 * v * a)
                ECHs_01[i] = tmp / (1 + tmp)
            assert ECHs_01[i] >= 0 and ECHs_01[i] <= 1
            ECHs[i] = ECHs_01[i] * 2 - 1
            assert ECHs[i] >= -1 and ECHs[i] <= 1

            VCHs_01[i] = ECHs_01[i] * (1 - ECHs_01[i])
            tmp = 4 * VCHs_01[i]
            tmp2 = 1 - np.tanh(a * v)**2
            VCHs[i] = 1 - ECHs[i]**2  # VarX=E[X^2]-E[X]^2
            assert abs(VCHs[i]-tmp) < 1e-10
            assert abs(VCHs[i]-tmp2) < 1e-10
            assert abs(ECHs[i] / ERTs[i] - v / a) <= 1e-10

        terms_LM_nonAsym = []
        terms_GLM_nonAsym = []
        terms_LM_asym = []
        terms_GLM_asym = []
        terms_LM_nonAsym_ignoreConst = []
        terms_GLM_nonAsym_ignoreConst = []

        for i, v in enumerate(vs):
            tmp = ERTs[i]**2 / (a**2)
            terms_LM_nonAsym_ignoreConst.append(tmp)

            tmp = ERTs[i]**2 / ((2 + 2 * np.sqrt(2))**2 * a**2)
            # terms_LM_nonAsym.append(tmp)
            terms_LM_nonAsym.append(tmp**0.5)  # looks better

            # https://en.wikipedia.org/wiki/Logistic_function#:~:text=.-,Derivative,-%5Bedit%5D
            h_prime = ECHs_01[i] * (1 - ECHs_01[i])
            assert abs(4 * h_prime-VCHs[i]) < 1e-10

            tmp = a**2 * 4 * h_prime
            terms_GLM_nonAsym_ignoreConst.append(tmp)

            tmp = a**2 * 4 * h_prime / (2.4**2)
            # terms_GLM_nonAsym.append(tmp)
            terms_GLM_nonAsym.append(tmp**0.5)  # looks better

            tmp = a**2 / ERTs[i]**2 * VCHs[i] + \
                a**2 * ECHs[i]**2 / ERTs[i]**4 * VRTs[i]
            terms_LM_asym.append(1.0/tmp)
            assert abs(1.0/tmp-ERTs[i]) < 1e-10

            tmp = 4 * a**2 * h_prime
            terms_GLM_asym.append(tmp)
            assert abs(tmp-a**2 * VCHs[i]) < 1e-10

        a_2_terms_GLM_nonAsym[a] = terms_GLM_nonAsym
        a_2_terms_LM_nonAsym[a] = terms_LM_nonAsym

        a_2_terms_GLM_nonAsym_ignoreConst[a] = terms_GLM_nonAsym_ignoreConst
        a_2_terms_LM_nonAsym_ignoreConst[a] = terms_LM_nonAsym_ignoreConst

        a_2_terms_GLM_asym[a] = terms_GLM_asym
        a_2_terms_LM_asym[a] = terms_LM_asym

        a_2_ERTs[a] = ERTs
        a_2_VRTs[a] = VRTs
        a_2_ECHs[a] = ECHs
        a_2_VCHs[a] = VCHs

    LM_labels = [r"$\mathbb{E}\left[t_x\right], a=$" + fr"${as_[0]}$"]
    GLM_labels = [r"$a^2\,\mathbb{V}\left[c_x\right], a=$" + fr"${as_[0]}$"]
    if len(as_) == 2:
        LM_labels.append(r"$\mathbb{E}\left[t_x\right], a=$" + fr"${as_[1]}$")
        GLM_labels.append(
            r"$a^2\,\mathbb{V}\left[c_x\right], a=$" + fr"${as_[1]}$")
    plot(a_2_terms_GLM_asym, a_2_terms_LM_asym, vs, as_, asymp=True,
         LM_labels=LM_labels, GLM_labels=GLM_labels)

    LM_labels = [r"$m_{\text{CH,DT}}^{\text{non-asym}}, a=$" +
                 fr"${as_[0]}$"]
    GLM_labels = [r"$m_{\text{CH}}^{\text{non-asym}}, a=$" +
                  fr"${as_[0]}$"]
    if len(as_) == 2:
        LM_labels.append(
            r"$m_{\text{CH,DT}}^{\text{non-asym}}, a=$" + fr"${as_[1]}$")
        GLM_labels.append(
            r"$m_{\text{CH}}^{\text{non-asym}}, a=$" + fr"${as_[1]}$")
    plot(a_2_terms_GLM_nonAsym, a_2_terms_LM_nonAsym, vs, as_, asymp=False,
         LM_labels=LM_labels, GLM_labels=GLM_labels)


def plot(a_2_terms_GLM, a_2_terms_LM, vs, as_, asymp, LM_labels, GLM_labels):
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figure_size)

    ax.grid(axis='y', color='gray', linestyle='-', linewidth=1, alpha=0.3)
    ax.grid(axis='x', color='gray', linestyle='-', linewidth=1, alpha=0.3)
    # https://stackoverflow.com/questions/1726391/matplotlib-draw-grid-lines-behind-other-graph-elements
    ax.set_axisbelow(True)

    if len(as_) == 1:
        line_LM_a0, = plt.plot(vs, a_2_terms_LM[as_[0]],
                               label=LM_labels[0], linewidth=5, linestyle='solid', color=color_LM, alpha=1, zorder=2)
        line_GLM_a0, = plt.plot(vs, a_2_terms_GLM[as_[0]],
                                label=GLM_labels[0], linewidth=5, linestyle='solid', color=color_GLM, alpha=1, zorder=1)
    elif len(as_) == 2:
        line_LM_a0, = plt.plot(vs, a_2_terms_LM[as_[0]],
                               label=LM_labels[0], linewidth=5, linestyle='solid', color=color_LM, alpha=0.6, zorder=2)
        line_GLM_a0, = plt.plot(vs, a_2_terms_GLM[as_[0]],
                                label=GLM_labels[0], linewidth=5, linestyle='solid', color=color_GLM, alpha=0.6, zorder=1)
        line_LM_a1, = plt.plot(vs, a_2_terms_LM[as_[1]],
                               label=LM_labels[1], linewidth=5, linestyle='dashed', color=color_LM, alpha=1, zorder=4)
        line_GLM_a1, = plt.plot(vs, a_2_terms_GLM[as_[1]],
                                label=GLM_labels[1], linewidth=5, linestyle='dashed', color=color_GLM, alpha=1, zorder=3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel(r"$x^\top \theta^*$", fontsize=30)
    ax.xaxis.set_label_coords(0.5, -0.125)

    if asymp:
        if y_ticks_asymp is not None:
            ax.set_yticks(y_ticks_asymp)
    else:
        if y_ticks_nonasymp is not None:
            ax.set_yticks(y_ticks_nonasymp)

    if len(as_) == 1:
        handles = [line_LM_a0,  line_GLM_a0]
        labels = [LM_labels[0],  GLM_labels[0]]
    elif len(as_) == 2:
        handles = [line_LM_a1, line_LM_a0, line_GLM_a1, line_GLM_a0]
        labels = [LM_labels[1], LM_labels[0], GLM_labels[1], GLM_labels[0]]
    if asymp:
        bbox_to_anchor = (0.6, 0.38)
        markerfirst = True
    else:
        bbox_to_anchor = (0.6, 0.38)
        markerfirst = True

    legend = ax.legend(
        handles, labels,
        bbox_to_anchor=bbox_to_anchor,
        fontsize=30, markerfirst=markerfirst,
        handletextpad=0.2,
        frameon=True,    # Enable the frame to set its color
        fancybox=True,   # Rounded edges for the box
        framealpha=1,    # Fully opaque background
        borderpad=0,     # Padding between the text and the edge of the box
    )
    # Set legend background color (white) and remove the border
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('none')  # No border

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    file_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(file_path)
    path = os.path.join(dir_path, "plot_DDM_")
    if asymp:
        path += "asymp"
    else:
        path += "nonAsymp"
    path += "_" + str(as_)+".pdf"
    plt.savefig(path, format='pdf', bbox_inches='tight')
    print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
