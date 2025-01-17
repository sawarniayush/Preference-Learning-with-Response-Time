import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
from matplotlib.lines import Line2D

figure_size = (10, 6)


def main():
    # Enable LaTeX rendering in Matplotlib
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # https://stackoverflow.com/questions/23824687/text-does-not-work-in-a-matplotlib-label
    # https://stackoverflow.com/questions/65426069/use-of-mathbb-in-matplotlib
    # for \text and \mathbb command
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amsfonts,amssymb}'

    as_ = [0.8, 1.8]
    vs = np.arange(-5, 5, 1e-3)

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

        plot(vs, ECHs, VCHs, ERTs, VRTs, a, plot_ch=True, plot_rt=True)
        plot(vs, ECHs, VCHs, ERTs, VRTs, a, plot_ch=True, plot_rt=False)
        plot(vs, ECHs, VCHs, ERTs, VRTs, a, plot_ch=False, plot_rt=True)


def plot(vs, ECHs, VCHs, ERTs, VRTs, a, plot_ch, plot_rt):
    assert plot_ch+plot_rt >= 1
    fig, ax = plt.subplots(figsize=figure_size)

    if plot_ch:
        ax.fill_between(vs, ECHs - np.sqrt(VCHs), ECHs + np.sqrt(VCHs),
                        alpha=0.3, color="#66c2a5", label='', edgecolor='none',)
    if plot_rt:
        ax.fill_between(vs, ERTs - np.sqrt(VRTs), ERTs + np.sqrt(VRTs),
                        alpha=0.3, color="#fc8d62", label='', edgecolor='none',)
    if plot_ch:
        ax.plot(vs, ECHs,
                label=r'$\mathbb{E}\left[c_x\right]\pm\sigma_{c,x}$',
                color="#66c2a5", linewidth=5)
    if plot_rt:
        ax.plot(vs, ERTs,
                label=r'$\mathbb{E}\left[t_x\right]\pm\sigma_{t,x}$',
                color="#fc8d62", linewidth=5)

    ax.axhline(y=0, color='gray', linestyle='-', linewidth=3, alpha=0.2)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=3, alpha=0.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel(r"$x^\top \theta^*$", fontsize=40)
    ax.xaxis.set_label_coords(0.5, -0.12)  # shift x label a bit

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    ax.set_ylim([-1.5, 6])
    ax.set_yticks([0, 2, 4, 6])

    custom_handles = [
        Line2D([0], [0], color="#fc8d62", lw=8),
        Line2D([0], [0], color="#66c2a5", lw=8),
    ]
    ax.legend(custom_handles,
              [
                  r'$\mathbb{E}\left[t_x\right]\pm\sigma_{t,x}$',
                  r'$\mathbb{E}\left[c_x\right]\pm\sigma_{c,x}$',
              ],
              fontsize=40, bbox_to_anchor=(-0.04, 1.08),
              loc='upper left',  frameon=False)

    tmp = ""
    if plot_ch:
        tmp += "CH"
    if plot_rt:
        tmp += "RT"
    file_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(file_path)
    path = os.path.join(dir_path, "plot_DDM_ERT_ECH_"+str(a)+"_"+tmp+".pdf")
    plt.savefig(path, format='pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
