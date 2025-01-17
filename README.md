# Enhancing Preference-based Linear Bandits via Human Response Time

## Authors

[Shen Li](https://shenlirobot.github.io/)<sup>1</sup>*,
[Yuyang Zhang](https://scholar.google.com/citations?user=NiBKGakAAAAJ&hl=en)<sup>2</sup>*,
[Zhaolin Ren](https://www.zhaolinren.com/)<sup>2</sup>,
[Claire Liang](https://cyl48.github.io/)<sup>1</sup>,
[Na Li](https://nali.seas.harvard.edu/)<sup>2</sup>,
[Julie A. Shah](https://interactive.mit.edu/about/people/julie-shah/)<sup>1</sup>

<sup>1</sup>MIT, <sup>2</sup>Harvard

First two authors have equal contribution.

**NeurIPS 2024 Oral (Acceptance Rate: 0.39%)** 🎉

If you have questions, please feel free to reach out to Shen at shenli@mit.edu.


## Links

- [📄 Paper on arXiv](https://arxiv.org/abs/2409.05798)
- [🖼️ Poster](https://shenlirobot.github.io/docs/24-NeurIPS-Li-Zhang-Ren-Liang-Li-Shah-poster.pdf)
- [🎥 Oral Video (0:59–19:19) (Public on 02/25)](https://neurips.cc/virtual/2024/session/98061)
- [📊 Slides](https://shenlirobot.github.io/docs/24-NeurIPS-Li-Zhang-Ren-Liang-Li-Shah-slides.pdf)


## Abstract

Interactive preference learning systems infer human preferences by presenting queries as pairs of options and collecting binary choices. Although binary choices are simple and widely used, they provide limited information about preference strength. To address this, we leverage human response times, which are inversely related to preference strength, as an additional signal. We propose a computationally efficient method that combines choices and response times to estimate human utility functions, grounded in the EZ diffusion model from psychology. Theoretical and empirical analyses show that for queries with strong preferences, response times complement choices by providing extra information about preference strength, leading to significantly improved utility estimation. We incorporate this estimator into preference-based linear bandits for fixed-budget best-arm identification. Simulations on three real-world datasets demonstrate that using response times significantly accelerates preference learning compared to choice-only approaches.


## Code to reproduce the simulation results
### Installation
* Tested on `julia version 1.10.4` by `Julia --version`
* Open a terminal
* `cd` the parent folder of this repository
* `julia`
* Enter the package manager mode by pressing `]`
  * https://docs.julialang.org/en/v1/stdlib/REPL/#Pkg-mode
* `activate linear_bandits_response_time`
* `instantiate`
* `update`
* Press `ctrl + c` to leave the package manager mode
* `exit()`

### Processing 3 datasets of choices and response times for empirical evaluation
* Food-risk dataset with choices (-1 or 1) (see Appendix D.2 in [the paper](https://arxiv.org/pdf/2409.05798))
  * This dataset was originally contributed by `S. M. Smith and I. Krajbich. Attention and choice across domains. Journal of Experimental Psychology: General, 147(12):1810, 2018.`
  * This dataset was downloaded from the `data` repository for the paper `X. Yang and I. Krajbich. A dynamic computational model of gaze and choice in multi-attribute decisions. Psychological Review, 130(1):52, 2023.` at https://osf.io/d7s6c/.
  * In our repository, the data file is located at `data/foodrisk.csv`.
  * Data processing
    * Run `data/foodrisk_DDM_training.jl` to generate DDM parameters for each subject in this dataset, saved in `data/foodrisk_subjectIdx_2_params.csv` and `data/foodrisk_subjectIdx_2_params.jld`

* Snack dataset with choices (yes or no) (see Appendix D.3 in [the paper](https://arxiv.org/pdf/2409.05798))
  * This dataset was originally contributed by `J. A. Clithero. Improving out-of-sample predictions using response times and a model of the decision process. Journal of Economic Behavior & Organization, 148:344–375, 2018. ISSN 0167-2681.`
  * This dataset was downloaded from the `Supplemental Material` section for the paper `C. Alós-Ferrer, E. Fehr, and N. Netzer. Time will tell: Recovering preferences when choices are noisy. Journal of Political Economy, 129(6):1828–1877, 2021.` at https://www.journals.uchicago.edu/doi/suppl/10.1086/713732
  * In our repository, the data file is located at `data/ClitheroDataset.dta`.
  * Data processing
    * Run `python data/ClitheroDataset_convert_dta_2_csv.py` to convert `dta` file to a `csv` file.
    * Run `data/Clithero_DDM_training.jl` to generate DDM parameters for each subject in this dataset, saved in `data/Clithero_subjectIdx_2_params.csv` and `data/Clithero_subjectIdx_2_params.jld`

* Snack dataset with choices (-1 or 1) (see Appendix D.4 in [the paper](https://arxiv.org/pdf/2409.05798))
  * This dataset was originally contributed by `I. Krajbich, C. Armel, and A. Rangel. Visual fixations and the computation and comparison of value in simple choice. Nature Neuroscience, 13(10):1292–1298, 2010. doi: 10.1038/nn.2635.`
  * This dataset was downloaded from the `data` repository for the paper `D. Fudenberg, P. Strack, and T. Strzalecki. Speed, accuracy, and the optimal timing of choices. American Economic Review, 108(12):3651–84, December 2018. doi: 10.1257/aer.20150742.` at https://www.aeaweb.org/articles?id=10.1257/aer.20150742.
  * In our repository, the data file is located at `data/KrajbichDataset.csv`.
  * Data processing
    * Run `data/Krajbich_DDM_training.jl` to generate DDM parameters for each subject in this dataset, saved in `data/Krajbich_subjectIdx_2_params.jld.csv` and `data/Krajbich_subjectIdx_2_params.jld.jld`

### Reproducing plots in the paper
* Fig.1(a) in [the paper](https://arxiv.org/pdf/2409.05798)
  * Run `otherPlots/plot_DDM_simulation.jl`, which will produce a plot `otherPlots/plot_DDM_simulation_S11.pdf`
* Fig.1(b)(c) in [the paper](https://arxiv.org/pdf/2409.05798)
  * Run `python otherPlots/plot_DDM_ERT_ECH.py`, which will produce two plots `otherPlots/plot_DDM_ERT_ECH_0.8_CHRT.pdf` and `otherPlots/plot_DDM_ERT_ECH_1.8_CHRT.pdf`

* Fig.2 in [the paper](https://arxiv.org/pdf/2409.05798)
  * Run `python otherPlots/plot_DDM_asymp_nonasymp.py`, which will produce two plots `otherPlots/plot_DDM_asymp_[0.8, 1.8].pdf` and `otherPlots/plot_DDM_nonAsymp_[0.8, 1.8].pdf`

* Fig.3 in [the paper](https://arxiv.org/pdf/2409.05798)
  * Run `experiments/run_SSM_estimation.jl`, which will produce 2 result files, `experiments/run_SSM_estimation/run_SSM_estimation.dat` and `experiments/run_SSM_estimation/run_SSM_estimation.yaml`
  * Run `run_SSM_estimation_plot.py`, which will produce 4 plots, `experiments/run_SSM_estimation/run_SSM_estimation_GLM_trans.pdf`, `experiments/run_SSM_estimation/run_SSM_estimation_GLM_weakPref.pdf`, `experiments/run_SSM_estimation/run_SSM_estimation_LM_trans.pdf`, and `experiments/run_SSM_estimation/run_SSM_estimation_colorbar.pdf`

---

* Fig.4(a) and Fig.5
  * Run `experiments/run_foodrisk.jl`, which will produce result files, `experiments/run_foodrisk/processed_result.yaml`, `experiments/run_foodrisk/results_12s1.dat`, `experiments/run_foodrisk/results_12s2.dat`, `experiments/run_foodrisk/results_12s3.dat`, ...
  * Run `python experiments/run_foodrisk_Clithero_Krajbich_plot.py` (set the global variable `problem_name = "foodrisk"`), which will produce `experiments/run_foodrisk/foodrisk_2_NoLegend.pdf` (Fig.4(a)) and `experiments/run_foodrisk/foodrisk_6_NoLegend.pdf` (Fig.5)

---

* Fig.4(b) and Fig.7
  * Run `experiments/run_Clithero.jl` (set the global variable in this file `tune_η = false`), which will produce result files, `experiments/run_Clithero/processed_result.yaml`, `experiments/run_Clithero/results_7s1.dat`, `experiments/run_Clithero/results_7s2.dat`, `experiments/run_Clithero/results_7s3.dat`, ...
  * Run `python experiments/run_foodrisk_Clithero_Krajbich_plot.py` (set the global variable `problem_name = "Clithero"`), which will produce `experiments/run_Clithero/Clithero_2_NoLegend.pdf` (Fig.4(b)) and `experiments/run_Clithero/Clithero_6_NoLegend.pdf` (Fig.7)
* Fig.6:
  * Run `experiments/run_Clithero.jl` (set the global variable in this file `tune_η = true`), which will produce result files, `experiments/run_Clithero_tune_eta/alg_name_2_eta_2_mistakeAtBudgetMeans.yaml`, `experiments/run_Clithero_tune_eta/best_etas.yaml` (saving the best eta's for each GSE variation), `experiments/run_Clithero_tune_eta/results_7s1.dat`, `experiments/run_Clithero_tune_eta/results_7s2.dat`, `experiments/run_Clithero_tune_eta/results_7s3.dat`, ...
  * Run `python experiments/run_foodrisk_Clithero_Krajbich_plot_to_determine_best_eta.py` (set the global variable `problem_name = "Clithero"`), which will produce these files corresponding to (a)-(f) in Fig.8:
    * `experiments/run_Clithero_tune_eta/Clithero_eta_LM_trans.pdf`
    * `experiments/run_Clithero_tune_eta/Clithero_eta_LM_trans_noSubt.pdf`
    * `experiments/run_Clithero_tune_eta/Clithero_eta_GLM_weakPref.pdf`
    * `experiments/run_Clithero_tune_eta/Clithero_eta_GLM_trans.pdf`
    * `experiments/run_Clithero_tune_eta/Clithero_eta_Wagenmakers07Eq5_trans.pdf`
    * `experiments/run_Clithero_tune_eta/Clithero_eta_Chiong24Lemma1_trans_legend.pdf`


---

* Fig.4(c) and Fig.9
  * Run `experiments/run_Krajbich.jl` (set the global variable in this file `tune_η = false`), which will produce result files, `experiments/run_Krajbich/processed_result.yaml`, `experiments/run_Krajbich/results_8s1.dat`, `experiments/run_Krajbich/results_8s2.dat`, `experiments/run_Krajbich/results_8s3.dat`, ...
  * Run `python experiments/run_foodrisk_Clithero_Krajbich_plot.py` (set the global variable `problem_name = "Krajbich"`), which will produce `experiments/run_Krajbich/Krajbich_2_NoLegend.pdf` (Fig.4(c)) and `experiments/run_Krajbich/Krajbich_6_NoLegend.pdf` (Fig.9)
* Fig.8:
  * Run `experiments/run_Krajbich.jl` (set the global variable in this file `tune_η = true`), which will produce result files, `experiments/run_Krajbich_tune_eta/alg_name_2_eta_2_mistakeAtBudgetMeans.yaml`, `experiments/run_Krajbich_tune_eta/best_etas.yaml` (saving the best eta's for each GSE variation), `experiments/run_Krajbich_tune_eta/results_8s1.dat`, `experiments/run_Krajbich_tune_eta/results_8s2.dat`, `experiments/run_Krajbich_tune_eta/results_8s3.dat`, ...
  * Run `python experiments/run_foodrisk_Clithero_Krajbich_plot_to_determine_best_eta.py` (set the global variable `problem_name = "Krajbich"`), which will produce these files corresponding to (a)-(f) in Fig.8:
    * `experiments/run_Krajbich_tune_eta/Krajbich_eta_LM_trans.pdf`
    * `experiments/run_Krajbich_tune_eta/Krajbich_eta_LM_trans_noSubt.pdf`
    * `experiments/run_Krajbich_tune_eta/Krajbich_eta_GLM_weakPref.pdf`
    * `experiments/run_Krajbich_tune_eta/Krajbich_eta_GLM_trans.pdf`
    * `experiments/run_Krajbich_tune_eta/Krajbich_eta_Wagenmakers07Eq5_trans.pdf`
    * `experiments/run_Krajbich_tune_eta/Krajbich_eta_Chiong24Lemma1_trans_legend.pdf`
























