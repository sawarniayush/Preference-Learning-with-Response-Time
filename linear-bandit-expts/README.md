## Code to reproduce the simulation results
* This code is designed based on the [code](https://github.com/shenlirobot/linear_bandits_response_time) for the paper [`Li, S., Zhang, Y., Ren, Z., Liang, C., Li, N., & Shah, J. A. (2024). Enhancing preference-based linear bandits via human response time. In Proceedings of the 38th Conference on Neural Information Processing Systems (NeurIPS 2024).`](https://proceedings.neurips.cc/paper_files/paper/2024/file/1e2dd2f1efbc6e65b68f17ce6e158b34-Paper-Conference.pdf)

This project requires both julia and python setup to run.


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

---

### Python Environment
* Create a virtual Python environment that includes the following libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`
* Ensure this environment is active before running any Julia scripts that call Python

---

### Testing whether parallel version of python is rightly integrated with Julia
* To run the parallel experiment script, execute `experiments/test_distributed_python.jl`
* Before running, set the correct Python path in the file `experiments/run_foodrisk_parallel.jl` by modifying  
  `ENV["PYTHON"] = "/path/to/your/python"`
* For example, if your virtual environment is located at `~/venv`, set  
  `ENV["PYTHON"] = "/home/username/venv/bin/python"`
* Verify that the Python setup works in Julia by running:  
  `using PyCall`  
  `pyimport("sklearn")`
* If no error occurs, your Python environment is correctly configured

### Running the FoodRisk Experiment (Parallel Version)
* To run the parallel experiment script, execute `julia run_foodrisk_parallel.jl` in folder `experiments/`
* Before running, set the correct Python path in the file `run_foodrisk_parallel.jl` by modifying  
  `ENV["PYTHON"] = "/path/to/your/python"`

### Processing 3 datasets of choices and response times for empirical evaluation
* Food-risk dataset with choices (-1 or 1) (see Appendix D.1 in [the paper](https://arxiv.org/abs/2505.22820))
  * This dataset was originally contributed by `S. M. Smith and I. Krajbich. Attention and choice across domains. Journal of Experimental Psychology: General, 147(12):1810, 2018.`
  * This dataset was downloaded from the `data` repository for the paper `X. Yang and I. Krajbich. A dynamic computational model of gaze and choice in multi-attribute decisions. Psychological Review, 130(1):52, 2023.` at https://osf.io/d7s6c/.
  * In our repository, the data file is located at `data/foodrisk.csv`.
  * Data processing
    * Run `data/foodrisk_DDM_training.jl` to generate DDM parameters for each subject in this dataset, saved in `data/foodrisk_subjectIdx_2_params.csv` and `data/foodrisk_subjectIdx_2_params.jld`
### Reproducing tabules in Appendix D.1

* Run `experiments/run_foodrisk.jl`, which will produce result files, `experiments/run_foodrisk/processed_result.yaml`, `experiments/run_foodrisk/results_12s1.dat`, `experiments/run_foodrisk/results_12s2.dat`, `experiments/run_foodrisk/results_12s3.dat`, ...
* One can additionally separately generate `experiments/run_foodrisk/processed_result.yaml` by running function `process_results_for_Python_plotting` function in `src/experiment_helpers.jl` with appropraite arguments.
* Finally run `summarise_yaml.py` in `experiments/` folder to get the desired table.



