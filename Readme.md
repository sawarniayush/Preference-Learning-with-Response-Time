This project requires sklearn, numpy and standard plotting libraries like matplotlib.

In this folder we have the code for the linear reward model and the plots in main paper can be generated from notebook plots.ipynb. The first two cells provide the code for plots in figure 1.

The data is generated and stored in data_gen/data_files and code to generate the data files is in data_gen/data_gen_main.py.

The code to process the data and train multiple models (based on log-loss, orthogonal loss and non-orthogonal) and the master code for it is in dictionary_gen_main.py. This file calls dictionary_gen_functions.py to train the linear models and then dumps the data (accuracy, ell_2 error) into dictionary_diff_diff_data.pkl. All this is for the first loss function where time estimate t is used as nuisance

For the second orthogonal loss function with y as nuisance, the master code is in dictionary_gen_y_nuisance_main.py which dumps the data into dictionary_diff_diff_data_y_nuisance.pkl


In the first file, we consider the case with a fixed value of a (ranging from 0.8 to 2.2) for every user which is used by the algorithm to construct the orthogonalized losses. 


