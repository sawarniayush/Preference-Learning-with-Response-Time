# Linear Reward Model Project

This project requires `sklearn`, `numpy`, and standard plotting libraries like `matplotlib`.

## Project Structure

### Data Generation
- Data is generated and stored in `data_gen/data_files`
- Code to generate data files: `data_gen/data_gen_main.py`

### Plotting
- Main plotting code: `plots.ipynb`
- The first two cells provide code for plots in Figure 1

### Model Training and Processing

#### First Loss Function (Time Estimate t as Nuisance)
- Master code: `dictionary_gen_main.py`
- Training functions: `dictionary_gen_functions.py`
- Output data (accuracy, ell_2 error): `dictionary_diff_diff_data.pkl`

#### Second Orthogonal Loss Function (y as Nuisance)
- Master code: `dictionary_gen_y_nuisance_main.py`
- Output data: `dictionary_diff_diff_data_y_nuisance.pkl`

## Algorithm Details

In the first implementation, we consider the case with a fixed value of `a` (ranging from 0.8 to 2.2) for every user, which is used by the algorithm to construct the orthogonalized losses.