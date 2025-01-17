# https://stackoverflow.com/questions/2536047/convert-a-dta-file-to-csv-without-stata-software
import os
import pandas as pd
file_path = os.path.realpath(__file__)
dir_path = os.path.dirname(file_path)
data = pd.io.stata.read_stata(os.path.join(dir_path, 'ClitheroDataset.dta'))
data.to_csv(os.path.join(dir_path, 'ClitheroDataset.csv'))
