import os
import pandas as pd

# read csvs from directory
directory = "MFTC"

# get all csv files
all_csvs = os.listdir(directory)
all_csvs = list(filter(lambda f: f.endswith(".csv"), all_csvs))

# merge csvs into one dataframe
all_data = pd.DataFrame()
for f in all_csvs:
    dataset = pd.read_csv(directory + "/" + f)
    all_data = all_data.append(dataset, ignore_index=True)

# shuffle data
all_data = all_data.sample(frac=1).reset_index(drop=True)

# select 4000 samples
all_data = all_data[:4000]

# make folder if doesn't exist
if not os.path.exists("MFTC/Selection"):
    os.makedirs("MFTC/Selection")

# save to csv file
all_data.to_csv("MFTC/Selection/MFTC_shuffled.csv", index=False)