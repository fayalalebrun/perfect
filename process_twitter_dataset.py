import argparse, os, json, random, shutil
import pandas as pd

# Load all .csv files
parser = argparse.ArgumentParser(description="Create train and test set for Twitter dataset in the PERFECT format")
parser.add_argument('-dir', dest='directory', type=str, help="Directory with Twitter dataset")
parser.add_argument('-out_dir', dest='out_dir', default="fewshot/datasets_processed/twitter", type=str, help="Directory to store test and train sets")
args = parser.parse_args()

directory = args.directory
out_dir = args.out_dir

# Get all csv files
all_csvs = os.listdir(directory)
all_csvs = list(filter(lambda f: f.endswith(".csv"), all_csvs))

# Construct list of labels
all_labels = dict()
label_count = 0
for f in all_csvs:
    dataset = pd.read_csv(directory + "/" + f)

    labels = dataset.columns[2:]
    for label in labels:
        if label not in all_labels:
            all_labels[label] = label_count
            label_count += 1

print(label_count, " labels: ", all_labels)
    
# Create .json datasets
all_samples = []
for f in all_csvs:
    dataset = pd.read_csv(directory + "/" + f)

    for i, row in dataset.iterrows():
        sample = {"label": [], "source": row["text"]}

        for (col, val) in zip(dataset.columns[2:], row[2:]):
            if val  == 1:
                sample["label"].append(all_labels[col])

        one_hot = [0] * label_count
        for label in sample["label"]: one_hot[label] = 1
        sample["label"] = one_hot

        sample = json.dumps(sample)
        all_samples.append(sample)

random.seed(10)
random.shuffle(all_samples)
n = len(all_samples)
train = all_samples[:int(n/2)]
test = all_samples[int(n/2):]

if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)

with open(out_dir + "/" + "train.json", 'w') as f:
    f.write("\n".join(map(str, train)))
f.close()
with open(out_dir + "/" + "test.json", 'w') as f:
    f.write("\n".join(map(str, test)))
f.close()
