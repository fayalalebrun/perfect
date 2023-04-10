import pandas as pd
import argparse, os, json, random, shutil

parser = argparse.ArgumentParser(description="Create train and test set for Reddit dataset in the PERFECT format")

parser.add_argument('-f', dest='file', type=str, help="MFRC .csv file")
parser.add_argument('-out_dir', dest='out_dir', default="fewshot/datasets_processed/reddit", type=str, help="Directory to store test and train sets")

args = parser.parse_args()

f = args.file
out_dir = args.out_dir

# Construct list of labels
all_labels = dict()
label_count = 0

dataset = pd.read_csv(f)

labels = dataset["annotation"]
for label_list in labels:
    label_list = label_list.split(",")

    for label in label_list:
        if label not in all_labels:
            all_labels[label] = label_count
            label_count += 1

print(label_count, " labels: ", all_labels)

# Create .json datasets
samples_dict = dict()
i = 0
for i, row in dataset.iterrows():
    if row["text"] not in samples_dict:
        samples_dict[row["text"]] = [0] * label_count

    label_list = row["annotation"].split(",")

    for label in label_list:
        samples_dict[row["text"]][all_labels[label]] = 1

all_samples = []
for sample in samples_dict.keys():
    sample = {"label": samples_dict[sample], "source": sample}
    sample = json.dumps(sample)
    all_samples.append(sample)

random.seed(10)
random.shuffle(all_samples)
train = all_samples[:1000]
test = all_samples[1000:1100]

if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)

with open(out_dir + "/" + "train.json", 'w') as f:
    f.write("\n".join(map(str, train)))
f.close()
with open(out_dir + "/" + "test.json", 'w') as f:
    f.write("\n".join(map(str, test)))
f.close()
