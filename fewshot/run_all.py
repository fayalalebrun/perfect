import json
import os

task = ["twitter", "reddit"]
ks = [16, 12, 8, 4]
models = ["roberta-base", "roberta-large"]

for t in task:
    for k in ks:
        for model in models:
            f = open("configs/perfect.json","r")
            lines = f.read()
            dec = json.loads(lines)
            dec["task"] = t
            dec["K"] = k
            dec["model_name_or_path"] = model
            enc = json.dumps(dec)
            text_file = open("configs/temp.json", "w")
            text_file.write(enc)
            text_file.close()
            os.system('script -c "python3 run_clm.py configs/temp.json" results/' + t + str(k) + model + '.txt')
            
