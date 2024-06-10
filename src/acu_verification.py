from model import StableBeluga
import sys
from tqdm import tqdm
import json
import numpy as np

lm = StableBeluga()

acu_verification_prompt = """Read the passage and the statement. Then, answer whether all the information in the statement can be found in the passage.

Passage: {Document}

Statement: {Fact}

You are ONLY allowed to answer with Yes or No."""

data = [json.loads(line) for line in open(sys.argv[1])]
out_file = sys.argv[2]

for dat in tqdm(data):
    document = dat["document"]
    acus = dat["acus"]

    acu_predictions = []
    for acu in tqdm(acus,leave=False):
        message = acu_verification_prompt.format(Document=document, Fact=acu)
        
        output, output_norm = lm.run_binary(message)
        acu_predictions.append(output_norm)
    
    dat["acu_predictions"] = acu_predictions
    dat["acueval_score"] = np.mean(acu_predictions)

with open(out_file, "w") as f:
    for dat in data:
        f.write( json.dumps(dat) + "\n" )