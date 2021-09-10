from nltk.corpus import wordnet as wn
from itertools import chain

import json
import re
import nltk
# nltk.download('wordnet')
def read_wizard_hyper_json(file_path):

    with open(file_path, 'r') as f:
        file = json.load(f)

        # print(file[0].keys())
        # print(file[0]['dialog'])
    data = []
    for line in file:

        for i in line['dialog']:
            utt = i['text']
            external_passage = i['retrieved_passages']
            for j in external_passage:
                # print(list(j.keys()))
                if list(j.keys())[0].lower() in utt.lower():
                    know_key = list(j.keys())[0]
                    # print(know_key)
                    try:
                        if len(know_key.split(' ')) > 1:
                            word = know_key.split(' ')[-1]
                            ori = (' ').join(know_key.split(' ')[:-1])
                            replace_word = wn.synsets(word.lower())[0].hypernyms()[0].name().split('.')[0]
                            if "_" in replace_word:
                                replace_word = (' ').join(replace_word.split('_'))
                            replace_word = ori + ' ' + replace_word
                            utt_mask = re.sub(know_key, replace_word, utt, flags=re.IGNORECASE)
                        else:
                            word = know_key
                            replace_word = wn.synsets(word.lower())[0].hypernyms()[0].name().split('.')[0]
                            if "_" in replace_word:
                                replace_word = (' ').join(replace_word.split('_'))
                            utt_mask = re.sub(know_key, replace_word, utt, flags=re.IGNORECASE)
                        data.append([utt, utt_mask])
                    except:
                        continue
    return data

sim_data = read_wizard_hyper_json('wizard_of_wikipedia/train.json')

with open('wizard_of_wikipedia/train_sim.txt', 'w', encoding='utf-8', errors='ignore') as f:
    for line in sim_data:
        f.write(line[0] + '\001' + line[1] + '\n')