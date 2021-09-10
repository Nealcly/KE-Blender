import pandas as pd
import logging
from seq2seq_model import Seq2SeqModel
import json
import re
import torch
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(42)

def read_wizard_json(file_path):
    with open(file_path, 'r') as f:
        file = json.load(f)

    data = []
    for line in file:
        tmp_source = ''
        for i in line['dialog']:
            utt = i['text']
            if tmp_source != '':
                data.append([tmp_source, "__start__ " + utt + " __end__"])
                # add split '\t' for blender
                tmp_source = tmp_source + "\t" + utt
            else:
                tmp_source = utt
    return data

def read_wizard_definition(file_path):

    with open(file_path, 'r') as f:
        file = json.load(f)

    data = []
    for line in file:
        # line.keys() ['chosen_topic', 'persona', 'wizard_eval', 'dialog', 'chosen_topic_passage']
        # dialog.keys() dict_keys(['speaker', 'text', 'candidate_responses', 'retrieved_passages', 'retrieved_topics'])
        for i in line['dialog']:
            utt = i['text']
            external_passage = i['retrieved_passages']
            for j in external_passage:
                if list(j.keys())[0].lower() in utt.lower():
                    know_key = list(j.keys())[0]
                    try:
                        # retrieved knowledge is available
                        # we do not use gold knowledge
                        utt_mask = re.sub(know_key, '[MASK]', utt, flags=re.IGNORECASE)
                        knowledge = ('\t').join(j[know_key])
                        data.append([utt_mask, "__defi__ " + knowledge + " __end__"])
                    except:
                        continue
    return data

def read_wizard_concat_json(file_path):

    with open(file_path, 'r') as f:
        file = json.load(f)

        # print(file[0].keys())
        # print(file[0]['dialog'])
    data = []
    for line in file:

        tmp_source = ''
        for i in line['dialog']:
            utt = i['text']

            external_passage = i['retrieved_passages']
            for j in external_passage:
                # print(list(j.keys()))
                if list(j.keys())[0].lower() in utt.lower():
                    # retrieved knowledge is available
                    know_key = list(j.keys())[0]
                    knowledge = (" ").join(j[know_key])
                    if tmp_source != '':
                        data.append([tmp_source + " " + knowledge, "__start__ " + utt + " __end__"])
                        tmp_source = tmp_source + "\t" + utt
                    else:
                        tmp_source = utt
                    break
    return data

def read_hypernym(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        file = f.readlines()
    for line in file:
        source, target = line.strip().split('\001')
        data.append([source, "__hype__ " + target + " __end__"])
    return data


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


train_data = read_wizard_json('wizard_of_wikipedia/train.json') + read_wizard_definition('wizard_of_wikipedia/train.json') + read_hypernym("wizard_of_wikipedia/train_sim.txt") + read_wizard_concat_json(('wizard_of_wikipedia/train.json'))
train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])
eval_data = read_wizard_json('wizard_of_wikipedia/valid_topic_split.json')
eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])
test_data = read_wizard_json('wizard_of_wikipedia/test_topic_split.json')
test_df = pd.DataFrame(test_data, columns=["input_text", "target_text"])

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 512,
    "train_batch_size": 16,
    "num_train_epochs": 3,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": True,
    "evaluate_during_training": True,
    "evaluate_generated_text": True,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": False,
    "max_length": 128,
    "manual_seed": 42,
    "n_gpu": 8,
    "gradient_accumulation_steps": 4,
    "output_dir": "/KE-Blender",
    # "weight_decay": 0.5, # weight - 0.5
}

# Initialize model
model = Seq2SeqModel(
    encoder_decoder_type="blender",
    encoder_decoder_name="facebook/blenderbot_small-90M",
    args=model_args,
)

# Train the model
model.train_model(train_df, eval_data=eval_df)