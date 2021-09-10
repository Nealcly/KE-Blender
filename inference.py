import pandas as pd
import logging
from seq2seq_model import Seq2SeqModel
import json
import re

def read_wizard_json(file_path):
    with open(file_path, 'r') as f:
        file = json.load(f)

        # print(file[0].keys())
        # print(file[0]['dialog'])
    data = []
    for line in file:
        tmp_source = ''
        for i in line['dialog']:
            utt = i['text']
            if tmp_source != '':
                data.append([tmp_source, "__start__ " + utt + " __end__"])
                # add split '\n' for blender
                tmp_source = tmp_source + "\t" + utt
            else:
                tmp_source = utt
    return data


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


# train_data = read_wizard_json('wizard_of_wikipedia/train.json')
# train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])
eval_data = read_wizard_json('wizard_of_wikipedia/valid_topic_split.json')
eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])
test_data = read_wizard_json('wizard_of_wikipedia/test_topic_split.json')
test_df = pd.DataFrame(test_data, columns=["input_text", "target_text"])

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 512,
    "train_batch_size": 64,
    "eval_batch_size": 4,
    "num_train_epochs": 3,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": True,
    "evaluate_during_training": True,
    "evaluate_generated_text": False,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": False,
    "max_length": 128,
    "manual_seed": 42,
    "n_gpu": 8,
    "gradient_accumulation_steps": 2,
    "output_dir": "bz128",
    "weight_decay": 0.5,
    "num_return_sequences": 1,
}
#     "do_sample": True,
#     "top_k": 50,

# Initialize model
model = Seq2SeqModel(
    encoder_decoder_type="blender",
    encoder_decoder_name="KE-Blender/dialogue_model/two_loss_withmask/checkpoint-8067-epoch-3",
    args=model_args,
)

results = model.eval_model(test_df)
print(results)
test_text = [i[0] for i in test_data]
decode_result = model.predict(test_text)
with open('dialogue_model/two_loss_withmask/blender_two_loss_withmask_decode_epoch3_greedy.txt', 'w') as f:
    for i in decode_result:
        f.write(str(i) + '\n')
