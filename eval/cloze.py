from transformers import BertTokenizer, BertForMaskedLM
import json

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
                data.append([tmp_source, utt])
                # add split '\n' for blender
                tmp_source = tmp_source + "\t" + utt
            else:
                tmp_source = utt
    return data

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertForMaskedLM.from_pretrained("bert-large-uncased")
model.eval()



file = read_wizard_json('wizard_of_wikipedia/test_topic_split.json')

decode_file = 'checkpoint-12788-epoch-4.deocode_withoutknow.txt'

with open(decode_file, 'r', errors='ignore') as f:
    decode_file = f.readlines()

count = 0
correct_20 = 0
correct_10 = 0
correct_5 = 0
total = 0
correct = 0
for line, decode_str in zip(file, decode_file):
    if line[0] != 'None':
        text = "[CLS] " + line[0] + " [SEP] " + decode_str + " [SEP]"
        gold = line[2]
        length = len(tokenizer.tokenize(gold))
        if len(tokenizer.tokenize(gold)) > 1:
            gold = tokenizer.tokenize(gold)[0]

        text = text.replace('[MASK]', (" ").join(['[MASK]' for i in range(length)]))
        token_text = tokenizer.tokenize(text)
        for i in range(len(token_text)):
            if token_text[i] == "[MASK]":
                break
        if len(tokenizer.tokenize(gold)) and len(line[2].split()) == 1:
            encoded_input = tokenizer(text, return_tensors='pt', add_special_tokens=False)
            output = model(**encoded_input)[0]
            logits = output[0, i, :].detach()
            values, predictions = logits.topk(20)
            predictions_list = [tokenizer.decode([word]) for word in predictions]
            if gold.lower() in predictions_list:
                correct_20 += 1
            if gold.lower() in predictions_list[:10]:
                correct_10 += 1
            if gold.lower() in predictions_list[:5]:
                correct_5 += 1
            if gold.lower() == predictions_list[0]:
                correct += 1
            total += 1
            count += 1

print(correct/total, correct_5/total, correct_10/total, correct_20/total)
