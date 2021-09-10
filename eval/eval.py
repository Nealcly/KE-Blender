import json

from metrics import bleu_metric, knowledge_metric, f_one, normalize_answer, bleu

from nltk.corpus import stopwords

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
                data.append([tmp_source, utt])
                # add split '\n' for blender
                tmp_source = tmp_source + "\t" + utt
            else:
                tmp_source = utt
    return data


stop_words = set(stopwords.words('english'))


def move_stop_words(str):
    item = " ".join([w for w in str.split() if not w.lower() in stop_words])
    return item


def detokenize(tk_str):
    tk_list = tk_str.strip().split()
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return " ".join(r_list)



if __name__ == "__main__":
    test_data = read_wizard_json('wizard_of_wikipedia/test_topic_split.json')
    # test_df = pd.DataFrame(test_data, columns=["input_text", "target_text"])
    # test_data = [i[0] for i in test_data]
    gold = [i[1] for i in test_data]

    with open('blender_baseline_3epoch.txt', 'r') as f:
        decode_result = f.readlines()
    assert len(gold) == len(decode_result)
    b1, b2, b3 = bleu_metric(gold, decode_result)

    print("b1:{},b2:{},b3:{}".format(round(b1, 4), round(b2, 4), round(b3, 4)))

    res = f_one(gold, decode_result)
    print('f1:{}'.format(round(res[0], 4)))

