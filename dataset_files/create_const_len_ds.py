#%%
'''
Create subsets of the datasets with constant length of the input and output
'''

import json
import os
from transformers import AutoTokenizer

model_name = 'gpt2'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

main_data_path = os.path.dirname(os.path.realpath(__file__))
subset_folder = 'abstractive'
folder_input_ds = os.path.join(main_data_path, subset_folder)
folder_out = os.path.join(main_data_path, subset_folder + '_const_len')
# %%
ds_names = os.listdir(folder_input_ds)
for ds_name in ds_names:
    input_ds = os.path.join(folder_input_ds, ds_name)
    with open(input_ds, 'r') as f:
        ds = json.load(f)


    # for answer_len in [1, 2]:  # possible but not used
    for answer_len in [1]:
        examples_by_len = {}
        ignored_examples = []
        for i in range(len(ds)):
            try:
                curr_target = ds[i]['output']
                tmp_target = curr_target
                if ' ' != tmp_target[-1]:
                    tmp_target = ' ' + tmp_target
                tokenized_output_len = len(tokenizer(tmp_target).input_ids)
                if tokenized_output_len != answer_len:
                    ignored_examples.append(i)
                    continue
            except:
                print(f'Error for example {i}')
                continue
            
            curr_sentence = tokenizer.pad_token + 'Q: ' + ds[i]['input'] + '\nA: ' + curr_target
            
            tokenized_input_len = len(tokenizer(curr_sentence).input_ids)
            if tokenized_input_len not in examples_by_len:
                examples_by_len[tokenized_input_len] = []
            examples_by_len[tokenized_input_len].append({"input": ds[i]['input'], "output": curr_target})
            if i % 1000 == 0:
                print(f'Example {i}: "{curr_sentence}", len: {tokenized_input_len}')

        print(f'Original dataset length: {len(ds)}')
        print(f'Ignored examples: {len(ignored_examples)}')
        for sentence_len, examples in examples_by_len.items():
            print(f'For length: {sentence_len} we have {len(examples)} examples')
            if len(examples) < 100:
                print(f'Not enough examples for length {sentence_len}, skipping')
            else:
                curr_path_out = os.path.join(folder_out, ds_name.split('.')[0] + '_len_' + str(sentence_len)+ '_' + str(answer_len) + '.json')
                with open(curr_path_out, 'w') as f:
                    json.dump(examples, f)


# %%
# filter out the datasets that are not used
tmp = [x for x in os.listdir(folder_out) if '.json' in x]
tmp = [x for x in tmp if 'ag_news' not in x and 'commonsense_qa' not in x]
for ds in tmp:
    print(f'"{ds.split(".json")[0]}"', end=' ')
# %%
