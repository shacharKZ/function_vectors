#%%
import json
import os
from transformers import AutoTokenizer

model_name = 'gpt2'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

main_data_path = os.path.dirname(os.path.realpath(__file__))
path_original_ds_linear_relations_paper = 'TODO'  # TODO: please provide the path to the dataset from the paper
subset_folder = 'factual'
folder_input_ds = os.path.join(path_original_ds_linear_relations_paper, subset_folder)
folder_out = os.path.join(main_data_path, subset_folder)

# %%
ds_names = os.listdir(folder_input_ds)
for ds_name in ds_names:
    input_ds = os.path.join(folder_input_ds, ds_name)
    with open(input_ds, 'r') as f:
        ds = json.load(f)

    prompt_format = ds['prompt_templates'][0]
    print(prompt_format)
    
    examples = []
    for i in range(len(ds['samples'])):
        curr_target = ds['samples'][i]['object']
        tmp_target = curr_target
        if ' ' != tmp_target[-1]:
            tmp_target = ' ' + tmp_target

        if '\\' in curr_target:
            print('WARNING!!!')
            print(f'Seems like the target is not tokenized: {curr_target}')
        curr_input = ds['samples'][i]['subject']
        if '\\' in curr_input:
            print('WARNING!!!')
            print(f'Seems like the input is not tokenized: {curr_input}')
        
        # curr_sentence = tokenizer.pad_token + 'Q: ' + prompt_format.format(ds['samples'][i]['subject']) + '\nA: ' + curr_target
        examples.append({"input": ds['samples'][i]['subject'], "output": curr_target})

    curr_path_out = os.path.join(folder_out, ds_name.split('.')[0] + '.json')
    with open(curr_path_out, 'w') as f:
        json.dump(examples, f)



# %%
tmp = [x for x in os.listdir(folder_out) if '.json' in x]
for ds in tmp:
    print(f'"{ds.split(".json")[0]}"', end=' ')
# %%
