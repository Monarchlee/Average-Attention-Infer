from tqdm import tqdm
import json
import torch
from transformers import AutoTokenizer, LlamaTokenizerFast, Qwen2TokenizerFast
from datasets import load_dataset
import argparse
import functools
import pdb

from ..utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../models/Llama-2/Llama-2-13b-chat-hf')
    # parser.add_argument('--model', type=str, default='../models/Qwen-14b-chat-hf')
    # parser.add_argument('--model', type=str, default='../models/ChatGLM3-6b')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--shot_num', type=int, default=0)
    parser.add_argument('--shot_path', type=str, default="../prompt/cosmos_5shot.json")
    parser.add_argument('--dataset', type=str, default='Samsoup/cosmos_qa')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--test_ratio', type=float, default=1, help='the test cases ratio in the dataset')
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--test_name', type=str, default='ft_biloss_8_1_lora_2e-3_40_16_expand8_81000')

    args = parser.parse_args()
    shot_num = args.shot_num
    r_seed = 256
    # load_dataset
    if args.data_path is not None:
        file_type = args.data_path.split('.')[-1]
        file_path = args.data_path
        test_data = (load_dataset(file_type, data_files={'validation': file_path}))['validation']
    else:
        test_data = load_dataset(args.dataset, split='validation')

    if shot_num > 0 and len(args.shot_path) < 2:
        few_shot, _ = torch.utils.data.random_split(test_data, [shot_num, len(test_data) - shot_num])
        few_shot_text = few_shot_map(few_shot)
    elif shot_num > 0 and len(args.shot_path) >= 2:
        # load few_shot text from json
        with open(args.shot_path, 'r') as f:
            few_shot_text = json.load(f)['few_shot']
    else:
        few_shot_text = ""

    if args.test_ratio < 1:
        if args.shuffle == 1:
            shuffled_dataset = test_data.shuffle(seed=r_seed)  # for seed reproduction
        else:
            shuffled_dataset = test_data
        test_data = shuffled_dataset.train_test_split(test_size=args.test_ratio, shuffle=False)['test']

    if "cosmos" in args.dataset:
        data_name = "cosmosqa"
    elif "openbookqa" in args.dataset:
        data_name = "openbookqa"
    elif "mmlu" in args.dataset:
        data_name = "mmlu"
    else:
        raise NotImplementedError

    # map into qa format and expand them into full permutation(N!)
    map_func = decide_func(args.dataset, few_shot=few_shot_text)
    # pdb.set_trace()
    permutation_data = test_data.map(map_func, batched=True, remove_columns=test_data.column_names)
    # the data key should be ['question_expand', 'choices', 'gt_ans']
    # pdb.set_trace()
    model_path = args.model
    device = torch.device(f"cuda:{args.device}")
    # prepare the tokenizer
    if "Llama" in args.model:
        tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
    elif "Qwen" in args.model:
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    test_name = args.test_name
    begin = 0
    end = len(test_data)
    choice_num = len(permutation_data['choices'][0])  # for cosmos qa, mmlu, openbookqa
    batch_size = 24
    count = 0
    data_len = len(permutation_data['question_expand'])
    final_pos = []
    final_perm = []
    for count in tqdm(range(1, data_len + 1), desc="Processing"):
        if (count - 1) % batch_size == 0:  # the start of new batch, should calculate the prior on position first
            batch_pos = []
            batch_permute = []
            origin_choice = [test_data[int((count - 1) / batch_size)][f"answer{i}"] for i in range(choice_num)]
            # this should be set differently according to tokenizers
            if 'Llama' in model_path:
                token_option_len = [tokenizer('\n' + op, return_tensors="pt").input_ids.shape[1] - 2
                                    for op in permutation_data['choices'][count-1]]
            elif 'Qwen' in model_path:  # for qwen
                token_option_len = [tokenizer(op + '\n', return_tensors="pt").input_ids.to(device).shape[1] for
                                    op in permutation_data['choices'][count-1]]
            elif 'GLM' in model_path:  # chatGLM
                token_option_len = [
                    tokenizer('\n' + op, return_tensors="pt").input_ids.to(device).shape[1] - 4 + 1 for op in
                    permutation_data['choices'][count-1]]
            else:
                raise NotImplementedError('Set the pos_option according to the tokenizer.')
            
            choice_len = len('\n'.join(permutation_data['choices'][count-1]) + "\nAnswer: ")
            question_txt = permutation_data['question_expand'][count-1][:-choice_len]
            question_token_len = tokenizer(question_txt, return_tensors="pt").input_ids.shape[1]

            for id_bias in range(batch_size):
                idx = count - 1 + id_bias
                choice_num = len(permutation_data['choices'][idx])
                begin_option = question_token_len
                end_option = 0
                option_pos = []
                option_att = []
                permute_single = []
                for i in range(len(permutation_data['choices'][idx])):
                    idx_in_origin = origin_choice.index(permutation_data['choices'][idx][i])
                    end_option = begin_option + token_option_len[idx_in_origin]
                    permute_single.append(idx_in_origin)
                    option_pos.append([begin_option, end_option])
                    begin_option = end_option
                batch_permute.append(torch.LongTensor(permute_single))
                option_pos = torch.LongTensor(option_pos)  # this should be N*2 for begin and end of each option's tokens
                batch_pos.append(option_pos)
                # test code
                #all_token = tokenizer(permutation_data['question_expand'][idx], return_tensors="pt").input_ids
                #pdb.set_trace()

            batch_pos_ts = torch.stack(batch_pos) # [bt_size, n, 2]
            batch_permute_ts = torch.stack(batch_permute)  # [bt_size, n]
            final_pos.append(batch_pos_ts)
            final_perm.append(batch_permute_ts)

    final_pos = torch.stack(final_pos).view(-1, choice_num, 2)  # should be [exp_data_len, n, 2]
    final_perm = torch.stack(final_perm).view(-1, choice_num)  # should be [exp_data_len, n]
    final_res = {'option_pos': final_pos.tolist(), 'permute': final_perm.tolist()}
    final_path = f"../results/batch_pos_output/{data_name}_preorder_{test_name}_nosymb.json"
    with open(final_path, 'w') as f:
        json.dump(final_res, f)

