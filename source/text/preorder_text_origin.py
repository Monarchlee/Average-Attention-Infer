import pyarrow.parquet as pq
from difflib import SequenceMatcher
from tqdm import tqdm
import pickle
import json
import re
# from LLM_conv import Conversation
import torch
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline, AutoTokenizer, AutoModel, StoppingCriteria, StoppingCriteriaList, \
    AutoModelForCausalLM, PretrainedConfig
from datasets import load_dataset
from peft import PeftConfig
import numpy as np
import argparse
from torch import randperm
import functools
import pdb
from ..utils import *


def filter_tf(ans_pred, gt_ans):
    label = []
    for i in range(len(ans_pred)):
        l_p = ans_pred[i].lower()
        l_g = gt_ans[i].lower()
        if l_g in l_p:
            ans = 1
        elif SequenceMatcher(None, l_p, l_g).ratio() > 0.5:
            ans = 1
        else:
            ans = 0
        label.append(ans)
    return label



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../models/Llama-2/Llama-2-13b-chat-hf')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--shot_num', type=int, default=0)
    parser.add_argument('--shot_path', type=str, default="../prompt/cosmos_5shot.json")
    parser.add_argument('--dataset', type=str, default='L4m3r/hotpotqa_dev_distractor_permut_4_500')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--test_ratio', type=float, default=1, help='the test cases ratio in the dataset')
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--test_name', type=str, default='origin')

    args = parser.parse_args()
    shot_num = args.shot_num

    # load_dataset
    if args.data_path is not None:
        file_type = args.data_path.split('.')[-1]
        file_path = args.data_path
        test_data = load_dataset(file_type, data_files={'validation': file_path})
    else:
        test_data = load_dataset(args.dataset, split='train')

    if shot_num > 0 and len(args.shot_path)<2:
        few_shot, _ = torch.utils.data.random_split(test_data, [shot_num, len(test_data)-shot_num])
        few_shot_text = few_shot_map(few_shot)
    elif shot_num > 0 and len(args.shot_path)>=2:
        # load few_shot text from json
        with open(args.shot_path, 'r') as f:
            few_shot_text = json.load(f)['few_shot']
    else:
        few_shot_text = ""

    

    if "hotpot" in args.dataset:
        data_name = "hotpotqa"
    else:
        raise NotImplementedError


    permutation_data = test_data
    model_path = args.model
    device = torch.device(f"cuda:{args.device}")

    model_gpu = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # pdb.set_trace()
    # build the pipeline
    # qa_pipe = pipeline(task="text-generation", model=model_gpu, tokenizer=tokenizer, device=args.device)
    # qa_pipe.tokenizer.pad_token_id = tokenizer.eos_token_id
    test_name = args.test_name

    question_answer_pairs = []
    question_answer_pairs_old = []

    begin = 0
    end = len(test_data)
    choice_num = 4 # for cosmos qa
    # debias param, debias which layer

    # just generate, test should be done after this
    origin_acc = 0
    expand_acc = 0

    batch_size = 24
    data_len = len(permutation_data['question_expand'])

    # debias set, list of idx of attention_bias, for full_permutation, this should be range(batch_size)
    debias_set = range(batch_size) # the full set
    potential_case = 0
    pi_case = 0

    for count in tqdm(range(1, data_len + 1, batch_size), desc="Processing"):
        diff_count = 0
        batch_old_text_ans = []
        batch_text_ans = []
        num_ans = []

        input_id_batch = tokenizer(permutation_data['question_expand'][count-1:count-1+batch_size], return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            output = model_gpu.generate(
                input_ids=input_id_batch, do_sample=False, max_new_tokens=32, 
            )
        ans_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        ans_text = [ans.split('\nAnswer:',1)[1] for ans in ans_text]
        ans_tf_batch = filter_tf(ans_text, permutation_data['gt_ans'][count-1:count-1+batch_size])
        batch_text_ans = ans_text
        num_ans = ans_tf_batch
        for bt in range(batch_size):
            idx = count - batch_size + bt
            question_answer_pair = {}
            question_answer_pair['question'] = permutation_data['question_expand'][idx]
            question_answer_pair['facts'] = permutation_data['choices'][idx]
            question_answer_pair['label'] = num_ans[bt]
            question_answer_pair['answer_text'] = batch_text_ans[bt]
            question_answer_pair['gt_ans'] = permutation_data['gt_ans'][idx]
            question_answer_pairs.append(question_answer_pair)
        right_num = sum(num_ans)
        origin_acc += num_ans[0]
        expand_acc += right_num
        if right_num > 0:
            potential_case += 1
        if right_num==batch_size:
            pi_case += 1
        print(f"result: {num_ans}, PPIR_batch: {sum(num_ans)/batch_size}")



    res_path = "../results/pkl_output/{}_preorder_validation_{}_text_results_{}_{}_nosymb_0temp.pkl".format(data_name, test_name, begin,
                                                                                            end)
    with open(res_path, 'wb') as f:
        pickle.dump(question_answer_pairs, f)

    final_res = {}
    final_res['original_acc'] = origin_acc / (data_len/batch_size)
    final_res['expand_acc'] = expand_acc / (data_len)
    final_res['PPIR'] = pi_case/(potential_case + 1)

    final_path = f"../results/text_output/{data_name}_preorder_textchoice_validation_{test_name}_analysis_nosymb_0temp.json"
    # pdb.set_trace()
    with open(final_path, 'w') as f:
        json.dump(final_res, f)

