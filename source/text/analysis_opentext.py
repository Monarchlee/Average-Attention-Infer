from tqdm import tqdm
from difflib import SequenceMatcher
import pickle
import json
import torch
from datasets import load_dataset
import numpy as np
import argparse
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
    parser.add_argument('--dataset', type=str, default='L4m3r/hotpotqa_dev_distractor_permut_4_500')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--test_ratio', type=float, default=1, help='the test cases ratio in the dataset')
    parser.add_argument('--shot_num', type=int, default=0)
    parser.add_argument('--shot_path', type=str, default="../prompt/cosmos_5shot.json")
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--forward_path', type=str,
                        default="../results/batch_forward_output/mmlu_preorder_validation_origin_0shot_results_0_299_nosymb_0temp.pkl",
                        help="the forward file path for prob_infer or text_infer")
    parser.add_argument('--test_name', type=str, default='test')
    args = parser.parse_args()
    shot_num = args.shot_num
    r_seed = 256

    # load forward
    with open(args.forward_path, 'r') as f:
        text_res = json.load(f)

    # load_dataset
    if args.data_path is not None:
        file_type = args.data_path.split('.')[-1]
        file_path = args.data_path
        test_data = (load_dataset(file_type, data_files={'validation': file_path}))['validation']
    else:
        test_data = load_dataset(args.dataset, split='train')

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


    if "hotpot" in args.dataset:
        data_name = "hotpotqa"
    else:
        raise NotImplementedError


    permutation_data = test_data

    test_name = args.test_name

    question_answer_pairs = []
    batch_size = 24
    begin = 0
    end = len(test_data)/batch_size
    choice_num = 4  # for cosmos qa
    # debias param, debias which layer

    # only analysis here, for a given forward file
    origin_acc = 0
    expand_acc = 0
    potential_case = 0
    pi_case = 0

    data_len = len(permutation_data['question_expand'])

    # debias set, list of idx of attention_bias, for full_permutation, this should be range(batch_size)
    debias_set = range(batch_size)  # the full set

    for count in tqdm(range(1, data_len + 1, batch_size), desc="Processing"):
        
        answer_text_batch = text_res['results'][count-1:count-1+batch_size] # should be [bsz] answer list

        batch_text_ans = answer_text_batch
        ans_tf_batch = filter_tf(batch_text_ans, permutation_data['gt_ans'][count-1:count-1+batch_size])
        num_ans = ans_tf_batch
        for bt in range(batch_size):
            idx = count - 1 + bt
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


    res_path = "../results/batch_pkl_output/{}_preorder_validation_{}_text_results_{}_{}_nosymb_0temp.pkl".format(data_name,
                                                                                                            test_name,
                                                                                                            begin,
                                                                                                            end)
    with open(res_path, 'wb') as f:
        pickle.dump(question_answer_pairs, f)

    final_res = {}
    final_res['original_acc'] = origin_acc / (data_len/batch_size)
    final_res['expand_acc'] = expand_acc / (data_len)
    final_res['PPIR'] = pi_case/(potential_case + 1)

    final_path = f"../results/batch_text_output/{data_name}_preorder_textchoice_validation_{test_name}_analysis_nosymb_0temp.json"
    # pdb.set_trace()
    with open(final_path, 'w') as f:
        json.dump(final_res, f)

