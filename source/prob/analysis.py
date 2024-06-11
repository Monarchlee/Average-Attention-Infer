from tqdm import tqdm
import pickle
import json
import torch
from datasets import load_dataset
import numpy as np
import argparse
import pdb
from ..utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Samsoup/cosmos_qa')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--test_ratio', type=float, default=1, help='the test cases ratio in the dataset')
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--shot_num', type=int, default=0)
    parser.add_argument('--shot_path', type=str, default="../prompt/cosmos_5shot.json")
    parser.add_argument('--forward_path', type=str,
                        default="../results/batch_forward_output/mmlu_preorder_validation_origin_0shot_results_0_299_nosymb_0temp.pkl",
                        help="the forward file path for prob_infer or text_infer")
    parser.add_argument('--cmp_path', type=str,
                        default="../results/pkl_output/mmlu_preorder_validation_origin_llama2_13b_chat_0shot_results_0_570_nosymb_0temp.pkl",
                        help="the compared result only enable when debias method is not No")
    parser.add_argument('--test_name', type=str, default='test')
    args = parser.parse_args()
    shot_num = args.shot_num
    r_seed = 256

    # load forward
    with open(args.forward_path, 'r') as f:
        prob_res = json.load(f)

    # load_dataset
    if args.data_path is not None:
        file_type = args.data_path.split('.')[-1]
        file_path = args.data_path
        test_data = (load_dataset(file_type, data_files={'validation': file_path}))['validation']
    else:
        test_data = load_dataset(args.dataset, split='train') # load from huggingface

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


    if len(args.cmp_path) > 0:
        try:
            with open(args.cmp_path, 'rb') as f:
                cmp_file = pickle.load(f)
        except:
            raise LookupError('Load failed.')
    else:
        cmp_file = None

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

    test_name = args.test_name

    question_answer_pairs = []
    begin = 0
    end = len(test_data)
    choice_num = 4  # for cosmos qa
    # debias param, debias which layer

    # only analysis here, for a given forward file
    origin_acc = 0
    expand_acc = 0
    invalid_num = 0

    # about the distribution of permutation inviaraint(same answer after permutation)
    permutation_dist = [0] * 24
    acc_dist = [0] * 25
    top_k = [0] * 4  # the average prob of top_k answers sorted in descented order
    # all_prob_dist = [] # the overall prob dist of test cases
    batch_size = 24
    count = 0
    batch_ans = []
    batch_oldans = []
    batch_choices = []
    batch_questions = []
    batch_attention = []
    batch_prob = []
    batch_pos = []
    total_diff_count = 0
    diff_class = [0] * 3  # for T->F F->F F->T
    # only passed the content in batch format to the model, but still return only one result once?

    data_len = len(permutation_data['question_expand'])

    # debias set, list of idx of attention_bias, for full_permutation, this should be range(batch_size)
    debias_set = range(batch_size)  # the full set

    for count in tqdm(range(1, data_len + 1, batch_size), desc="Processing"):
        diff_count = 0
        debiased_probs = torch.tensor(prob_res['results'][count-1:count-1+batch_size]) # should be [bsz, 4] tensor
        if cmp_file is not None:  # just load the former res
            old_probs = np.array([cmp_file[idx]['probs'] for idx in range(count-1,count-1+batch_size)])  # should be a list of [bsz, 4]
        else:
            raise NotImplementedError('no cmp_file provided')

        question_txt = permutation_data['question_expand'][count - 1]
        batch_oldans = torch.argmax(torch.tensor(old_probs), dim=1).tolist()  # should be list of [bsz]

        # origin in batch end, should be modified here
        batch_ans_order = torch.argmax(debiased_probs, dim=1)  # this ans are in the order of original choice
        batch_ans = []
        original_choice = permutation_data['choices'][count - 1]
        for bt in range(batch_size):
            idx = count - 1 + bt
            answer_num = permutation_data['choices'][idx].index(original_choice[batch_ans_order[bt]])
            batch_ans.append(answer_num)
            question_answer_pair = {}
            question_answer_pair['question'] = permutation_data['question_expand'][idx]
            question_answer_pair['choices'] = permutation_data['choices'][idx]
            question_answer_pair['answer'] = answer_num
            question_answer_pair['probs'] = debiased_probs[bt].tolist()
            question_answer_pair['gt_ans'] = permutation_data['choices'][idx].index(
                permutation_data['gt_ans'][count - 1])
            question_answer_pairs.append(question_answer_pair)

        # the count-1 should be the last one
        batch_choices = permutation_data['choices'][count - 1:count - 1 + batch_size]
        first_idx = count - 1
        answers = batch_ans
        num_ans = answers
        first_answer = permutation_data['choices'][first_idx][num_ans[0]]  # the first answer text
        gt_answer = permutation_data['gt_ans'][count - 1]  # all gt_answers in one batch are the same

        # analysis and save batch
        print(f"Idx:{round(count / batch_size)}\n" + permutation_data['question_expand'][first_idx] + "\n")

        for x in range(batch_size):
            idx = count - 1 + x
            if answers[x] != batch_oldans[x]:
                diff_count += 1
                total_diff_count += 1
                old_ans = permutation_data['choices'][idx][batch_oldans[x]]
                ans = permutation_data['choices'][idx][answers[x]]
                if old_ans == gt_answer:  # original ans is T and the ans is F
                    diff_class[0] += 1
                else:  # origin is F, the ans can be T or F
                    if ans == gt_answer:
                        diff_class[2] += 1
                    else:
                        diff_class[1] += 1
        print(f"\nAnswer: {batch_choices[0][answers[0]]}")
        # num_ans = filter_ans(answers, batch_choices)  # this should be [batch_size]

        acc_num = 0
        if first_answer == gt_answer:
            origin_acc += 1  # the original answer is true
            expand_acc += 1
            acc_num += 1
        same_num = 0
        print(f"Old answers in vector:{batch_oldans}\nDiffer number: {diff_count}\n")
        print(f"All answers in vector:{num_ans}\nAvg_differ number: {total_diff_count / ((count-1) / batch_size + 1)}")
        # according to the order of first question's choice and will be sorted in descending order in the end
        ans_dict = {f"{c}": 0 for c in permutation_data['choices'][count - 1]}
        ans_dict[batch_choices[0][num_ans[0]]] += 1
        for j in range(1, batch_size):
            if num_ans[j] == -1:  # invalid answer
                invalid_num += 1
                continue
            if first_answer == batch_choices[j][num_ans[j]]:
                same_num += 1
            if gt_answer == batch_choices[j][num_ans[j]]:
                expand_acc += 1
                acc_num += 1
            ans_dict[batch_choices[j][num_ans[j]]] += 1
        num_vec = sorted(list(ans_dict.values()), reverse=True)
        choice_num = len(permutation_data['choices'][0])
        if len(num_vec)<choice_num: # this happends when there are 2 or more choices are the same
            num_vec.extend([0]*(choice_num-len(num_vec)))
        top_k = np.sum([top_k, [x / batch_size for x in num_vec]], axis=0).tolist()
        # pdb.set_trace()
        permutation_dist[same_num] += 1
        acc_dist[acc_num] += 1

        print(
            f"-------Analysis-------origin_acc:{origin_acc / ((count-1)/24 + 1)}-------total_acc:{expand_acc / (count-1+batch_size)}------same_ratio:{same_num / 23}")
        # if count % 2400 == 0:
        # pdb.set_trace()
        batch_ans = []
        batch_oldans = []
        batch_choices = []
        batch_questions = []
        batch_attention = []
        batch_prob = []
        batch_pos = []

    res_path = "../results/batch_pkl_output/{}_preorder_validation_{}_results_{}_{}_nosymb_prob.pkl".format(data_name,
                                                                                                            test_name,
                                                                                                            begin,
                                                                                                            end)
    with open(res_path, 'wb') as f:
        pickle.dump(question_answer_pairs, f)
    diff_class_ratio = [k / total_diff_count for k in diff_class]
    final_res = {}
    final_res['original_acc'] = origin_acc / (end - begin)
    final_res['expand_acc'] = expand_acc / ((end - begin) * batch_size)
    final_res['invalid_ratio'] = invalid_num / ((end - begin) * batch_size)
    final_res['permutation_dist'] = [p / (end - begin) for p in permutation_dist]
    final_res['acc_dist'] = [p / (end - begin) for p in acc_dist]
    final_res['expect_pmnum'] = sum([final_res['permutation_dist'][i] * i for i in range(len(permutation_dist))])
    final_res['expect_accnum'] = sum([final_res['acc_dist'][i] * i for i in range(len(acc_dist))])
    final_res['top_k'] = [k / (end - begin) for k in top_k]
    final_res['avg_differ'] = total_diff_count / (end - begin)
    final_res['diff_class_ratio'] = {'T->F': diff_class_ratio[0],
                                     'F->F': diff_class_ratio[1],
                                     'F->T': diff_class_ratio[2]}
    final_res['shot_text'] = few_shot_text

    final_path = f"../results/batch_p_output/{data_name}_preorder_probchoice_{test_name}_analysis_nosymb.json"
    # pdb.set_trace()
    with open(final_path, 'w') as f:
        json.dump(final_res, f)

