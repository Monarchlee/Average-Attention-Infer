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


def similar(a, b, ratio=0.9):
    seq_matcher = SequenceMatcher(None, a, b)

    similarity_ratio = seq_matcher.ratio()

    return similarity_ratio > ratio


def permute(nums):
    result = []
    _permute(nums, 0, result)
    return result


def _permute(nums, start, result):
    if start == len(nums):
        result.append(nums.copy())
        return

    for i in range(start, len(nums)):
        # 交换当前元素与第一个元素
        nums[start], nums[i] = nums[i], nums[start]

        # 递归生成子问题的全排列
        _permute(nums, start + 1, result)

        # 恢复交换，回溯到上一状态
        nums[start], nums[i] = nums[i], nums[start]


def generate_char_list(num):
    # 使用列表推导生成对应的字符列表
    char_list = [chr(ord('A') + i) for i in range(num)]
    return char_list


def filter_ans(ans, choices):
    # just for filtering the choice index A,B,C,D from the answer list, 0,1,2,3 as result
    res = []
    for i in range(len(ans)):
        match_res = re.findall('[ABCD][^a-zA-Z]+', ans[
            i])  # there still a bug such as answer: "The  Answer is A", which means we can't get the $ symbol

        if len(match_res) > 0:
            num_r = ord(match_res[0][0]) - 65
            res.append(num_r)
        else:
            # nosymbol here
            num_r = filter_ans_nosymb([ans[i]], [choices[i]])[0]
            res.append(num_r)
    return res


def filter_ans_nosymb(ans, choices):
    # just for filtering the choice index A,B,C,D from the answer list, 0,1,2,3 as result
    res = []
    for i in range(len(ans)):
        num_r = -1
        ans_now = ans[i].split('\n')[0]
        # check if the answer is close enough to the choice if the answer is not in symbol type
        sim_vec = [SequenceMatcher(None, ans_now.lower(), choice.lower()).ratio() for choice in choices[i]]
        max_v = max(sim_vec)
        if max_v > 0.3:  # valid answer
            num_r = sim_vec.index(max_v)
        else:  # may the answer is too long, containing the answer
            num_candi = sim_vec.index(max_v)
            if choices[i][num_candi].lower() in ans_now.lower():
                num_r = num_candi

        # -1 invalid answer
        res.append(num_r)
    return res

def get_questions(question, choices, field="", few_shot="", if_symbol=0):
    #prompt_str = "Choose the correct option to the question according to the passage." #for cosmosqa
    # for mmlu
    prompt_str = f"The following are multiple choice questions about {field}. You should directly answer the question by choosing the correct option."
    all_choices = [list(l) for l in permute(choices)]
    questions = []
    num = len(choices)
    char_list = generate_char_list(num)
    for cs in all_choices:
        choices_str = "Options: \n"
        for i in range(num):
            if if_symbol == 1:
                choices_str += char_list[i] + "." + cs[i] + '\n'
            else:
                choices_str += cs[i] + '\n'
        questions.append(prompt_str + '\n' + few_shot + '\n' + question + '\n' + choices_str + 'Answer: ')
    return questions, all_choices


def decide_func(dataset, few_shot="", avg_att=False):
    if 'cosmos_qa' in dataset:
        if len(few_shot)>1:
            return functools.partial(cosmos_map, few_shot=few_shot, avg_att=avg_att)
        else:
            return cosmos_map
    elif 'mmlu' in dataset:
        if len(few_shot)>1:
            return functools.partial(mmlu_map, few_shot=few_shot, avg_att=avg_att)
        else:
            return mmlu_map
    else:
        raise ValueError('No implementation, please insert the map function here.')


def cosmos_map(examples, few_shot="", avg_att=False):
    q_outputs = []
    choice_outputs = []
    gt_answers = []
    # pdb.set_trace()
    for k in range(len(examples['label'])):
        ctx_q = f"Passage: {examples['context'][k]}\nQuestion: {examples['question'][k]}"
        choice = [examples[f"answer{i}"][k] for i in range(4)]
        questions, choices = get_questions(ctx_q, choice, few_shot=few_shot)
        gt_answer = [examples[f"answer{examples['label'][k]}"][k]] * len(questions)
        gt_answers.extend(gt_answer)
        q_outputs.extend(questions)
        choice_outputs.extend(choices)
    return {'question_expand': q_outputs, 'choices': choice_outputs, 'gt_ans': gt_answers}

def few_shot_map(examples):
    res = ""
    for example in examples:
        choices_text = ''.join([example[f"answer{i}"]+'\n' for i in range(4)])
        gt_ans = example[f"answer{example['label']}"]
        text_exp = f"Passage: {example['context']}\nQuestion: {example['question']}\nOptions:\n{choices_text}Answer: {gt_ans}\n"
        res += text_exp
    return res


def mmlu_map(examples, few_shot="",avg_att=False):
    q_outputs = []
    choice_outputs = []
    gt_answers = []
    # pdb.set_trace()
    for k in range(len(examples['label'])):
        ctx_q = f"Question: {examples['question'][k]}"
        choice = [examples[f"answer{i}"][k] for i in range(4)]
        questions, choices = get_questions(ctx_q, choice, field=examples["field"][k], few_shot=few_shot)
        gt_answer = [examples[f"answer{examples['label'][k]}"][k]] * len(questions)
        gt_answers.extend(gt_answer)
        q_outputs.extend(questions)
        choice_outputs.extend(choices)
    return {'question_expand': q_outputs, 'choices': choice_outputs, 'gt_ans': gt_answers}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../models/Llama-2/Llama-2-13b-chat-hf')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--if_lora', type=int, default=1)
    parser.add_argument('--shot_num', type=int, default=0)
    parser.add_argument('--if_avgatt', type=int, default=0, help="if enbale the avg_att in inference")
    parser.add_argument('--shot_path', type=str, default="../prompt/cosmos_5shot.json")
    parser.add_argument('--if_softmax', type=int, default=1)  # if enabling softmax in average attention
    parser.add_argument('--dataset', type=str, default='Samsoup/cosmos_qa')
    parser.add_argument('--debias', type=str, default='No', help="debias method")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--if_single', type=int, default=1, help="if debias with attention every single sample of permutations")
    parser.add_argument('--test_ratio', type=float, default=1, help='the test cases ratio in the dataset')
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--cmp_path', type=str, default="../results/pkl_output/cosmosqa_preorder_validation_origin_5shot_results_0_299_nosymb_0temp.pkl",
                        help="the compared result only enable when debias method is not No")
    parser.add_argument('--adapter_path', type=str,
                        default='../results/openbookqa/ft_biloss_8_1_lora_2e-3_40_16_expand8/checkpoint-81000')
    parser.add_argument('--test_name', type=str, default='ft_biloss_8_1_lora_2e-3_40_16_expand8_81000')
    # the batch size should be set carefully as the full loss will be calculated as n! if there are n choices

    # parser.add_argument('--epoch', type=int, default=20)
    # parser.add_argument('--sample_ratio', type=float, default=1.0)
    # will sample from the total permutation questions with sample_ratio when calculating the loss
    # parser.add_argument('--save_dir', type=str, default='../results/fine-tuning/Llama-2')

    args = parser.parse_args()
    debias_method = args.debias
    enable_lora = args.if_lora
    shot_num = args.shot_num
    r_seed = 256


    # load_dataset
    if args.data_path is not None:
        file_type = args.data_path.split('.')[-1]
        file_path = args.data_path
        test_data = load_dataset(file_type, data_files={'validation': file_path})
    else:
        test_data = load_dataset(args.dataset, split='validation')

    if shot_num > 0 and len(args.shot_path)<2:
        few_shot, _ = torch.utils.data.random_split(test_data, [shot_num, len(test_data)-shot_num])
        few_shot_text = few_shot_map(few_shot)
    elif shot_num > 0 and len(args.shot_path)>=2:
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
    else:
        test_data = test_data['validation']

    if len(args.cmp_path)>0:
        try:
            with open(args.cmp_path, 'rb') as f:
                cmp_file = pickle.load(f)
        except:
            raise LookupError('Load failed.')
    else:
        cmp_file = None
    #pdb.set_trace()

    if "cosmos" in args.dataset:
        data_name = "cosmosqa"
    elif "openbookqa" in args.dataset:
        data_name = "openbookqa"
    elif "mmlu" in args.dataset:
        data_name = "mmlu"
    else:
        raise NotImplementedError

    # map into qa format and expand them into full permutation(N!)
    avg_att = bool(args.if_avgatt)
    map_func = decide_func(args.dataset, few_shot=few_shot_text, avg_att=avg_att)
    permutation_data = test_data.map(map_func, batched=True, remove_columns=test_data.column_names)
    # the data key should be ['question_expand', 'choices', 'gt_ans']
    model_path = args.model
    device = torch.device(f"cuda:{args.device}")

    
    #pdb.set_trace()
    model_gpu = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    #tokenizer.padding_side = "right"
    #pdb.set_trace()
    if enable_lora == 1:
        adapter_path = args.adapter_path
        peft_config = PeftConfig.from_pretrained(adapter_path)
        peft_config.init_lora_weights = False
        model_gpu.add_adapter(peft_config)
        model_gpu.enable_adapters()
    # pdb.set_trace()
    # build the pipeline
    # qa_pipe = pipeline(task="text-generation", model=model_gpu, tokenizer=tokenizer, device=args.device)
    # qa_pipe.tokenizer.pad_token_id = tokenizer.eos_token_id
    test_name = args.test_name

    question_answer_pairs = []
    begin = 0
    end = len(test_data)

    # just generate, test should be done after this
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
    total_diff_count = 0
    # only passed the content in batch format to the model, but still return only one result once?

    # config using in generation
    # generate_dict = {
    #    "do_sample": False,
    #    "max_new_tokens": 16
    # }
    # gen_config = PretrainedConfig.from_dict(generate_dict)
    # pdb.set_trace()
    # pbar = tqdm(total=batch_size * (end - begin))
    data_len = len(permutation_data['question_expand'])

    for count in tqdm(range(1, data_len + 1), desc="Processing"):

        # get the prob ans in MCP format but not CP format
        if cmp_file is not None: # just load the former res
            probs = np.array(cmp_file[count-1]['probs'])
        else: # infer again
            losses = []
            lengths = []
            prefix_input_ids = tokenizer(permutation_data['question_expand'][count - 1], truncation=False,
                                        return_tensors="pt").input_ids[:,:-1] 
            # ignore the sentencepiece token, actually only works for llama2 tokenizer, while others will take
            # a blank space into account as answer label.

            for option in permutation_data['choices'][count - 1]:
                prefix_and_option_text = permutation_data['question_expand'][count - 1] + option.strip()
                input_ids = tokenizer(prefix_and_option_text, truncation=False, return_tensors="pt").input_ids.to(device)
                lengths.append(input_ids.size(1) - prefix_input_ids.size(1))

                labels = input_ids.clone()
                labels[:, :prefix_input_ids.size(1)] = -100  # ignore the question and calculate only the answer

                #input_ids = input_ids[..., -1536:]
                #labels = labels[..., -1536:]
                

                with torch.no_grad():
                    loss = model_gpu(
                        input_ids=input_ids,
                        labels=labels,
                    ).loss.detach().to(torch.float32).cpu().item()
                losses.append(loss)
                # pdb.set_trace()

            # pdb.set_trace()
            nll = - np.array(losses,dtype=np.float32)
            probs = np.exp(nll - np.max(nll))
            probs = probs / (probs.sum() + 1e-10)  # the obs probs,
            # pdb.set_trace()

        if debias_method == "mask_choice": # this debiased method should calculate every sample even if the permutations
            token_num = 5  # avg the first token_num tokens' attention
            #layer_num = 40  # total layer number, different on different LLM
            layer_num = 1 # only take the first layer into account
            # debias the position permutation with masked option
            pad_text = 'mask' # the <unk> maybe not that good here
            #option_len = [tokenizer(pad_text + '\n', return_tensors="pt").input_ids.shape[1]] - 1  # -1 for bos_token
            pad_nums = [len(choice.split(' ')) for choice in permutation_data['choices'][count - 1]]
            masked_choices = [' '.join([pad_text]*c_n) for c_n in pad_nums] # get the corresponding num of pad_token in word level
            
            # pdb.set_trace()
            choice_len = len(''.join(permutation_data['choices'][count - 1])) + 12  # 12 is the constent of cosmosqa, representing 'answer'
            question_txt = permutation_data['question_expand'][count - 1][:-choice_len]
            question_token_len = tokenizer(question_txt, return_tensors="pt").input_ids.to(device).shape[1]
            choice_num = len(permutation_data['choices'][count - 1])
            for i in range(choice_num):
                question_txt += masked_choices[i] + '\n'
            question_txt += 'Answer: '
            input_ids_att = tokenizer(question_txt, return_tensors="pt").input_ids.to(device)

            #pdb.set_trace()
            #input_cpu = input_ids.cpu()
            with torch.no_grad():
                output_att = model_gpu.generate(
                    input_ids=input_ids_att, do_sample=False, max_new_tokens=32, return_dict_in_generate=True,
                    output_attentions=True
                )
            #pdb.set_trace()
            attention = output_att.attentions
            option_att = []
            start_pos = 0
            end_pos = question_token_len
            for k in range(choice_num):
                score = 0
                start_pos = end_pos
                end_pos = end_pos + pad_nums[k] + 1 # 1 for \n constent
                #pdb.set_trace()
                for v in range(min(token_num, len(attention))):
                    for l in range(layer_num):
                        score += torch.sum(attention[v][l][0, :, -1, start_pos:end_pos])
                option_att.append(score)
            #pdb.set_trace()
            op_att_tensor = torch.tensor(option_att).to(device)
            if args.if_softmax == 1:
                option_att = torch.softmax((op_att_tensor / (torch.sum(op_att_tensor) + 1e-10)), dim=0).cpu()
            else:
                option_att = (op_att_tensor / (torch.sum(op_att_tensor) + 1e-10)).cpu()
            # constent to avoid divide by 0 error
            # normalization and softmax, on GPU fro fp16 data
            
            # pdb.set_trace()
        if (count - 1) % batch_size == 0:  # the start of new batch, should calculate the prior on position first
            diff_count = 0
            if debias_method == "mask_avg_attention":
                token_num = 5  # avg the first token_num tokens' attention
                #layer_num = 40  # total layer number, different on different LLM
                layer_num = 1 # only take the first layer into account
                # debias the position permutation with masked option
                #pad_text = '<unk>'
                pad_text = 'mask'
                option_len = tokenizer(pad_text + '\n', return_tensors="pt").input_ids.shape[1] - 1  # -1 for bos_token
                #pdb.set_trace()
                choice_len = len(''.join(permutation_data['choices'][count - 1])) + 12  # 12 is the constent of cosmosqa
                question_txt = permutation_data['question_expand'][count - 1][:-choice_len]
                choice_num = len(permutation_data['choices'][count - 1])
                for i in range(choice_num):
                    question_txt += pad_text + '\n'
                question_txt += 'Answer: '
                input_ids_att = tokenizer(question_txt, return_tensors="pt").input_ids.to(device)
                #pdb.set_trace()
                #input_cpu = input_ids.cpu()
                with torch.no_grad():
                    output_att = model_gpu.generate(
                        input_ids=input_ids_att, do_sample=False, max_new_tokens=32, return_dict_in_generate=True,
                        output_attentions=True
                    )
                #pdb.set_trace()
                attention = output_att.attentions
                option_att = []
                for k in range(choice_num):
                    score = 0
                    for v in range(min(token_num,len(attention))): # sometimes the length of generated output may less than token_num settings
                        for l in range(layer_num):
                            score += torch.sum(attention[v][l][0, :, -1, -(3 + option_len * (choice_num - k) + v):-(
                                        3 + option_len * (choice_num - k) + v) + option_len])
                    option_att.append(score)
                #pdb.set_trace()
                op_att_tensor = torch.tensor(option_att).to(device)

                if args.if_softmax == 1:
                    option_att = torch.softmax((op_att_tensor / (torch.sum(op_att_tensor) + 1e-10)), dim=0).cpu()
                else:
                    option_att = (op_att_tensor / (torch.sum(op_att_tensor) + 1e-10)).cpu()
                # constent to avoid divide by 0 error
                # normalization and softmax, on GPU fro fp16 data

                # pdb.set_trace()
            elif debias_method == "mask_choice":
                option_att = option_att
            elif debias_method == "No":
                option_att = torch.ones(probs.shape)
            else:
                raise NotImplementedError

        debiased_prob = probs / option_att
        answer_num = torch.argmax(debiased_prob).item()

        #pdb.set_trace()
        batch_ans.append(answer_num)
        #if cmp_file is not None:
        #    batch_oldans.append(cmp_file[count-1]['answer'])
        #else:
        batch_oldans.append(torch.argmax(torch.tensor(probs)).item())

        batch_prob.append(probs)
        batch_attention.append(option_att)
        # pdb.set_trace()
        question_answer_pair = {}
        question_answer_pair['question'] = permutation_data['question_expand'][count - 1]
        question_answer_pair['choices'] = permutation_data['choices'][count - 1]
        question_answer_pair['answer'] = answer_num
        question_answer_pair['probs'] = probs.tolist()
        question_answer_pair['gt_ans'] = permutation_data['choices'][count - 1].index(permutation_data['gt_ans'][count - 1])

        question_answer_pairs.append(question_answer_pair)
        if count % batch_size == 0:
            # the count-1 should be the last one
            batch_choices = permutation_data['choices'][count - batch_size:count]
            first_idx = count - batch_size
            # analysis and save batch
            print(f"Idx:{round(count / batch_size)}\n" + permutation_data['question_expand'][first_idx] + "\n")
            if debias_method=="mask_choice" and args.if_single==0:
                avg_attention = torch.stack(batch_attention).mean(dim=0)
                answers = [torch.argmax(batch_prob[x]/avg_attention).item() for x in range(batch_size)]
            else:
                answers = batch_ans

            for x in range(batch_size):
                if answers[x] != batch_oldans[x]:
                    diff_count += 1
                    total_diff_count += 1
            print(f"\nAnswer: {batch_choices[0][answers[0]]}")
            # num_ans = filter_ans(answers, batch_choices)  # this should be [batch_size]
            num_ans = answers

            first_answer = permutation_data['choices'][first_idx][num_ans[0]]  # the first answer text
            gt_answer = permutation_data['gt_ans'][count - 1]  # all gt_answers in one batch are the same

            acc_num = 0
            if first_answer==gt_answer:
                origin_acc += 1  # the original answer is true
                expand_acc += 1
                acc_num += 1
            same_num = 0
            print(f"Old answers in vector:{batch_oldans}\nDiffer number: {diff_count}\n")
            print(f"All answers in vector:{num_ans}\nAvg_differ number: {total_diff_count/(count/batch_size)}")
            # according to the order of first question's choice and will be sorted in descending order in the end
            ans_dict = {f"{c}": 0 for c in permutation_data['choices'][count - 1]}
            ans_dict[batch_choices[0][num_ans[0]]] += 1
            for j in range(1, batch_size):
                if num_ans[j] == -1:  # invalid answer
                    invalid_num += 1
                    continue
                if first_answer==batch_choices[j][num_ans[j]]:
                    same_num += 1
                if gt_answer==batch_choices[j][num_ans[j]]:
                    expand_acc += 1
                    acc_num += 1
                ans_dict[batch_choices[j][num_ans[j]]] += 1
            num_vec = sorted(list(ans_dict.values()), reverse=True)
            choice_num = len(permutation_data['choices'][0])
            if len(num_vec)<choice_num: # this happends when there are 2 or more choices are the same
                num_vec.extend([0]*(choice_num-len(num_vec)))
            top_k = np.sum([top_k, [x / batch_size for x in num_vec]], axis=0).tolist()
            permutation_dist[same_num] += 1
            acc_dist[acc_num] += 1

            print(
                f"-------Analysis-------origin_acc:{origin_acc / (round(count / 24))}-------total_acc:{expand_acc / count}------same_ratio:{same_num / 23}")
            # if count % 2400 == 0:
            # pdb.set_trace()
            batch_ans = []
            batch_oldans = []
            batch_choices = []
            batch_questions = []
            batch_attention = []
            batch_prob = []


    res_path = "../results/pkl_output/{}_preorder_validation_{}_results_{}_{}_nosymb_0temp.pkl".format(data_name, test_name, begin,
                                                                                            end)
    with open(res_path, 'wb') as f:
        pickle.dump(question_answer_pairs, f)

    final_res = {}
    final_res['original_acc'] = origin_acc / (end - begin)
    final_res['expand_acc'] = expand_acc / ((end - begin) * batch_size)
    final_res['invalid_ratio'] = invalid_num / ((end - begin) * batch_size)
    final_res['permutation_dist'] = [p / (end - begin) for p in permutation_dist]
    final_res['acc_dist'] = [p / (end - begin) for p in acc_dist]
    final_res['expect_pmnum'] = sum([final_res['permutation_dist'][i] * i for i in range(len(permutation_dist))])
    final_res['expect_accnum'] = sum([final_res['acc_dist'][i] * i for i in range(len(acc_dist))])
    final_res['top_k'] = [k / (end - begin) for k in top_k]
    final_res['avg_differ'] = total_diff_count/(end - begin)
    final_res['shot_text'] = few_shot_text

    final_path = f"../results/p_output/{data_name}_preorder_probchoice_validation_{test_name}_analysis_nosymb_0temp.json"
    # pdb.set_trace()
    with open(final_path, 'w') as f:
        json.dump(final_res, f)

