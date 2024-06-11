from tqdm import tqdm
import json
import torch
from transformers import AutoTokenizer, LlamaTokenizerFast, Qwen2TokenizerFast, AutoModelForCausalLM
from datasets import load_dataset
from peft import PeftConfig
import argparse
import random
import pdb

from ..utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../models/Llama-2/Llama-2-13b-chat-hf')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--if_lora', type=int, default=0)
    parser.add_argument('--shot_num', type=int, default=0)
    parser.add_argument('--shot_path', type=str, default="../prompt/cosmos_5shot.json")
    parser.add_argument('--if_softmax', type=int, default=1)  # if enabling softmax in average attention
    parser.add_argument('--dataset', type=str, default='L4m3r/hotpotqa_dev_distractor_permut_4_500')
    parser.add_argument('--debias', type=str, default='avg_att_batch_infer', help="debias method")
    parser.add_argument('--debias_layers', nargs='*', type=int, help="debias which layers, should be 0~39 for llama2")
    parser.add_argument('--debias_set', type=int, default=24, help="the set used to avg the att")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--if_single', type=int, default=1,
                        help="if debias with attention every single sample of permutations")
    parser.add_argument('--test_ratio', type=float, default=1, help='the test cases ratio in the dataset')
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--pos_path', type=str,
                        default="../results/batch_pos_output/mmlu_preorder_llama2_13b_chat_pos_debug_nosymb.json", help="the positional file path")
    parser.add_argument('--adapter_path', type=str,
                        default='../results/openbookqa/ft_biloss_8_1_lora_2e-3_40_16_expand8/checkpoint-81000')
    parser.add_argument('--test_name', type=str, default='ft_biloss_8_1_lora_2e-3_40_16_expand8_81000')
    args = parser.parse_args()
    # for debug
    print(f'Test on {args.debias_layers[0]} to {args.debias_layers[1]} layers\n Test name: {args.test_name}')

    debias_method = args.debias
    enable_lora = args.if_lora
    shot_num = args.shot_num
    r_seed = 256
    pos_path = args.pos_path
    with open(pos_path, 'r') as f:
        all_pos_permute = json.load(f)
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


    if "cosmos" in args.dataset:
        data_name = "cosmosqa"
    elif "openbookqa" in args.dataset:
        data_name = "openbookqa"
    elif "mmlu" in args.dataset:
        data_name = "mmlu"
    elif "hotpot" in args.dataset:
        data_name = "hotpotqa"
    else:
        raise NotImplementedError


    # map into qa format and expand them into full permutation(N!)
    map_func = decide_func(args.dataset, few_shot=few_shot_text)
    # pdb.set_trace()
    if data_name == "hotpotqa":
        permutation_data = test_data
    else:
        permutation_data = test_data.map(map_func, batched=True, remove_columns=test_data.column_names)
    # the data key should be ['question_expand', 'choices', 'gt_ans']
    # pdb.set_trace()
    model_path = args.model
    device = torch.device(f"cuda:{args.device}")

    # pdb.set_trace()
    model_gpu = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).to(
        device)
    if "Llama" in args.model:
        tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
    elif "Qwen" in args.model:
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # tokenizer.padding_side = "right"
    # pdb.set_trace()
    if enable_lora == 1:
        adapter_path = args.adapter_path
        peft_config = PeftConfig.from_pretrained(adapter_path)
        peft_config.init_lora_weights = False
        model_gpu.add_adapter(peft_config)
        model_gpu.enable_adapters()
    test_name = args.test_name
    batch_size = 24
    begin = 0
    end = len(test_data)/batch_size
    choice_num = 4  # for cosmos qa
    # debias param, debias which layer

    layer_vec_interval = args.debias_layers
    layer_vec = torch.LongTensor(range(layer_vec_interval[0], layer_vec_interval[1])).to(device)

    
    data_len = len(permutation_data['question_expand'])

    # debias set, list of idx of attention_bias, for full_permutation, this should be range(batch_size)
    debias_set = range(batch_size)  # the full set
    debiased_set_num = args.debias_set
    if debiased_set_num == 24:
        set_mask = list(range(batch_size))
    elif debiased_set_num == choice_num:  # the circular permutation
        set_mask = [0, 9, 16, 23]
    else:  # random choose num-1 other permutation
        set_mask = [0]
        if debiased_set_num > 1:
            new_set = random.sample(range(batch_size)[1:], debiased_set_num - 1)
            set_mask = set_mask + new_set
            set_mask.sort()
    all_answer = []
    # use pipeline maybe good here, however, the input not only the input ids in our test script
    
    for count in tqdm(range(1, data_len + 1, batch_size), desc="Processing"):
        batch_questions = permutation_data['question_expand'][count-1:count-1+batch_size]
        batch_input_ts = tokenizer(batch_questions, return_tensors="pt").input_ids.to(device) # should be [bt_size, token_len]
        batch_pos_ts = torch.tensor(all_pos_permute['option_pos'][count-1:count-1+batch_size]).to(device)
        batch_permute_ts = torch.tensor(all_pos_permute['permute'][count-1:count-1+batch_size]).to(device)  # [bt_size, n]

        with torch.no_grad():
            output = model_gpu.generate(
                input_ids=batch_input_ts, do_sample=False, max_new_tokens=32,
                option_pos=batch_pos_ts, permute=batch_permute_ts,
                debiased_layer=layer_vec, batch_mask=torch.LongTensor(set_mask)
            )
        q_len = len(permutation_data['question_expand'][count - 1])
        answer_full_text_batch = tokenizer.batch_decode(output, skip_special_tokens=True)
        answer_text_batch = [ans.split('\nAnswer: ',1)[1] for ans in answer_full_text_batch]
        all_answer.extend(answer_text_batch)

    final_path = "../results/batch_forward_output/{}_preorder_validation_{}_results_{}_{}_nosymb_text.json".format(data_name,
                                                                                                            test_name,
                                                                                                            begin,
                                                                                                            end)
    final_res = {"results": all_answer}
    with open(final_path, 'w') as f:
        json.dump(final_res, f)

