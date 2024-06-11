from tqdm import tqdm
import pickle
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, LlamaTokenizerFast, Qwen2TokenizerFast
from datasets import load_dataset
from peft import PeftConfig
import argparse
import random
import pdb
from ..utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../models/Llama-2/Llama-2-13b-chat-hf')
    # parser.add_argument('--model', type=str, default='../models/Qwen-14b-chat-hf')
    # parser.add_argument('--model', type=str, default='../models/ChatGLM3-6b')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--if_lora', type=int, default=0)
    parser.add_argument('--shot_num', type=int, default=0)
    parser.add_argument('--shot_path', type=str, default="../prompt/cosmos_5shot.json")
    parser.add_argument('--dataset', type=str, default='Samsoup/cosmos_qa')
    parser.add_argument('--debias', type=str, default='avg_att_infer_batch', help="debias method")
    parser.add_argument('--debias_layers', nargs='*', type=int, help="debias which layers, should be 0~39 for llama2")
    parser.add_argument('--debias_set', type=int, default=24, help="number of samples in permutation set")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--if_single', type=int, default=1,
                        help="if debias with attention every single sample of permutations")
    parser.add_argument('--test_ratio', type=float, default=1, help='the test cases ratio in the dataset')
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--adapter_path', type=str,
                        default='../results/openbookqa/ft_biloss_8_1_lora_2e-3_40_16_expand8/checkpoint-81000')
    parser.add_argument('--pos_path', type=str,
                        default="../results/batch_pos_output/cosmosqa_preorder_llama2_13b_chat_nosymb.json", help="the positional file path")
    parser.add_argument('--test_name', type=str, default='ft_biloss_8_1_lora_2e-3_40_16_expand8_81000')
    # the batch size should be set carefully as the full loss will be calculated as n! if there are n choices

    # parser.add_argument('--epoch', type=int, default=20)
    # parser.add_argument('--sample_ratio', type=float, default=1.0)
    # will sample from the total permutation questions with sample_ratio when calculating the loss
    # parser.add_argument('--save_dir', type=str, default='../results/fine-tuning/Llama-2')

    args = parser.parse_args()
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
    permutation_data = test_data.map(map_func, batched=True, remove_columns=test_data.column_names)
    model_path = args.model
    device = torch.device(f"cuda:{args.device}")
    model_gpu = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).to(
        device)
    if "Llama" in args.model:
        tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
    elif "Qwen" in args.model:
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if enable_lora == 1:
        adapter_path = args.adapter_path
        peft_config = PeftConfig.from_pretrained(adapter_path)
        peft_config.init_lora_weights = False
        model_gpu.add_adapter(peft_config)
        model_gpu.enable_adapters()
    test_name = args.test_name


    begin = 0
    end = len(test_data)
    choice_num = 4  # for cosmos qa
    # debias param, debias which layer

    layer_vec_interval = args.debias_layers
    layer_vec = range(layer_vec_interval[0], layer_vec_interval[1])
    # layer_vec = [4,12,14,15,16,17,18,21,38,40] # the top-10 of first 300
    batch_size = 24
    count = 0
    batch_questions = []
    batch_attention = []
    batch_prob = []
    batch_pos = []
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
    forward_res = []
    for count in tqdm(range(1, data_len + 1, batch_size), desc="Processing"):
        diff_count = 0
        prob_dict = {}
        batch_pos = []
        batch_permute = []
        losses = []
        batch_pos_ts = torch.tensor(all_pos_permute['option_pos'][count-1:count-1+batch_size]).to(device)
        batch_permute_ts = torch.tensor(all_pos_permute['permute'][count-1:count-1+batch_size]).to(device)  # [bt_size, n]
        for c_text in permutation_data['choices'][count - 1]:
            prob_dict[c_text] = 0
            option_batch_ids = []
            labels_batch = []
            for id_bias in debias_set:
                idx = count - 1 + id_bias
                # ignore the sentencepiece token, only works when model is llama2,otherwise will take a white space in to answer labels
                prefix_input_ids = tokenizer(permutation_data['question_expand'][idx], truncation=False,
                                                return_tensors="pt").input_ids[:, :-1]

                option_batch_id = tokenizer(permutation_data['question_expand'][idx] + c_text,
                                            return_tensors="pt").input_ids
                option_batch_ids.append(option_batch_id)
                labels = option_batch_id.clone()
                labels[:, :prefix_input_ids.size(1)] = -100  # ignore the question and calculate only the answer
                labels_batch.append(labels)

            labels_batch_ts = torch.stack(labels_batch).squeeze().to(device)  # [bt_size, seq_len]
            option_batch_ids_ts = torch.stack(option_batch_ids).squeeze().to(
                device)  # should be [batch_size, seq_len]
            # test code
            #tst_text = tokenizer.decode(option_batch_ids_ts[0,batch_pos_ts[0,0,0]:batch_pos_ts[0,0,1]])
            #pdb.set_trace()
            
            with torch.no_grad():
                loss_op = model_gpu(
                    input_ids=option_batch_ids_ts, labels=labels_batch_ts, option_pos=batch_pos_ts,
                    permute=batch_permute_ts, average_loss=False, debiased_layer=torch.LongTensor(layer_vec),
                    batch_mask=torch.LongTensor(set_mask)
                ).loss.detach().cpu()
            # the loss should be in [bt_size]
            losses.append(loss_op)
        losses_ts = torch.stack(losses).t().float()
        #losses_ts = torch.tensor(torch.stack(losses).t(), dtype=torch.float32)  # this can be [bt_size,n]
        nll = - losses_ts
        probs = torch.exp(torch.sub(nll, torch.max(nll, dim=1, keepdim=True).values))
        debiased_probs = torch.div(probs,
                                    (probs.sum(dim=1, keepdim=True) + 1e-10))  # the debiased probs [bsz, n]
        forward_res.append(debiased_probs)
        # notice that the debiased porbs are in the order of origin option order!!!!!
    choice_num = forward_res[0].shape[-1]
    p_res_ts = torch.stack(forward_res).view(-1, choice_num) # should be[bsz*b_num, n]

    res_path = "../results/batch_forward_output/{}_preorder_validation_{}_results_{}_{}_nosymb_prob.json".format(data_name,
                                                                                                            test_name,
                                                                                                            begin,
                                                                                                            end)
    final_res = {"results": p_res_ts.tolist()}
    with open(res_path, 'w') as f:
        json.dump(final_res, f)

