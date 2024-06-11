import functools
from difflib import SequenceMatcher
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

def get_questions(question, choices, dataset="mmlu", field="", few_shot="", if_symbol=0):
    if dataset=="cosmosqa":
        prompt_str = "Choose the correct option to the question according to the passage." #for cosmosqa
    elif dataset=="mmlu":
        # for mmlu
        prompt_str = f"The following are multiple choice questions about {field}. You should directly answer the question by choosing the correct option."
    else:
        raise NotImplementedError('Dataset is not supported now.')
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

def get_questions_hotpot(question, choices, field="", few_shot="", if_symbol=0):
    # for hotpot QA
    prompt_str = f"The following are facts and the question. You should answer the question according to the facts directly."
    fact_text = [fc[0]+'\n'+''.join(fc[1]) for fc in choices]
    all_choices = [list(l) for l in permute(fact_text)]
    questions = []
    num = len(choices)
    char_list = generate_char_list(num)
    filter_choices = []
    for cs in all_choices:
        if len(cs) != 4:
            continue
        choices_str = "Facts: \n"
        for i in range(num):
            if if_symbol == 1:
                choices_str += char_list[i] + "." + cs[i] + '\n'
            else:
                choices_str += cs[i] + '\n'
        questions.append(prompt_str + '\n' + few_shot + '\n' + choices_str + question + '\n' + 'Answer: ')
        filter_choices.append(cs)
    return questions, filter_choices



def decide_func(dataset, few_shot="", avg_att=False):
    if 'cosmos_qa' in dataset:
        if len(few_shot) > 1:
            return functools.partial(cosmos_map, few_shot=few_shot, avg_att=avg_att)
        else:
            return cosmos_map
    elif 'mmlu' in dataset:
        if len(few_shot) > 1:
            return functools.partial(mmlu_map, few_shot=few_shot, avg_att=avg_att)
        else:
            return mmlu_map
    elif 'hotpot' in dataset:
        if len(few_shot) > 1:
            return functools.partial(hotpotqa_map, few_shot=few_shot, avg_att=avg_att)
        else:
            return hotpotqa_map
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
        questions, choices = get_questions(ctx_q, choice, dataset="cosmosqa", few_shot=few_shot)
        gt_answer = [examples[f"answer{examples['label'][k]}"][k]] * len(questions)
        gt_answers.extend(gt_answer)
        q_outputs.extend(questions)
        choice_outputs.extend(choices)
    return {'question_expand': q_outputs, 'choices': choice_outputs, 'gt_ans': gt_answers}


def few_shot_map(examples):
    res = ""
    for example in examples:
        choices_text = ''.join([example[f"answer{i}"] + '\n' for i in range(4)])
        gt_ans = example[f"answer{example['label']}"]
        text_exp = f"Passage: {example['context']}\nQuestion: {example['question']}\nOptions:\n{choices_text}Answer: {gt_ans}\n"
        res += text_exp
    return res


def mmlu_map(examples, few_shot="", avg_att=False):
    q_outputs = []
    choice_outputs = []
    gt_answers = []
    # pdb.set_trace()
    for k in range(len(examples['label'])):
        ctx_q = f"Question: {examples['question'][k]}"
        choice = [examples[f"answer{i}"][k] for i in range(4)]
        questions, choices = get_questions(ctx_q, choice, dataset="mmlu", field=examples["field"][k], few_shot=few_shot)
        gt_answer = [examples[f"answer{examples['label'][k]}"][k]] * len(questions)
        gt_answers.extend(gt_answer)
        q_outputs.extend(questions)
        choice_outputs.extend(choices)
    return {'question_expand': q_outputs, 'choices': choice_outputs, 'gt_ans': gt_answers}

def hotpotqa_map(examples, few_shot=""):
    q_outputs = []
    choice_outputs = []
    gt_answers = []
    sp_facts = []
    for k in range(len(examples)):
        ctx_q = f"Question: {examples[k]['question']}"
        fact = examples[k]["context"]
        questions, facts = get_questions_hotpot(ctx_q, fact, few_shot=few_shot)
        gt_answer = [examples[k]["answer"]] * len(questions)
        sp_fact = [examples[k]["supporting_facts"]] * len(questions)
        sp_facts.extend(sp_fact)
        gt_answers.extend(gt_answer)
        q_outputs.extend(questions)
        choice_outputs.extend(facts)
    return {'question_expand': q_outputs, 'choices': choice_outputs, 'gt_ans': gt_answers, 'supporting_facts': sp_facts}



def filter_ans_nosymb(ans, choices):
    # just for filtering the choice from the answer list, 0,1,2,3 as result
    res = -1
    idx = 0
    for cs in choices:
        if cs in ans:
            res = idx
            break
        idx += 1
    if res == -1:
        sim_vec = [SequenceMatcher(None, ans.lower(), choice.lower()).ratio() for choice in choices]
        max_v = max(sim_vec)
        if max_v > 0.3:  # valid answer
            res = sim_vec.index(max_v)
    return res