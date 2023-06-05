#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import os
import shutil
import copy
import pickle
import datetime
import torch
import numpy as np
import scipy.stats as stats

from rouge import Rouge

from .logger import *

import sys
sys.setrecursionlimit(10000)

_ROUGE_PATH = "/remote-home/dqwang/ROUGE/RELEASE-1.5.5"
_PYROUGE_TEMP_FILE = "/remote-home/dqwang/"


REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}", 
        "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'} 

CEFR2INT = {
    0: 0,
    'A1': 1,
    'A2': 2,
    'B1': 3,
    'B2': 4,
    'C': 5,
}
INT2CEFR = {v: k for k, v in CEFR2INT.items()}

BERT2ABB = {
    'bert-base-uncased': 'bert',
    'roberta-base': 'roberta',
    'xlm-roberta-base': 'xlm',
    'distilroberta-base': 'distil',
    'allenai/longformer-base-4096': 'longformer',
    'sentence-transformers/all-mpnet-base-v2': 'mpnet',
    'databricks/dolly-v2-12b': 'dolly'
}

def clean(x): 
    x = x.lower()
    return re.sub( 
            r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''", 
            lambda m: REMAP.get(m.group()), x)

def rouge_eval(hyps, refer):
    rouge = Rouge()
    try:
        score = rouge.get_scores(hyps, refer)[0]
        mean_score = np.mean([score["rouge-1"]["f"], score["rouge-2"]["f"], score["rouge-l"]["f"]])
    except:
        mean_score = 0.0
    return mean_score

def rouge_all(hyps, refer):
    rouge = Rouge()
    score = rouge.get_scores(hyps, refer)[0]
    return score

def eval_label(match_true, pred, true, total, match):
    match_true, pred, true, match = match_true.float(), pred.float(), true.float(), match.float()
    try:
        accu = match / total
        precision = match_true / pred
        recall = match_true / true
        F = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        accu, precision, recall, F = 0.0, 0.0, 0.0, 0.0
        logger.error("[Error] float division by zero")
    return accu, precision, recall, F


def pyrouge_score(hyps, refer, remap = True):
    return pyrouge_score_all([hyps], [refer], remap)
    
def pyrouge_score_all(hyps_list, refer_list, remap = True):
    from pyrouge import Rouge155
    nowTime=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    PYROUGE_ROOT = os.path.join(_PYROUGE_TEMP_FILE, nowTime)
    SYSTEM_PATH = os.path.join(PYROUGE_ROOT,'result')
    MODEL_PATH = os.path.join(PYROUGE_ROOT,'gold')
    if os.path.exists(SYSTEM_PATH):
        shutil.rmtree(SYSTEM_PATH)
    os.makedirs(SYSTEM_PATH)
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)
    os.makedirs(MODEL_PATH)

    assert len(hyps_list) == len(refer_list)
    for i in range(len(hyps_list)):
        system_file = os.path.join(SYSTEM_PATH, 'Model.%d.txt' % i)
        model_file = os.path.join(MODEL_PATH, 'Reference.A.%d.txt' % i)

        refer = clean(refer_list[i]) if remap else refer_list[i]
        hyps = clean(hyps_list[i]) if remap else hyps_list[i]

        with open(system_file, 'wb') as f:
            f.write(hyps.encode('utf-8'))
        with open(model_file, 'wb') as f:
            f.write(refer.encode('utf-8'))

    r = Rouge155(_ROUGE_PATH)

    r.system_dir = SYSTEM_PATH
    r.model_dir = MODEL_PATH
    r.system_filename_pattern = 'Model.(\d+).txt'
    r.model_filename_pattern = 'Reference.[A-Z].#ID#.txt'

    try:
        output = r.convert_and_evaluate(rouge_args="-e %s -a -m -n 2 -d" % os.path.join(_ROUGE_PATH, "data"))
        output_dict = r.output_to_dict(output)
    finally:
        logger.error("[ERROR] Error stop, delete PYROUGE_ROOT...")
        shutil.rmtree(PYROUGE_ROOT)

    scores = {}
    scores['rouge-1'], scores['rouge-2'], scores['rouge-l'] = {}, {}, {}
    scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f'] = output_dict['rouge_1_precision'], output_dict['rouge_1_recall'], output_dict['rouge_1_f_score']
    scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f'] = output_dict['rouge_2_precision'], output_dict['rouge_2_recall'], output_dict['rouge_2_f_score']
    scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f'] = output_dict['rouge_l_precision'], output_dict['rouge_l_recall'], output_dict['rouge_l_f_score']
    return scores


def pyrouge_score_all_multi(hyps_list, refer_list, remap = True):
    from pyrouge import Rouge155
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    PYROUGE_ROOT = os.path.join(_PYROUGE_TEMP_FILE, nowTime)
    SYSTEM_PATH = os.path.join(PYROUGE_ROOT, 'result')
    MODEL_PATH = os.path.join(PYROUGE_ROOT, 'gold')
    if os.path.exists(SYSTEM_PATH):
        shutil.rmtree(SYSTEM_PATH)
    os.makedirs(SYSTEM_PATH)
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)
    os.makedirs(MODEL_PATH)

    assert len(hyps_list) == len(refer_list)
    for i in range(len(hyps_list)):
        system_file = os.path.join(SYSTEM_PATH, 'Model.%d.txt' % i)
        # model_file = os.path.join(MODEL_PATH, 'Reference.A.%d.txt' % i)

        hyps = clean(hyps_list[i]) if remap else hyps_list[i]
        
        with open(system_file, 'wb') as f:
            f.write(hyps.encode('utf-8'))

        referType = ["A", "B", "C", "D", "E", "F", "G"]
        for j in range(len(refer_list[i])):
            model_file = os.path.join(MODEL_PATH, "Reference.%s.%d.txt" % (referType[j], i))
            refer = clean(refer_list[i][j]) if remap else refer_list[i][j]
            with open(model_file, 'wb') as f:
                f.write(refer.encode('utf-8'))

    r = Rouge155(_ROUGE_PATH)

    r.system_dir = SYSTEM_PATH
    r.model_dir = MODEL_PATH
    r.system_filename_pattern = 'Model.(\d+).txt'
    r.model_filename_pattern = 'Reference.[A-Z].#ID#.txt'

    output = r.convert_and_evaluate(rouge_args="-e %s -a -m -n 2 -d" % os.path.join(_ROUGE_PATH, "data"))
    output_dict = r.output_to_dict(output)

    shutil.rmtree(PYROUGE_ROOT)

    scores = {}
    scores['rouge-1'], scores['rouge-2'], scores['rouge-l'] = {}, {}, {}
    scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f'] = output_dict['rouge_1_precision'], output_dict['rouge_1_recall'], output_dict['rouge_1_f_score']
    scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f'] = output_dict['rouge_2_precision'], output_dict['rouge_2_recall'], output_dict['rouge_2_f_score']
    scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f'] = output_dict['rouge_l_precision'], output_dict['rouge_l_recall'], output_dict['rouge_l_f_score']
    return scores


def cal_label(article, abstract):
    hyps_list = article

    refer = abstract
    scores = []
    for hyps in hyps_list:
        mean_score = rouge_eval(hyps, refer)
        scores.append(mean_score)

    selected = []
    selected.append(int(np.argmax(scores)))
    selected_sent_cnt = 1

    best_rouge = np.max(scores)
    while selected_sent_cnt < len(hyps_list):
        cur_max_rouge = 0.0
        cur_max_idx = -1
        for i in range(len(hyps_list)):
            if i not in selected:
                temp = copy.deepcopy(selected)
                temp.append(i)
                hyps = "\n".join([hyps_list[idx] for idx in np.sort(temp)])
                cur_rouge = rouge_eval(hyps, refer)
                if cur_rouge > cur_max_rouge:
                    cur_max_rouge = cur_rouge
                    cur_max_idx = i
        if cur_max_rouge != 0.0 and cur_max_rouge >= best_rouge:
            selected.append(cur_max_idx)
            selected_sent_cnt += 1
            best_rouge = cur_max_rouge
        else:
            break

    return selected


def accuracy(preds: torch.Tensor,
             labels: torch.Tensor) -> float:
    correct = (preds == labels).double()
    correct = correct.sum().item()
    return correct / len(labels)


def correlation(output: torch.Tensor,
                labels: torch.Tensor,
                mask: torch.Tensor,
                mode: str,
                conversion: str = "max") -> float:
    if mode == "regression":
        preds = output.squeeze()
    elif mode == "classification":
        # Convert classification to real values
        betas = output.new_tensor([cefr_to_beta(cefr) for cefr in CEFR_LEVELS])
        if conversion == "max":
            pred_idx = torch.argmax(output, dim=1)
            preds = torch.gather(betas, 0, pred_idx)
        else:
            preds = F.softmax(output, dim=1)
            preds = torch.sum(preds * betas, dim=1)


    preds = preds[mask.nonzero()].cpu()
    labels = labels[mask.nonzero()].cpu()
    rho, _ = stats.spearmanr(preds.detach().numpy(), labels.numpy())
    return rho


def compute_micro_overestimate_rate(preds: torch.Tensor,
                                    labels: torch.Tensor) -> float:
    return torch.sum((preds > labels).double()).item() / len(preds) * 100
    
def compute_macro_overestimate_rate(preds: torch.Tensor,
                                    labels: torch.Tensor) -> float:
    unique_classes = torch.unique(labels)
    num_classes = len(unique_classes)
    score_rate = 0.
    
    for c in unique_classes:
        indices = torch.where(labels == c)
        score_predictions = preds[indices]
        score_targets = labels[indices]
        score_rate += 1 / num_classes * compute_micro_overestimate_rate(score_predictions, score_targets)

    return score_rate

def compute_micro_underestimate_rate(preds: torch.Tensor,
                              labels: torch.Tensor) -> float:
    return torch.sum((preds < labels).double()).item() / len(preds) * 100
    
def compute_macro_underestimate_rate(preds: torch.Tensor,
                              labels: torch.Tensor) -> float:
    unique_classes = torch.unique(labels)
    num_classes = len(unique_classes)
    score_rate = 0.
    
    for c in unique_classes:
        indices = torch.where(labels == c)
        score_predictions = preds[indices]
        score_targets = labels[indices]
        score_rate += 1 / num_classes * compute_micro_underestimate_rate(score_predictions, score_targets)

    return score_rate


def _compute_over_estimate_rate(score_predictions, score_target):
    margin = 0.5
    return torch.sum(
        torch.where(
            (score_predictions - score_target) > margin,
            torch.ones(len(score_predictions)),
            torch.zeros(len(score_predictions)))).item() / len(score_predictions) * 100

def _compute_under_estimate_rate(score_predictions, score_target):
    margin = -0.5
    return torch.sum(
        torch.where(
            (score_predictions - score_target) < margin,
            torch.ones(len(score_predictions)),
            torch.zeros(len(score_predictions)))).item() / len(score_predictions) * 100

def _accuracy_within_margin(score_predictions, score_target, margin):
    """ Returns the percentage of predicted scores that are within the provided margin from the target score. """
    return torch.sum(
        torch.where(
            torch.abs(score_predictions - score_target) <= margin,
            torch.ones(len(score_predictions)),
            torch.zeros(len(score_predictions)))).item() / len(score_predictions) * 100

# See: https://github.com/teinhonglo/automated-english-transcription-grader/blob/master/helpers/metrics.py
# See: https://github.com/teinhonglo/automated-english-transcription-grader/blob/master/local/metrics_cpu.py
def _compute_rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()

def _compute_mcrmse(all_score_predictions, all_score_targets):
    # print(all_score_predictions.shape, all_score_targets.shape)
    unique_classes = torch.unique(all_score_targets)
    num_classes = len(unique_classes)
    score_rmse = 0.
    
    for c in unique_classes:
        indices = (all_score_targets == c)
        score_predictions = all_score_predictions[indices]
        score_targets = all_score_targets[indices]
        score_rmse += 1 / num_classes * _compute_rmse(score_predictions, score_targets)
    
    return score_rmse

def _compute_within_mcacc(all_score_predictions, all_score_targets, margin):
    unique_classes = torch.unique(all_score_targets)
    num_classes = len(unique_classes)
    score_rmse = 0.
    
    for c in unique_classes:
        indices = torch.where(all_score_targets == c)
        score_predictions = all_score_predictions[indices]
        score_targets = all_score_targets[indices]
        score_rmse += 1 / num_classes * _accuracy_within_margin(score_predictions, score_targets, margin)
    
    return score_rmse

def _compute_underestimate_mcrate(all_score_predictions, all_score_targets):
    unique_classes = torch.unique(all_score_targets)
    num_classes = len(unique_classes)
    score_rmse = 0.
    
    for c in unique_classes:
        indices = torch.where(all_score_targets == c)
        score_predictions = all_score_predictions[indices]
        score_targets = all_score_targets[indices]
        score_rmse += 1 / num_classes * _compute_under_estimate_rate(score_predictions, score_targets)
    
    return score_rmse

def _compute_overestimate_mcrate(all_score_predictions, all_score_targets):
    unique_classes = torch.unique(all_score_targets)
    num_classes = len(unique_classes)
    score_rmse = 0.
    
    for c in unique_classes:
        indices = torch.where(all_score_targets == c)
        score_predictions = all_score_predictions[indices]
        score_targets = all_score_targets[indices]
        score_rmse += 1 / num_classes * _compute_over_estimate_rate(score_predictions, score_targets)
    
    return score_rmse

def cal_pccs(x, y):
    """
    warning: data format must be narray
    :param x: Variable 1
    :param y: The variable 2
    :param n: The number of elements in x
    :return: pccs
    """
    n = len(x)
    sum_xy = np.sum(np.sum(x*y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x*x))
    sum_y2 = np.sum(np.sum(y*y))
    pcc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))
    return pcc

def weight_groups(numbers, hps):
    numbers = numbers.tolist()
    groups = []
    current_group = []
    for number in numbers:
        if not current_group or number == current_group[-1] + 1:
            current_group.append(number)
        else:
            groups.append(current_group)
            current_group = [number]
    groups.append(current_group)
    # assert len(groups) == hps.batch_size
    w = []
    for g in groups:
        w += [float(1/len(g))]*len(g)
    w = torch.FloatTensor(w).to(hps.device)
    return w

def make_up_weights(weight_dict, labels, hps):
    if isinstance(labels, torch.Tensor):
        labels = labels.to(torch.int64).cpu().detach().tolist()
    r = list()
    for i in labels:
        r.append(1-weight_dict[i])
    r = torch.FloatTensor(r).to(hps.device)
    return r

def pikleOpen(filename):
    file_to_read = open( filename , "rb" )
    p = pickle.load( file_to_read )
    return p

def pickleStore(savethings , filename):
    dbfile = open( filename , 'wb' )
    pickle.dump( savethings , dbfile )
    dbfile.close()
    return