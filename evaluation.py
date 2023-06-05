#!/usr/bin/python
# -*- coding: utf-8 -*-

# __author__="Danqing Wang"

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import datetime
import os
import time
import json
import glob

import numpy as np
import pandas as pd
from pytablewriter import ExcelXlsxTableWriter

import torch
import torch.nn as nn

from HiGraph import HSumGraph, HSumPromptGraph, HSumDocGraph
from module.dataloader import ExampleSet, ExamplePromptSet, MultiExampleSet, graph_collate_fn
from module.INFOembedding import CEFREmbed, FILLEDEmbed
from module.embedding import Word_Embedding
from module.vocabulary import Vocab
from tools.args import get_eval_args
from tools.logger import *
import seaborn as sns
import matplotlib.pyplot as plt
from tools.utils import (
    CEFR2INT,
    BERT2ABB,
    compute_micro_underestimate_rate, 
    compute_macro_underestimate_rate,
    compute_micro_overestimate_rate,
    compute_macro_overestimate_rate,
    cal_pccs,
    _accuracy_within_margin,
    _compute_within_mcacc,
    _compute_mcrmse,
    _compute_over_estimate_rate,
    _compute_under_estimate_rate,
    _compute_overestimate_mcrate,
    _compute_underestimate_mcrate,
    pickleStore
)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report, f1_score

from transformers import AutoTokenizer, AutoConfig

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def load_test_model(model, model_name, eval_dir, save_root, save_dir_name, device):
    """ choose which model will be loaded for evaluation """
    if model_name.startswith('eval'):
        bestmodel_load_path = os.path.join(eval_dir, model_name[4:])
    elif model_name.startswith('train'):
        train_dir = os.path.join(save_root, save_dir_name, "train")
        bestmodel_load_path = os.path.join(train_dir, model_name[5:])
    elif model_name == "earlystop":
        train_dir = os.path.join(save_root, save_dir_name, "train")
        bestmodel_load_path = os.path.join(train_dir, 'earlystop')
    else:
        logger.error("None of such model! Must be one of evalbestmodel/trainbestmodel/earlystop")
        raise ValueError("None of such model! Must be one of evalbestmodel/trainbestmodel/earlystop")
    if not os.path.exists(bestmodel_load_path):
        logger.error("[ERROR] Restoring %s for testing...The path %s does not exist!", model_name, bestmodel_load_path)
        return None
    logger.info("[INFO] Restoring %s for testing...The path is %s", model_name, bestmodel_load_path)

    model.load_state_dict(torch.load(bestmodel_load_path, map_location=device))

    return model

def round_num(num):
    return round(num, 3)

def run_test(model, dataset, loader, model_name, hps):
    test_dir = os.path.join(hps.save_root, hps.save_dir_name, "test") # make a subdir of the root dir for eval data
    eval_dir = os.path.join(hps.save_root, hps.save_dir_name, "eval")
    alsy_dir = os.path.join(hps.save_root, hps.save_dir_name, "analysis")
    tsne_dir = os.path.join(hps.save_root, hps.save_dir_name, "tsne")
    if not os.path.exists(test_dir) : os.makedirs(test_dir)
    if not os.path.exists(alsy_dir) : os.makedirs(alsy_dir)
    if not os.path.exists(tsne_dir) : os.makedirs(tsne_dir)
    if not os.path.exists(eval_dir) :
        logger.exception("[Error] eval_dir %s doesn't exist. Run in train mode to create it.", eval_dir)
        raise Exception("[Error] eval_dir %s doesn't exist. Run in train mode to create it." % (eval_dir))

    resfile = None
    if hps.save_label:
        log_dir = os.path.join(test_dir, hps.cache_dir.split("/")[-1])
        resfile = open(log_dir, "w")
        logger.info("[INFO] Write the Evaluation into %s", log_dir)

    model = load_test_model(model, model_name, eval_dir, hps.save_root, hps.save_dir_name, hps.device)
    model.eval()

    predictions, labels, wlabels = [], [], []
    if hps.tsne:
        sembeds, wembeds = [], []
    iter_start_time=time.time()
    with torch.no_grad():
        logger.info("[Model] Sequence Labeling!")

        for i, (G, index, bert_input_ids, G_itvr) in enumerate(loader):
            if hps.cuda:
                G.to(hps.device)
                if G_itvr:
                    G_itvr.to(hps.device)

            if hps.tsne:
                p2w_embed = []

            if hps.problem_type == 'classification':

                outputs, sent_states, word_feature = model.forward(G, G_itvr, bert_input_ids)  # [n_snodes, 6]
                snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
                wnode_id = G.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
                wid = G.nodes[wnode_id].data["id"]
                label = torch.argmax(G.ndata["label"][snode_id], dim=1)

                if hps.eval_speaker_wise:
                    predictions.append(np.mean(torch.argmax(outputs, dim=1).cpu().numpy()).item())
                    labels.append(label.tolist()[0])
                    wlabels.extend(wid.cpu().tolist())
                else:
                    predictions.extend(torch.argmax(outputs, dim=1).tolist())
                    labels.extend(label.tolist())
                    wlabels.extend(wid.cpu().tolist())

            if hps.problem_type == 'regression':
                
                outputs, sent_states, word_feature = model.forward(G, G_itvr, bert_input_ids)  # [n_snodes, 6]
                snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
                wnode_id = G.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
                wid = G.nodes[wnode_id].data["id"]
                label = torch.argmax(G.ndata["label"][snode_id], dim=1)

                if hps.tsne:
                    sembeds.append(sent_states.cpu())
                    wembeds.append(word_feature.cpu())
                
                if hps.eval_speaker_wise:
                    predictions.append(np.mean(outputs.squeeze(-1).cpu().numpy()).item())
                    labels.append(label.tolist()[0])
                    wlabels.append(wid.cpu().tolist())
                else:
                    predictions.extend(outputs.squeeze(-1).tolist())
                    labels.extend(label.tolist())
                    wlabels.append(wid.cpu().tolist())

    if hps.tsne:
        sembeds_save_file_path = os.path.join(alsy_dir, 'slabels.sembeds')
        wembeds_save_file_path = os.path.join(alsy_dir, 'wlabels.wembeds')
        pickleStore((labels, sembeds), sembeds_save_file_path)
        pickleStore((wlabels, wembeds), wembeds_save_file_path)
        print('Paragraph Nodes Embeddings for t-SNE were saved at {}'.format(sembeds_save_file_path))
        print('Word Nodes Embeddings for t-SNE were saved at {}'.format(wembeds_save_file_path))

    predictions = torch.FloatTensor(predictions)
    labels = torch.FloatTensor(labels)

    INT2CEFR = { idx:scale for scale, idx in CEFR2INT.items() }

    if hps.problem_type == 'regression':
        labels = labels + torch.ones_like(labels)

    if hps.problem_type == 'classification':
        logger.info('[INFO] End of test | time: {:5.2f}s | acc | test micro accuracy {:5.3f} | '.format((time.time() - iter_start_time), round_num(precision_score(predictions, labels, average='micro')) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | mc_acc |test macro accuracy {:5.3f} | '.format((time.time() - iter_start_time), round_num(precision_score(predictions, labels, average='macro')) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | rc | test micro recall {:5.3f} | '.format((time.time() - iter_start_time), round_num(recall_score(predictions, labels, average='micro')) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | mc_rc | test macro recall {:5.3f} | '.format((time.time() - iter_start_time), round_num(recall_score(predictions, labels, average='macro')) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | f1| test micro f1 {:5.3f} | '.format((time.time() - iter_start_time), round_num(f1_score(predictions, labels, average='micro')) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | mc_f1 | test macro f1 {:5.3f} | '.format((time.time() - iter_start_time), round_num(f1_score(predictions, labels, average='macro')) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | ur | test micro under-estimate rate {:5.3f} | '.format((time.time() - iter_start_time), round_num(compute_micro_underestimate_rate(predictions, labels)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | mc_ur| test macro under-estimate rate {:5.3f} | '.format((time.time() - iter_start_time), round_num(compute_macro_underestimate_rate(predictions, labels)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | or | test micro over-estimate rate {:5.3f} | '.format((time.time() - iter_start_time), round_num(compute_micro_overestimate_rate(predictions, labels)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | mc_or | test macro over-estimate rate {:5.3f} | '.format((time.time() - iter_start_time), round_num(compute_macro_overestimate_rate(predictions, labels)) ))
    elif hps.problem_type == 'regression':
        logger.info('[INFO] End of test | time: {:5.2f}s | rmse | test micro rmse {:5.3f} | '.format((time.time() - iter_start_time), round_num(np.sqrt(((predictions.cpu().numpy() - labels.cpu().numpy()) ** 2).mean())) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | mc_rmse | test macro rmse {:5.3f} | '.format((time.time() - iter_start_time), round_num(_compute_mcrmse(predictions, labels)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | pearson | test pcc {:5.3f} | '.format((time.time() - iter_start_time), round_num(cal_pccs(predictions.cpu().numpy(), labels.cpu().numpy())) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | within_0.5 | test micro marginal accuracy 0.5 {:5.3f} | '.format((time.time() - iter_start_time), round_num(_accuracy_within_margin(predictions, labels, 0.5)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | within_1 | test micro marginal accuracy 1.0 {:5.3f} | '.format((time.time() - iter_start_time), round_num(_accuracy_within_margin(predictions, labels, 1)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | mc_within_0.5 | test macro marginal accuracy 0.5 {:5.3f} | '.format((time.time() - iter_start_time), round_num(_compute_within_mcacc(predictions, labels, 0.5)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | mc_within_1 | test macro marginal accuracy 1.0 {:5.3f} | '.format((time.time() - iter_start_time), round_num(_compute_within_mcacc(predictions, labels, 1)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | oe_rate | test micro over-estimate rate {:5.3f} | '.format((time.time() - iter_start_time), round_num(_compute_over_estimate_rate(predictions, labels)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | mc_oe_rate | test macro over-estimate rate {:5.3f} | '.format((time.time() - iter_start_time), round_num(_compute_overestimate_mcrate(predictions, labels)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | ue_rate | test micro under-estimate rate {:5.3f} | '.format((time.time() - iter_start_time), round_num(_compute_under_estimate_rate(predictions, labels)) ))
        logger.info('[INFO] End of test | time: {:5.2f}s | mc_ue_rate | test macro under-estimate rate {:5.3f} | '.format((time.time() - iter_start_time), round_num(_compute_underestimate_mcrate(predictions, labels)) ))

    if hps.problem_type == 'regression':
        y_pred = []
        for score_num in predictions.tolist():
            ori_list = [ num - score_num for num in list(INT2CEFR.keys()) ]
            abs_list = [ abs(num) for num in ori_list ]
            min_num = min(abs_list) # get index is zero
            if min_num >=0.0 and min_num <= 1.0:
                get_idx_num = abs_list.index(min_num)
                scale_label = list(INT2CEFR.keys())[get_idx_num]
            else:
                get_idx_num = abs_list.index(min_num)
                get_ori_num = ori_list[get_idx_num]
                if get_ori_num > 0:
                    get_idx_num += 1
                    if get_idx_num > 6:
                        get_idx_num = 6
                else:
                    get_idx_num -= 1
                    if get_idx_num < 0:
                        get_idx_num = 0
                scale_label = list(INT2CEFR.keys())[get_idx_num]
            y_pred.append(scale_label)
        predictions = torch.FloatTensor(y_pred)

    labels_keys = list(CEFR2INT.keys())
    labels_keys.remove(0)
    if hps.problem_type == 'regression':
        labels = [INT2CEFR[i] for i in labels.to(int).tolist()] # 1 - 5
        predictions = [INT2CEFR[i] for i in predictions.to(int).tolist()]
    elif hps.problem_type == 'classification':
        labels = [INT2CEFR[i] for i in (labels + torch.ones_like(labels)).to(int).tolist()] # (0 - 4) ++ 1 = (1 - 5)
        predictions = [INT2CEFR[i] for i in (predictions + torch.ones_like(predictions)).tolist()]
    cm = confusion_matrix(labels, predictions, labels=labels_keys)

    # confusion matrix - by count show count
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cbar_kws={'label': 'counts'}) #annot=True to annotate cells, ftm='g' to disable scientific notation
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(labels_keys)
    ax.yaxis.set_ticklabels(labels_keys)
    save_file_path = os.path.join(alsy_dir, 'confusion_matrix.count.sns.{}.png'.format(model_name))
    fig.savefig(save_file_path, bbox_inches='tight')
    print('confusion matrix (by count show count) is saved at {}'.format(save_file_path))

    # confusion matrix - by correctness show count
    cf_matrix_dict = dict()
    for scale in labels_keys:
        cf_matrix_dict.setdefault(
            scale,
            [0] * len(labels_keys)
        )
    for (true, pred) in zip(labels, predictions):
        scale_row = cf_matrix_dict[true]
        get_row_idx = labels_keys.index(pred)
        scale_row[get_row_idx] += 1
        cf_matrix_dict[true] = scale_row
    cf_matrix = np.array(list(cf_matrix_dict.values()))
    df_confusion = pd.DataFrame(cf_matrix, index=labels_keys, columns=labels_keys)
    df_confusion['TOTAL'] = df_confusion.sum(axis=1)
    df_confusion.loc['TOTAL']= df_confusion.sum()
    df_percentages = df_confusion.div(df_confusion.TOTAL, axis=0) # get percentages
    df_percentages.TOTAL = 0
    df_percentages.drop('TOTAL', inplace=True, axis=1) # drop col TOTAL
    df_percentages.drop('TOTAL', inplace=True, axis=0) # drop row TOTAL
    df_confusion.drop('TOTAL', inplace=True, axis=1) # drop col TOTAL
    df_confusion.drop('TOTAL', inplace=True, axis=0) # drop row TOTAL

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    sns.heatmap(data=df_percentages, annot=df_percentages, cmap='Blues', fmt=".3f", ax=ax, cbar_kws={'label': 'percentages'})
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    save_file_path = os.path.join(alsy_dir, 'confusion_matrix.correctness.percentage.sns.{}.png'.format(model_name))
    fig.savefig(save_file_path, bbox_inches='tight')
    print('confusion matrix (by correctness show correctness) is saved at {}'.format(save_file_path))
    
    # confusion matrix - by correctness show count
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    sns.heatmap(data=df_percentages, annot=df_confusion, cmap='Blues', fmt="d", ax=ax, cbar_kws={'label': 'percentages'})
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    save_file_path = os.path.join(alsy_dir, 'confusion_matrix.correctness.count.sns.{}.png'.format(model_name))
    fig.savefig(save_file_path, bbox_inches='tight')
    print('confusion matrix (by correctness show count) is saved at {}'.format(save_file_path))
    
    return alsy_dir

def write_xlsx_table(hps, log_file, alsy_dir):
    # write table
    log_content = open(log_file, 'r').readlines()
    log_content = {line.split('|')[2].strip(): line.split('|')[3].strip().split()[-1].strip() for line in log_content if '[INFO] End of test |' in line}
    writer = ExcelXlsxTableWriter()
    writer.table_name = "output"
    if hps.problem_type == 'regression':
        writer.headers = ["rmse", "mc_rmse", "pearson", "within_0.5", "within_1", "mc_within_0.5", "mc_within_1", "oe_rate", "mc_oe_rate", "ue_rate", "mc_ue_rate"]
        writer.value_matrix = [
            [str(float(log_content[k])) for k in writer.headers],
        ]
    elif hps.problem_type == 'classification':
        writer.headers = ["acc", "mc_acc", "rc", "mc_rc", "f1", "mc_f1", "ur", "mc_ur", "or", "mc_or"]
        writer.value_matrix = [
            [str(float(log_content[k])) for k in writer.headers],
        ]
    logger.info('[INFO] Save report.xlsx file at {}'.format(os.path.join(alsy_dir, 'report.xlsx')))
    writer.dump(os.path.join(alsy_dir, 'report.xlsx'))

def main():
    # parser = argparse.ArgumentParser(description='HeterSumGraph Model')

    # # Where to find data
    # parser.add_argument('--data_dir', type=str, default='data/CNNDM', help='The dataset directory.')
    # parser.add_argument('--cache_dir', type=str, default='cache/CNNDM', help='The processed dataset directory')
    # parser.add_argument('--embedding_path', type=str, default='/remote-home/dqwang/Glove/glove.42B.300d.txt', help='Path expression to external word embedding.')

    # # Important settings
    # parser.add_argument('--model', type=str, default="HSumGraph", help="model structure[HSG|HDSG]")
    # parser.add_argument('--test_model', type=str, default='evalbestmodel', help='choose different model to test [multi/evalbestmodel/trainbestmodel/earlystop]')
    # parser.add_argument('--use_pyrouge', action='store_true', default=False, help='use_pyrouge')

    # # Where to save output
    # parser.add_argument('--save_root', type=str, default='save/', help='Root directory for all model.')
    # parser.add_argument('--log_root', type=str, default='log/', help='Root directory for all logging.')
    # parser.add_argument('--save_dir_name', type=str, default=None, help='Root directory for all logging.')

    # # Speaker-wise
    # parser.add_argument('--eval_speaker_wise', action='store_true', default=False, help='Run evaluation in terms of speaker-wise results.')

    # # Hyperparameters
    # parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    # parser.add_argument('--bert_gpu', type=int, default=1, help='GPU ID of BERT model to use. [default: 4]')
    # parser.add_argument('--cuda', action='store_true', default=False, help='use cuda')
    # parser.add_argument('--num_workers', type=int, default=0, help='numbers of workers [default: 32]')
    # parser.add_argument('--vocab_size', type=int, default=50000, help='Size of vocabulary.')
    # parser.add_argument('--batch_size', type=int, default=1, help='Mini batch size [default: 32]')
    # parser.add_argument('--n_iter', type=int, default=1, help='iteration ')
    # parser.add_argument('--reweight', action='store_true', default=False, help='Reweight the loss when training and evaluation')
    # parser.add_argument('--rw_alpha', type=float, default=1.5, help='iteration hop [default: 1.5]')

    # # BERT fusion
    # parser.add_argument('--bert', action='store_true', default=False, help='BERT embedding fusion')
    # parser.add_argument('--bert_model_path', type=str, default="sentence-transformers/all-mpnet-base-v2", choices=['bert-base-uncased', 'roberta-base', 'xlm-roberta-base', 'distilroberta-base', 'allenai/longformer-base-4096', 'sentence-transformers/all-mpnet-base-v2', 'databricks/dolly-v2-12b'], help="Pre-trained BERT model to extend. e.g. ['bert-base-uncased', 'roberta-base', 'xlm-roberta-base', 'distilroberta-base', 'sentence-transformers/all-mpnet-base-v2', 'databricks/dolly-v2-12b]")
    # parser.add_argument('--bert_mp', action='store_true', default=False, help='Mean Pooling After BERT encoding')

    # parser.add_argument('--word_embedding', action='store_true', default=False, help='whether to use Word embedding')
    # parser.add_argument('--word_emb_dim', type=int, default=300, help='Word embedding size [default: 300]')
    # parser.add_argument('--embed_train', action='store_true', default=False, help='whether to train Word embedding [default: False]')
    # parser.add_argument('--feat_embed_size', type=int, default=50, help='feature embedding size [default: 50]')
    # parser.add_argument('--n_layers', type=int, default=1, help='Number of GAT layers [default: 1]')
    # parser.add_argument('--lstm_hidden_state', type=int, default=128, help='size of lstm hidden state')
    # parser.add_argument('--lstm_layers', type=int, default=2, help='lstm layers')
    # parser.add_argument('--bidirectional', action='store_true', default=True, help='use bidirectional LSTM')
    # parser.add_argument('--n_feature_size', type=int, default=128, help='size of node feature')
    # parser.add_argument('--hidden_size', type=int, default=64, help='hidden size [default: 64]')
    # parser.add_argument('--gcn_hidden_size', type=int, default=128, help='hidden size [default: 64]')
    # parser.add_argument('--ffn_inner_hidden_size', type=int, default=512, help='PositionwiseFeedForward inner hidden size [default: 512]')
    # parser.add_argument('--n_head', type=int, default=8, help='multihead attention number [default: 8]')
    # parser.add_argument('--recurrent_dropout_prob', type=float, default=0.1, help='recurrent dropout prob [default: 0.1]')
    # parser.add_argument('--atten_dropout_prob', type=float, default=0.1,help='attention dropout prob [default: 0.1]')
    # parser.add_argument('--ffn_dropout_prob', type=float, default=0.1, help='PositionwiseFeedForward dropout prob [default: 0.1]')
    # parser.add_argument('--use_orthnormal_init', action='store_true', default=True, help='use orthnormal init for lstm [default: true]')
    # parser.add_argument('--pmi_window_width', type=int, default=-1,help='Use PMI information for word node to word node')
    # parser.add_argument('--sent_max_len', type=int, default=2000, help='max length of sentences (max source text sentence tokens)')
    # parser.add_argument('--doc_max_timesteps', type=int, default=10, help='max length of documents (max timesteps of documents)')
    # parser.add_argument('--save_label', action='store_true', default=False, help='require multihead attention')
    # parser.add_argument('--limited', action='store_true', default=False, help='limited hypo length')
    # parser.add_argument('--blocking', action='store_true', default=False, help='ngram blocking')
    # parser.add_argument('--sentaspara', type=str, default='sent', choices=['sent', 'para'], help='for gradient clipping max gradient normalization')
    # parser.add_argument('--mean_paragraphs', type=str, default=None, choices=[None, 'mean', 'mean_residual', 'mean_residual_add'],help='max length of documents (max timesteps of documents)')
    # parser.add_argument('--problem_type', type=str, default='classification', choices=['regression', 'classification'], help='Regard problem as regression classification')
    # parser.add_argument('--head', type=str, default='linear', choices=['linear', 'predictionhead'], help="Prediction Head")
    # parser.add_argument('--pred_gated_fusion', action='store_true', default=False, help='Use Gate Weight for element-wise add')

    # # information embedding
    # parser.add_argument('--cefr_word', action='store_true', default=False, help='Use CEFR vocabulary profile information')
    # parser.add_argument('--cefr_info', type=str, default="embed_init", choices=['embed_init', 'graph_init'], help="CEFR node embedding")
    # parser.add_argument('--filled_pauses_word', action='store_true', default=False, help='Use disfluency tag information')
    # parser.add_argument('--filled_pauses_info', type=str, default="embed_init", choices=['embed_init', 'graph_init'], help="Filled Pause embedding")

    # # debug
    # parser.add_argument('--test_final', action='store_true', default=False, help='Use CEFR vocabulary profile information')

    # # ordinal entropy
    # parser.add_argument('--oe', action='store_true', default=False, help='Use ordinal entropy')

    # # Interviewer information
    # parser.add_argument('--interviewer', action='store_true', default=False, help='Use interviewer information')

    # # cheat baseline
    # parser.add_argument('--baseline', action='store_true', default=False, help='Use ordinal entropy')

    # parser.add_argument('--tsne', action='store_true', default=False, help='Save final hidden states form visuailizing in t-SNE.')
    # parser.add_argument('-m', type=int, default=5, help='decode summary length')

    # args = parser.parse_args()
    args = get_eval_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.set_printoptions(threshold=50000)

    # File paths
    DATA_FILE = os.path.join(args.data_dir, "eval_combo.{}.label.jsonl".format(args.sentaspara))
    VOCAL_FILE = os.path.join(args.cache_dir, "vocab.combine" if args.interviewer else 'vocab')
    FILTER_WORD = os.path.join(args.cache_dir, "filter_word.{}.txt".format(args.sentaspara))
    LOG_PATH = args.log_root
    INTERVIEWER_DATA_FILE, BERT_DATA_PATH = None, None
    if args.interviewer:
        INTERVIEWER_DATA_FILE  = os.path.join(args.data_dir, "eval_combo.{}.interviewer.label.jsonl".format(args.sentaspara))
        logger.info("[INFO] Use interviewer's information")
    if args.bert:
        BERT_DATA_PATH = os.path.join(args.data_dir, "eval_combo.{}.combine.label.jsonl".format(args.sentaspara))

    # CEFR node and Filled Pauses node
    cefr_loader, filled_pauses_loader = None, None
    if args.cefr_word and (args.cefr_info == 'embed_init'):
        VOCABPROFILE_FILE = os.path.join(args.data_dir, 'cefrj1.6_c1c2.final.txt')
        cefr_loader = CEFREmbed(args.word_emb_dim, VOCABPROFILE_FILE)
    if args.filled_pauses_word and (args.cefr_info == 'embed_init'):
        FLUENCYPAUSE_FILE = os.path.join(args.data_dir, 'all.filled_pauses.txt')
        filled_pauses_loader = FILLEDEmbed(args.word_emb_dim, FLUENCYPAUSE_FILE)

    # train_log setting
    if not os.path.exists(LOG_PATH):
        logger.exception("[Error] Logdir %s doesn't exist. Run in train mode to create it.", LOG_PATH)
        raise Exception("[Error] Logdir %s doesn't exist. Run in train mode to create it." % (LOG_PATH))
    log_path = os.path.join(LOG_PATH, "do_test_log")
    file_handler = logging.FileHandler(log_path, 'w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Pytorch %s", torch.__version__)
    logger.info("[INFO] Create Vocab, vocab path is %s", VOCAL_FILE)
    vocab = Vocab(VOCAL_FILE, args.vocab_size)
    embed = torch.nn.Embedding(vocab.size(), args.word_emb_dim)
    if args.word_embedding:
        embed_loader = Word_Embedding(args.embedding_path, vocab)
        vectors = embed_loader.load_my_vecs_with_infos(cefr_loader, filled_pauses_loader, k=args.word_emb_dim) if (cefr_loader or filled_pauses_loader) else embed_loader.load_my_vecs(args.word_emb_dim)
        pretrained_weight = embed_loader.add_unknown_words_by_avg(vectors, args.word_emb_dim)
        embed.weight.data.copy_(torch.Tensor(pretrained_weight))
        embed.weight.requires_grad = args.embed_train
        logger.info("[INFO] Use GLOVE to get the embeddings of both words and paragraph")
    else:
        embed.weight.requires_grad = args.embed_train
        logger.info("[INFO] Use random initial embeddings of both words and paragraph")

    hps = args
    logger.info(hps)

    tokenizer = None
    args.bert_config = None
    if args.bert:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model_path,
                                                  is_split_into_words=True)
        bert_config = AutoConfig.from_pretrained(args.bert_model_path, output_hidden_states=True)
        args.bert_config = bert_config

    test_w2s_path = os.path.join(args.cache_dir, "eval_combo.{}.w2s.tfidf.jsonl".format(args.sentaspara))

    if hps.model == "HSG":
        model = HSumGraph(hps, embed)
        logger.info("[MODEL] HeterSumGraph ")
        dataset = ExampleSet(DATA_FILE, INTERVIEWER_DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, test_w2s_path, hps.pmi_window_width, tokenizer, hps)
        loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=graph_collate_fn)
    elif hps.model == "HSPG":
        model = HSumPromptGraph(hps, embed)
        logger.info("[MODEL] HSumPromptGraph ")
        dataset = ExamplePromptSet(DATA_FILE, INTERVIEWER_DATA_FILE, BERT_DATA_PATH, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, test_w2s_path, hps.pmi_window_width, tokenizer, hps)
        loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=graph_collate_fn)
    elif hps.model == "HDSG":
        model = HSumDocGraph(hps, embed)
        logger.info("[MODEL] HeterDocSumGraph ")
        test_w2d_path = os.path.join(args.cache_dir, "eval_combo.{}.w2s.tfidf.jsonl".format(args.sentaspara))
        dataset = MultiExampleSet(DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, test_w2s_path, test_w2d_path)
        loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers,collate_fn=graph_collate_fn)
    else:
        logger.error("[ERROR] Invalid Model Type!")
        raise NotImplementedError("Model Type has not been implemented")

    hps.device = torch.device("cpu")
    if args.cuda:
        hps.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
        model = model.to(hps.device)
        logger.info("[INFO] Use cuda")

    if hps.save_dir_name is None:
        hps.save_dir_name = "{}".format(
            '.'.join(
                [hps.model, hps.sentaspara] + \
                (['reweight{}'.format('' if hps.rw_alpha == 1. else hps.rw_alpha)] if hps.reweight else []) + \
                ([] if hps.mean_paragraphs is None else [hps.mean_paragraphs]) + \
                [hps.problem_type] + \
                [hps.head] + \
                ([BERT2ABB[hps.bert_model_path]] if hps.bert else []) + \
                (['bmp'] if hps.bert_mp else []) + \
                (['glove'] if hps.word_embedding else ['randembed']) + \
                (['pmi{}'.format(hps.pmi_window_width)] if hps.pmi_window_width > -1 else []) + \
                (['interviewer'] if hps.interviewer else []) + \
                (['gw'] if hps.pred_gated_fusion else []) + \
                (['cefr{}'.format(hps.cefr_info)] if hps.cefr_word else []) + \
                (['fp{}'.format(hps.filled_pauses_info)] if hps.filled_pauses_word else []) + \
                (['test'] if hps.test_final else []) + \
                (['oe'] if hps.oe else [])
            )
        )
    if hps.baseline:
        hps.save_dir_name = "{}".format(
            '.'.join(
                [hps.model, hps.sentaspara] + \
                (['reweight{}'.format('' if hps.rw_alpha == 1. else hps.rw_alpha)] if hps.reweight else []) + \
                ([] if hps.mean_paragraphs is None else [hps.mean_paragraphs]) + \
                [hps.problem_type] + \
                [hps.head] + \
                ([BERT2ABB[hps.bert_model_path]] if hps.bert else []) + \
                (['bmp'] if hps.bert_mp else []) + \
                ['baseline']
            )
        )

    logger.info("[INFO] Decoding...")
    if hps.test_model == "multi":
        spath = os.path.join(hps.save_root, hps.save_dir_name, "eval", "bestmodel_*")
        for model_name in sorted(glob.glob(spath)):
            model_name = "eval" + os.path.basename(model_name)
            run_test(model, dataset, loader, model_name, hps)
    else:
        alsy_dir = run_test(model, dataset, loader, hps.test_model, hps)
        write_xlsx_table(hps, log_path, alsy_dir)

if __name__ == '__main__':
    main()
