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
import sys
import shutil
import time
import random

import wandb

import dgl
import numpy as np
import torch
from rouge import Rouge

from HiGraph import HSumGraph, HSumPromptGraph, HSumDocGraph

from module.dataloader import ExampleSet, ExamplePromptSet, MultiExampleSet, graph_collate_fn
from module.embedding import Word_Embedding
from module.INFOembedding import CEFREmbed, FILLEDEmbed
from module.vocabulary import Vocab
from tools.args import get_train_args
from tools.logger import *
from tools.utils import CEFR2INT, INT2CEFR, BERT2ABB, make_up_weights, weight_groups
from helpers.OrdinalEntropy import ordinal_entropy

import deepspeed
from tools.distri import init_dataloader, torch_distributed_master_process_first
from transformers import AutoTokenizer, AutoConfig

from sklearn.metrics import precision_score, recall_score, f1_score
from tools.utils import (
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
    _compute_underestimate_mcrate
)

_DEBUG_FLAG_ = False

def precompute_loss_weights(train_levels, max_score, epsilon=1e-5, alpha=2e-1):
    train_sentlv_ratio = np.array([np.sum(train_levels == float(lv)) for lv in range(0, int(max_score))])
    train_sentlv_ratio = train_sentlv_ratio / np.sum(train_sentlv_ratio)
    train_sentlv_weights = np.power(train_sentlv_ratio, alpha) / np.sum(
        np.power(train_sentlv_ratio, alpha)) / (train_sentlv_ratio + epsilon)
    return torch.Tensor(train_sentlv_weights)


def save_model(model, save_file):
    with open(save_file, 'wb') as f:
        torch.save(model.state_dict(), f)
    logger.info('[INFO] Saving model to %s', save_file)


def setup_training(model, train_loader, valid_loader, valset, hps):
    """ Does setup before starting training (run_training)
    
        :param model: the model
        :param train_loader: train dataset loader
        :param valid_loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param hps: hps for model
        :return: 
    """

    train_dir = os.path.join(hps.save_root, hps.save_dir_name, "train")
    if os.path.exists(train_dir) and hps.restore_model != 'None':
        logger.info("[INFO] Restoring %s for training...", hps.restore_model)
        bestmodel_file = os.path.join(train_dir, hps.restore_model)
        model.load_state_dict(torch.load(bestmodel_file))
        hps.save_root = hps.save_root + "_reload"
    else:
        logger.info("[INFO] Create new model for training...")
        if os.path.exists(train_dir): shutil.rmtree(train_dir)
        os.makedirs(train_dir)

    try:
        run_training(model, train_loader, valid_loader, valset, hps, train_dir)
    except KeyboardInterrupt:
        logger.error("[Error] Caught keyboard interrupt on worker. Stopping supervisor...")
        save_model(model, os.path.join(train_dir, "earlystop"))


def calculate_local_metrics(predictions, labels, hps, mode):
    assert mode in ['trn', 'dev', 'eval']
    metrics = {}
    if hps.problem_type == 'classification':
        metrics['acc'] = precision_score(predictions, labels, average='micro')
        metrics['mc_acc'] = precision_score(predictions, labels, average='macro')
        metrics['rc'] = recall_score(predictions, labels, average='micro')
        metrics['mc_rc'] = recall_score(predictions, labels, average='macro')
        metrics['f1'] = f1_score(predictions, labels, average='micro')
        metrics['mc_f1'] = f1_score(predictions, labels, average='macro')
        metrics['ur'] = compute_micro_underestimate_rate(predictions, labels)
        metrics['mc_ur'] = compute_macro_underestimate_rate(predictions, labels)
        metrics['or'] = compute_micro_overestimate_rate(predictions, labels)
        metrics['mc_or'] = compute_macro_overestimate_rate(predictions, labels)
    elif hps.problem_type == 'regression':
        metrics['rmse'] = np.sqrt(((predictions.cpu().numpy() - labels.cpu().numpy()) ** 2).mean())
        metrics['mc_rmse'] = _compute_mcrmse(predictions, labels)
        metrics['pearson'] = cal_pccs(predictions.cpu().numpy(), labels.cpu().numpy())
        metrics['within_0.5'] = _accuracy_within_margin(predictions, labels, 0.5)
        metrics['within_1'] = _accuracy_within_margin(predictions, labels, 1)
        metrics['mc_within_0.5'] = _compute_within_mcacc(predictions, labels, 0.5)
        metrics['mc_within_1'] = _compute_within_mcacc(predictions, labels, 1)
        metrics['oe_rate'] = _compute_over_estimate_rate(predictions, labels)
        metrics['mc_oe_rate'] = _compute_overestimate_mcrate(predictions, labels)
        metrics['ue_rate'] = _compute_under_estimate_rate(predictions, labels)
        metrics['mc_ue_rate'] = _compute_underestimate_mcrate(predictions, labels)
    metrics = {'{}_{}'.format(mode, k): v for k, v in metrics.items()}
    return metrics

def setup_device(args, logger):
    if torch.cuda.is_available():
        if torch.distributed.is_initialized() and args.multiple_device:
            device = torch.device("cuda", args.local_rank)
        else:
            device = torch.device("cuda", int(args.gpu))
    else:
        device = torch.device("cpu")
    logger.info("[INFO] Use cuda")
    return device

def run_training(model, train_loader, valid_loader, valset, hps, train_dir):
    '''  Repeatedly runs training iterations, logging loss to screen and log files
    
        :param model: the model
        :param train_loader: train dataset loader
        :param valid_loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param hps: hps for model
        :param train_dir: where to save checkpoints
        :return: 
    '''
    logger.info("[INFO] Starting run_training")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)

    # loss reweight
    reweight = None
    if hps.reweight:
        labels = []
        for i, (G, index, bert_input_ids, G_itvr) in enumerate(train_loader):
            snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            label = torch.argmax(G.ndata["label"][snode_id], dim=1)
            labels.extend(label.tolist())
        labels = np.array(labels)
        if hps.problem_type == 'classfication':
            reweight = precompute_loss_weights(labels, 6, alpha=hps.rw_alpha) # it will give the weight of each class
        elif hps.problem_type == 'regression':
            labels = (labels + np.ones_like(labels.shape)).tolist()
            reweight = {i: (labels.count(i)/len(labels)) ** hps.rw_alpha for i in range(0, 7)}

    criterion = torch.nn.CrossEntropyLoss(weight=reweight, reduction='none', ignore_index=-1) if hps.problem_type == 'classfication' else torch.nn.MSELoss(reduction='none')
    if hps.cuda:
        criterion = criterion.to(hps.device)

    best_train_loss = None
    best_loss = None
    best_F = None
    non_descent_cnt = 0
    saveNo = 0
    Lambda_d = 1e-3

    torch.cuda.empty_cache() # out of memory
    for epoch in range(1, hps.n_epochs + 1):
        epoch_loss = 0.0
        train_loss = 0.0
        trn_pred, trn_lab = [], []
        epoch_start_time = time.time()
        for i, (G, index, bert_input_ids, G_itvr) in enumerate(train_loader):
            iter_start_time = time.time()
            model.train()

            if hps.cuda:
                G.to(hps.device)
                if G_itvr:
                    G_itvr.to(hps.device)

            if hps.problem_type == 'regression':
                outputs, sent_states, word_feature = model.forward(G, G_itvr, bert_input_ids)  # [n_snodes, 6]
                snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
                label = torch.argmax(G.ndata["label"][snode_id], dim=1).unsqueeze(-1).to(torch.float32)
                label = label + torch.ones_like(label).to(hps.device)
                outputs = outputs.squeeze(-1)
                label = label.squeeze(-1)
                loss_value = criterion(outputs, label)
                if hps.oe:
                    loss_oe = ordinal_entropy(sent_states, label) * Lambda_d
                    loss_value = loss_value + loss_oe
                if hps.reweight:
                    loss_value = loss_value * make_up_weights(reweight, label.to(int), hps)
                if hps.train_speaker_wise:
                    loss_value = loss_value * weight_groups(snode_id, hps)
                if hps.gradient_accumulation_steps:
                    loss_value = loss_value / hps.gradient_accumulation_steps
                G.nodes[snode_id].data["loss"] = loss_value  # [n_nodes, 1]
                loss = dgl.sum_nodes(G, "loss")  # [batch_size, 1]
                loss = loss.mean()
                trn_pred.extend(outputs.tolist())
                trn_lab.extend(label.tolist())
            elif hps.problem_type == 'classification':
                if hps.oe:
                    raise NotImplementedError('Not yet!')
                outputs, sent_states, word_feature = model.forward(G, G_itvr, bert_input_ids)  # [n_snodes, 6]
                snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
                label = G.ndata["label"][snode_id].float()
                loss_value = criterion(outputs, label).unsqueeze(-1)
                if hps.gradient_accumulation_steps:
                    loss_value = loss_value / hps.gradient_accumulation_steps
                G.nodes[snode_id].data["loss"] = loss_value   # [n_nodes, 1]
                loss = dgl.sum_nodes(G, "loss")  # [batch_size, 1]
                loss = loss.mean()
                trn_pred.extend(torch.argmax(outputs, dim=1).tolist())
                trn_lab.extend(torch.argmax(label, dim=1).tolist())

            if not (np.isfinite(loss.data.cpu())).numpy():
                logger.error("train Loss is not finite. Stopping.")
                logger.info(loss)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        logger.info(name)
                        # logger.info(param.grad.data.sum())
                raise Exception("train Loss is not finite. Stopping.")

            loss.backward()
            torch.cuda.empty_cache() # out of memory
            if hps.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hps.max_grad_norm)

            if (i + 1) % hps.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += float(loss.data)
            epoch_loss += float(loss.data)

            if i % 100 == 0:
                if _DEBUG_FLAG_:
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            logger.debug(name)
                            logger.debug(param.grad.data.sum())
                logger.info('       | end of iter {:3d} | time: {:5.2f}s | train loss {:5.4f} | '
                                .format(i, (time.time() - iter_start_time),float(train_loss / 100)))
                train_loss = 0.0

        if hps.lr_descent:
            new_lr = max(5e-6, hps.lr / (epoch + 1))
            for param_group in list(optimizer.param_groups):
                param_group['lr'] = new_lr
            logger.info("[INFO] The learning rate now is %f", new_lr)

        if torch.distributed.is_initialized():
            with torch_distributed_master_process_first(torch.distributed.get_rank()):
                trn_pred, trn_lab = torch.FloatTensor(trn_pred), torch.FloatTensor(trn_lab)
                std_metric = calculate_local_metrics(trn_pred, trn_lab, hps, 'trn')
                if (torch.distributed.get_rank() == 0) and hps.stdout_metric:
                    print(std_metric)
                if hps.wandb:
                    wandb_info = {
                        'train_loss': train_loss,
                        'epoch_loss': epoch_loss,
                        'lr': list(optimizer.param_groups)[0]['lr']
                    }
                    wandb_info.update(std_metric)
                    wandb.log(wandb_info)
        else:
            trn_pred, trn_lab = torch.FloatTensor(trn_pred), torch.FloatTensor(trn_lab)
            std_metric = calculate_local_metrics(trn_pred, trn_lab, hps, 'trn')
            if hps.stdout_metric:
                print(std_metric)
            if hps.wandb:
                wandb_info = {
                    'train_loss': train_loss,
                    'epoch_loss': epoch_loss,
                    'lr': list(optimizer.param_groups)[0]['lr']
                }
                wandb_info.update(std_metric)
                wandb.log(wandb_info)            

        epoch_avg_loss = epoch_loss / len(train_loader)
        logger.info('   | end of epoch {:3d} | time: {:5.2f}s | epoch train loss {:5.4f} | '
                    .format(epoch, (time.time() - epoch_start_time), float(epoch_avg_loss)))

        if not best_train_loss or epoch_avg_loss < best_train_loss:
            save_file = os.path.join(train_dir, "bestmodel")
            logger.info('[INFO] Found new best model with %.3f running_train_loss. Saving to %s', float(epoch_avg_loss),
                        save_file)
            save_model(model, save_file)
            best_train_loss = epoch_avg_loss
        
        # MAYBE try several times, not stop training while encounter temporary bad result
        # elif epoch_avg_loss >= best_train_loss:
        #     logger.error("[Error] training loss does not descent. Stopping supervisor...")
        #     save_model(model, os.path.join(train_dir, "earlystop"))
        #     sys.exit(1)

        best_loss, best_F, non_descent_cnt, saveNo = run_eval(model, valid_loader, valset, hps, best_loss, best_F, non_descent_cnt, saveNo, criterion, reweight)

        if non_descent_cnt >= hps.non_descent_count:
            
            if hps.wandb:
                wandb.finish()
            
            logger.error("[Error] val loss does not descent for {} times. Stopping supervisor...".format(hps.non_descent_count))
            save_model(model, os.path.join(train_dir, "earlystop"))
            return


def run_eval(model, loader, valset, hps, best_loss, best_F, non_descent_cnt, saveNo, criterion, reweight):
    ''' 
        Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far.
        :param model: the model
        :param loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param hps: hps for model
        :param best_loss: best valid loss so far
        :param best_F: best valid F so far
        :param non_descent_cnt: the number of non descent epoch (for early stop)
        :param saveNo: the number of saved models (always keep best saveNo checkpoints)
        :return: 
    '''
    logger.info("[INFO] Starting eval for this model ...")
    eval_dir = os.path.join(hps.save_root, hps.save_dir_name, "eval")  # make a subdir of the root dir for eval data
    if not os.path.exists(eval_dir): os.makedirs(eval_dir)

    model.eval()

    with torch.no_grad():

        dev_pred, dev_lab = [], []
        running_loss, batch_number = 0., 0
        for i, (G, index, bert_input_ids, G_itvr) in enumerate(loader):

            if hps.cuda:
                G.to(hps.device)
                if G_itvr:
                    G_itvr.to(hps.device)

            if hps.problem_type == 'regression':
                outputs, sent_states, word_feature = model.forward(G, G_itvr, bert_input_ids)  # [n_snodes, 6]
                snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
                label = torch.argmax(G.ndata["label"][snode_id], dim=1).unsqueeze(-1).to(torch.float32)
                label = label + torch.ones_like(label).to(hps.device)
                outputs = outputs.squeeze(-1)
                label = label.squeeze(-1)
                loss = criterion(outputs, label)
                loss = loss / torch.distributed.get_world_size() if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1 else loss
                if hps.oe:
                    loss_oe = ordinal_entropy(sent_states, label) * Lambda_d
                    loss = loss + loss_oe
                if hps.reweight:
                    loss = loss * make_up_weights(reweight, label.to(int), hps)
                if hps.train_speaker_wise:
                    loss = loss * weight_groups(snode_id, hps)
                loss = loss.mean()
                running_loss += float(loss.data)
                batch_number += 1
                dev_pred.extend(outputs.tolist())
                dev_lab.extend(label.tolist())
            elif hps.problem_type == 'classification':
                outputs, sent_states, word_feature = model.forward(G, G_itvr, bert_input_ids)  # [n_snodes, 6]
                snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
                label = G.ndata["label"][snode_id].float()
                loss = criterion(outputs, label).unsqueeze(-1)  # [n_nodes, 1]
                loss = loss / torch.distributed.get_world_size() if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1 else loss
                G.nodes[snode_id].data["loss"] = loss
                loss = dgl.sum_nodes(G, "loss")  # [batch_size, 1]
                loss = loss.mean()
                running_loss += float(loss.data)
                batch_number += 1
                dev_pred.extend(torch.argmax(outputs, dim=1).tolist())
                dev_lab.extend(torch.argmax(label, dim=1).tolist())

    running_avg_loss = running_loss / batch_number
    
    if hps.wandb:
        dev_pred, dev_lab = torch.FloatTensor(dev_pred), torch.FloatTensor(dev_lab)
        wandb_info = {
            "valid_loss": running_avg_loss
        }
        wandb_info.update(calculate_local_metrics(dev_pred, dev_lab, hps, 'dev'))
        wandb.log(wandb_info)

    if best_loss is None or running_avg_loss < best_loss:
        bestmodel_save_path = os.path.join(eval_dir, 'bestmodel_%d' % saveNo)  # this is where checkpoints of best models are saved
        if best_loss is not None:
            logger.info(
                '[INFO] Found new best model with %.6f running_avg_loss. The original loss is %.6f, Saving to %s',
                float(running_avg_loss), float(best_loss), bestmodel_save_path)
        else:
            logger.info(
                '[INFO] Found new best model with %.6f running_avg_loss. The original loss is None, Saving to %s',
                float(running_avg_loss), bestmodel_save_path)
        with open(bestmodel_save_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        best_loss = running_avg_loss
        non_descent_cnt = 0
        saveNo += 1
    else:
        non_descent_cnt += 1

    return best_loss, None, non_descent_cnt, saveNo


def fix_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if not args.use_amp:
    #     torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(cmd_args):
    
    # parser = argparse.ArgumentParser(description='HeterSumGraph Model')

    # # Where to find data
    # parser.add_argument('--data_dir', type=str, default='data/CNNDM',help='The dataset directory.')
    # parser.add_argument('--cache_dir', type=str, default='cache/CNNDM',help='The processed dataset directory')
    # parser.add_argument('--embedding_path', type=str, default='/remote-home/dqwang/Glove/glove.42B.300d.txt', help='Path expression to external word embedding.')

    # # Important settings
    # parser.add_argument('--model', type=str, default='HSG', help='model structure[HSG|HDSG|HSPG]')
    # parser.add_argument('--restore_model', type=str, default='None', help='Restore model for further training. [bestmodel/bestFmodel/earlystop/None]')

    # # Where to save output
    # parser.add_argument('--save_root', type=str, default='save/', help='Root directory for all model.')
    # parser.add_argument('--log_root', type=str, default='log/', help='Root directory for all logging.')
    # parser.add_argument('--save_dir_name', type=str, default=None, help='Root directory for all logging.')

    # # Speaker-wise
    # parser.add_argument('--train_speaker_wise', action='store_true', default=False, help='Run training in speaker-wise.')

    # # Hyperparameters
    # parser.add_argument('--seed', type=int, default=666, help='set the random seed [default: 666]')
    # parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use. [default: 0]')
    # parser.add_argument('--bert_gpu', type=int, default=1, help='GPU ID of BERT model to use. [default: 4]')
    # parser.add_argument('--cuda', action='store_true', default=False, help='GPU or CPU [default: False]')
    # parser.add_argument('--num_workers', type=int, default=0, help='numbers of workers [default: 32]')
    # parser.add_argument('--vocab_size', type=int, default=50000,help='Size of vocabulary. [default: 50000]')
    # parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs [default: 20]')
    # parser.add_argument('--batch_size', type=int, default=4, help='Mini batch size [default: 32]')
    # parser.add_argument('--n_iter', type=int, default=1, help='iteration hop [default: 1]')
    # parser.add_argument('--reweight', action='store_true', default=False, help='Reweight the loss when training and evaluation')
    # parser.add_argument('--rw_alpha', type=float, default=1.5, help='iteration hop [default: 1.5]')

    # parser.add_argument('--word_embedding', action='store_true', default=False, help='whether to use Word embedding [default: True]')
    # parser.add_argument('--word_emb_dim', type=int, default=300, help='Word embedding size [default: 300]')
    # parser.add_argument('--embed_train', action='store_true', default=False,help='whether to train Word embedding [default: False]')
    # parser.add_argument('--feat_embed_size', type=int, default=50, help='feature embedding size [default: 50]')
    # parser.add_argument('--n_layers', type=int, default=1, help='Number of GAT layers [default: 1]')
    # parser.add_argument('--lstm_hidden_state', type=int, default=128, help='size of lstm hidden state [default: 128]')
    # parser.add_argument('--lstm_layers', type=int, default=2, help='Number of lstm layers [default: 2]')
    # parser.add_argument('--bidirectional', action='store_true', default=True, help='whether to use bidirectional LSTM [default: True]')
    # parser.add_argument('--n_feature_size', type=int, default=128, help='size of node feature [default: 128]')
    # parser.add_argument('--hidden_size', type=int, default=64, help='hidden size [default: 64]')
    # parser.add_argument('--ffn_inner_hidden_size', type=int, default=512,help='PositionwiseFeedForward inner hidden size [default: 512]')
    # parser.add_argument('--n_head', type=int, default=8, help='multihead attention number [default: 8]')
    # parser.add_argument('--recurrent_dropout_prob', type=float, default=0.1,help='recurrent dropout prob [default: 0.1]')
    # parser.add_argument('--atten_dropout_prob', type=float, default=0.1, help='attention dropout prob [default: 0.1]')
    # parser.add_argument('--ffn_dropout_prob', type=float, default=0.1,help='PositionwiseFeedForward dropout prob [default: 0.1]')
    # parser.add_argument('--use_orthnormal_init', action='store_true', default=True,help='use orthnormal init for lstm [default: True]')
    # parser.add_argument('--pmi_window_width', type=int, default=-1,help='Use PMI information for word node to word node')
    # parser.add_argument('--sent_max_len', type=int, default=2000,help='max length of sentences (max source text sentence tokens)')
    # parser.add_argument('--doc_max_timesteps', type=int, default=10,help='max length of documents (max timesteps of documents)')
    # parser.add_argument('--mean_paragraphs', type=str, default=None, choices=[None, 'mean', 'mean_residual'],help='max length of documents (max timesteps of documents)')
    # parser.add_argument('--head', type=str, default='linear', choices=['linear', 'predictionhead'], help="Prediction Head")

    # # BERT fusion
    # parser.add_argument('--bert', action='store_true', default=False, help='BERT embedding fusion')
    # parser.add_argument('--bert_model_path', type=str, default="sentence-transformers/all-mpnet-base-v2", choices=['bert-base-uncased', 'roberta-base', 'xlm-roberta-base', 'distilroberta-base', 'allenai/longformer-base-4096', 'sentence-transformers/all-mpnet-base-v2', 'databricks/dolly-v2-12b'], help="Pre-trained BERT model to extend. e.g. ['bert-base-uncased', 'roberta-base', 'xlm-roberta-base', 'distilroberta-base', 'sentence-transformers/all-mpnet-base-v2', 'databricks/dolly-v2-12b]")
    # parser.add_argument('--bert_mp', action='store_true', default=False, help='Mean Pooling After BERT encoding')
    
    # # Interviewer information
    # parser.add_argument('--interviewer', action='store_true', default=False, help='Use interviewer information')
    
    # # information embedding
    # parser.add_argument('--cefr_word', action='store_true', default=False, help='Use CEFR vocabulary profile information')
    # parser.add_argument('--cefr_info', type=str, default="embed_init", choices=['embed_init', 'graph_init'], help="CEFR node embedding")
    # parser.add_argument('--filled_pauses_word', action='store_true', default=False, help='Use disfluency tag information')
    # parser.add_argument('--filled_pauses_info', type=str, default="embed_init", choices=['embed_init', 'graph_init'], help="Filled Pause embedding")

    # # debug
    # parser.add_argument('--test_final', action='store_true', default=False, help='Use CEFR vocabulary profile information')

    # # ordinal entropy
    # parser.add_argument('--oe', action='store_true', default=False, help='Use ordinal entropy')

    # # cheat baseline
    # parser.add_argument('--baseline', action='store_true', default=False, help='Use ordinal entropy')

    # # Training
    # parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    # parser.add_argument('--lr_descent', action='store_true', default=False, help='learning rate descent')
    # parser.add_argument('--grad_clip', action='store_true', default=False, help='for gradient clipping')
    # parser.add_argument('--max_grad_norm', type=float, default=1.0, help='for gradient clipping max gradient normalization')
    # parser.add_argument('--sentaspara', type=str, default='sent', choices=['sent', 'para'], help='for gradient clipping max gradient normalization')
    # parser.add_argument('--problem_type', type=str, default='classification', choices=['regression', 'classification'], help='Regard problem as regression classification')
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of updates steps to accumulate before performing a backward/update pass.')
    # parser.add_argument('--non_descent_count', type=int, default=6, help='Early stop if loss not getting lower within specific times')
    # parser.add_argument('--wandb', action='store_true', default=False, help='WanDB to record your training states')
    # parser.add_argument('--pred_gated_fusion', action='store_true', default=False, help='Use Gate Weight for element-wise add')
    # parser.add_argument('--stdout_metric', action='store_true', default=False, help='Print metrics during training')

    # parser.add_argument('-m', type=int, default=5, help='decode summary length')

    # args = parser.parse_args()
    args = get_train_args()
    
    # set the seed
    fix_seed(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.set_printoptions(threshold=50000)

    # File paths
    DATA_FILE = os.path.join(args.data_dir, "trn_combo.{}.label.jsonl".format(args.sentaspara))
    VALID_FILE = os.path.join(args.data_dir, "dev_combo.{}.label.jsonl".format(args.sentaspara))
    VOCAL_FILE = os.path.join(args.cache_dir, "vocab.combine" if args.interviewer else 'vocab')
    FILTER_WORD = os.path.join(args.cache_dir, "filter_word.{}.txt".format(args.sentaspara))
    LOG_PATH = args.log_root
    INTERVIEWER_DATA_FILE, INTERVIEWER_VALID_FILE, BERT_DATA_PATH, BERT_VALID_FILE = None, None, None, None
    if args.interviewer:
        INTERVIEWER_DATA_FILE  = os.path.join(args.data_dir, "trn_combo.{}.interviewer.label.jsonl".format(args.sentaspara))
        INTERVIEWER_VALID_FILE = os.path.join(args.data_dir, "dev_combo.{}.interviewer.label.jsonl".format(args.sentaspara))
        logger.info("[INFO] Use interviewer's information")
    if args.bert:
        BERT_DATA_PATH = os.path.join(args.data_dir, "trn_combo.{}.combine.label.jsonl".format(args.sentaspara))
        BERT_VALID_FILE = os.path.join(args.data_dir, "dev_combo.{}.combine.label.jsonl".format(args.sentaspara))

    # train_log setting
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_PATH, "train_" + nowTime)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # CEFR node and Filled Pauses node
    cefr_loader, filled_pauses_loader = None, None
    if args.cefr_word and (args.cefr_info == 'embed_init'):
        VOCABPROFILE_FILE = os.path.join(args.data_dir, 'cefrj1.6_c1c2.final.txt')
        cefr_loader = CEFREmbed(args.word_emb_dim, VOCABPROFILE_FILE)
    if args.filled_pauses_word and (args.filled_pauses_info == 'embed_init'):
        FLUENCYPAUSE_FILE = os.path.join(args.data_dir, 'all.filled_pauses.txt')
        filled_pauses_loader = FILLEDEmbed(args.word_emb_dim, FLUENCYPAUSE_FILE)

    # Word Embedding with or without Glove
    logger.info("Pytorch %s", torch.__version__)
    logger.info("[INFO] Create Vocab, vocab path is %s", VOCAL_FILE)
    vocab = Vocab(VOCAL_FILE, args.vocab_size)
    logger.info("[INFO] Vocab size is %s", vocab.size())
    embed = torch.nn.Embedding(vocab.size(), args.word_emb_dim, padding_idx=0)
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

    # BERT encoder
    tokenizer = None
    args.bert_config = None
    if args.bert:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model_path,
                                                  is_split_into_words=True)
        bert_config = AutoConfig.from_pretrained(args.bert_model_path, output_hidden_states=True)
        args.bert_config = bert_config
        logger.info("[INFO] Use BERT Encoder for paragraph")

    # Multiple GPUs
    args.multiple_device = True if len(args.gpu.split(',')) > 1 else False
    if args.multiple_device:
        deepspeed.init_distributed()
        args.local_rank = int(args.gpu.split(',')[0])
        logger.info("[INFO] Distributed Training")

    hps = args
    logger.info(hps)

    train_w2s_path = os.path.join(args.cache_dir, "trn_combo.{}.w2s.tfidf.jsonl".format(args.sentaspara))
    val_w2s_path = os.path.join(args.cache_dir, "dev_combo.{}.w2s.tfidf.jsonl".format(args.sentaspara))

    hps.device = setup_device(args, logger)

    if args.multiple_device:
        if hps.model == "HSG":
            model = HSumGraph(hps, embed)
            logger.info("[MODEL] HeterSumGraph ")
            dataset = ExampleSet(DATA_FILE, INTERVIEWER_DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, train_w2s_path, hps.pmi_window_width, tokenizer, hps)
            train_loader = init_dataloader(dataset=dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=graph_collate_fn)
            del dataset
            valid_dataset = ExampleSet(VALID_FILE, INTERVIEWER_VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, val_w2s_path, hps.pmi_window_width, tokenizer, hps)
            valid_loader = init_dataloader(dataset=valid_dataset, batch_size=hps.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=graph_collate_fn)
        elif hps.model == "HSPG":
            model = HSumPromptGraph(hps, embed)
            logger.info("[MODEL] HeterSumPromptGraph ")
            dataset = ExamplePromptSet(DATA_FILE, INTERVIEWER_DATA_FILE, BERT_DATA_PATH, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, train_w2s_path, hps.pmi_window_width, tokenizer, hps)
            train_loader = init_dataloader(dataset=dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=graph_collate_fn)
            del dataset
            valid_dataset = ExamplePromptSet(VALID_FILE, INTERVIEWER_VALID_FILE, BERT_VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, val_w2s_path, hps.pmi_window_width, tokenizer, hps)
            valid_loader = init_dataloader(dataset=valid_dataset, batch_size=hps.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=graph_collate_fn)
        elif hps.model == "HDSG":
            model = HSumDocGraph(hps, embed)
            logger.info("[MODEL] HeterDocSumGraph ")
            train_w2d_path = os.path.join(args.cache_dir, "trn_combo.{}.w2d.tfidf.jsonl".format(args.sentaspara))
            dataset = MultiExampleSet(DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, train_w2s_path, train_w2d_path)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=graph_collate_fn)
            del dataset
            val_w2d_path = os.path.join(args.cache_dir, "dev_combo.{}.w2d.tfidf.jsonl".format(args.sentaspara))
            valid_dataset = MultiExampleSet(VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, val_w2s_path, val_w2d_path)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hps.batch_size, shuffle=False, collate_fn=graph_collate_fn, num_workers=args.num_workers)  # Shuffle Must be False for ROUGE evaluation
        else:
            logger.error("[ERROR] Invalid Model Type!")
            raise NotImplementedError("Model Type has not been implemented")
    else:
        if hps.model == "HSG":
            model = HSumGraph(hps, embed)
            logger.info("[MODEL] HeterSumGraph ")
            dataset = ExampleSet(DATA_FILE, INTERVIEWER_DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, train_w2s_path, hps.pmi_window_width, tokenizer, hps)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=graph_collate_fn)
            del dataset
            valid_dataset = ExampleSet(VALID_FILE, INTERVIEWER_VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, val_w2s_path, hps.pmi_window_width, tokenizer, hps)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hps.batch_size, shuffle=False, collate_fn=graph_collate_fn, num_workers=args.num_workers)
        elif hps.model == "HSPG":
            model = HSumPromptGraph(hps, embed)
            logger.info("[MODEL] HSumPromptGraph ")
            dataset = ExamplePromptSet(DATA_FILE, INTERVIEWER_DATA_FILE, BERT_DATA_PATH, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, train_w2s_path, hps.pmi_window_width, tokenizer, hps)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=graph_collate_fn)
            del dataset
            valid_dataset = ExamplePromptSet(VALID_FILE, INTERVIEWER_VALID_FILE, BERT_VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, val_w2s_path, hps.pmi_window_width, tokenizer, hps)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hps.batch_size, shuffle=False, collate_fn=graph_collate_fn, num_workers=args.num_workers)
        elif hps.model == "HDSG":
            model = HSumDocGraph(hps, embed)
            logger.info("[MODEL] HeterDocSumGraph ")
            train_w2d_path = os.path.join(args.cache_dir, "trn_combo.{}.w2d.tfidf.jsonl".format(args.sentaspara))
            dataset = MultiExampleSet(DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, train_w2s_path, train_w2d_path)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=graph_collate_fn)
            del dataset
            val_w2d_path = os.path.join(args.cache_dir, "dev_combo.{}.w2d.tfidf.jsonl".format(args.sentaspara))
            valid_dataset = MultiExampleSet(VALID_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, val_w2s_path, val_w2d_path)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hps.batch_size, shuffle=False, collate_fn=graph_collate_fn, num_workers=args.num_workers)  # Shuffle Must be False for ROUGE evaluation
        else:
            logger.error("[ERROR] Invalid Model Type!")
            raise NotImplementedError("Model Type has not been implemented")

    model = model.to(hps.device)

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
                (['oe'] if hps.oe else []) + \
                (['spw'] if hps.train_speaker_wise else [])
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

    # WanDB
    # start a new wandb run to track this script
    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project=hps.save_dir_name,
            group="ddp" if args.multiple_device else "single",
            # track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "architecture": args.model,
                "dataset": "NICTJLE",
                "epochs": args.n_epochs,
            }
        )
        logger.info("[INFO] Use WANDB")

    setup_training(model, train_loader, valid_loader, valid_dataset, hps)


if __name__ == '__main__':
    main(sys.argv[1:])
