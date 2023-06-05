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

import numpy as np

import torch
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

import dgl

# from module.GAT import GAT, GAT_ffn
from module.Encoder import sentEncoder
from module.GAT import WSWGAT
from module.Attention import SelfAttention
from module.PositionEmbedding import get_sinusoid_encoding_table

from transformers import AutoModel

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def ruled_min_max(embed, min=0., max=6.):
    a = torch.where(embed > min, embed, torch.full_like(embed, min))
    b = torch.where(a < max, a, torch.full_like(a, max))
    return b

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.dense = nn.Linear(hps.n_feature, hps.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(hps.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class PredictionHead(nn.Module):
    '''
    A prediction head for a single objective of the SpeechGraderModel.
    Args:
        hps (Autohps): the hps for the the pre-trained BERT model
        num_labels (int): the number of labels that can be predicted
    Attributes:
        transform (transformers.modeling_bert.BertPredictionHeadTransform): a dense linear layer with gelu activation
            function
        decoder (torch.nn.Linear): a linear layer that makes predictions across the labels
        bias (torch.nn.Parameter): biases per label
    '''
    def __init__(self, hps, num_labels):
        super(PredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(hps)
        self.decoder = nn.Linear(hps.hidden_size, num_labels, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class HSumPromptGraph(nn.Module):
    """ without sent2sent and add residual connection """
    def __init__(self, hps, embed):
        """

        :param hps: 
        :param embed: word embedding
        """
        super().__init__()

        self._hps = hps
        self._n_iter = hps.n_iter
        self._embed = embed
        self.embed_size = hps.word_emb_dim

        # BERT encoder
        if hps.bert_config is not None:
            self.bert_device = torch.device("cuda", hps.bert_gpu)
            self.bert = AutoModel.from_config(hps.bert_config).to(self.bert_device)
            if hps.bert_mp:
                self.bert_pl_linear = nn.Linear(hps.bert_config.hidden_size*2, hps.bert_config.hidden_size)
                self.bert_pl_linear = self.bert_pl_linear.to(self.bert_device)

        # sent node mean
        if hps.mean_paragraphs == 'mean_residual':
            self.m_para_residual_linear = nn.Linear(hps.hidden_size * 2, hps.hidden_size)

        # sent node feature
        self._init_sn_param()
        self._TFembed = nn.Embedding(10, hps.feat_embed_size)   # box=10
        self.n_feature_proj = nn.Linear(hps.n_feature_size * 2, hps.hidden_size, bias=False)

        # word -> sent
        embed_size = hps.word_emb_dim
        gat_hidden_size = hps.hidden_size*2 if hps.interviewer else hps.hidden_size
        self.word2sent = WSWGAT(in_dim=embed_size,
                                out_dim=gat_hidden_size,
                                num_heads=hps.n_head,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="W2S"
                                )

        # sent -> word
        self.sent2word = WSWGAT(in_dim=gat_hidden_size,
                                out_dim=embed_size,
                                num_heads=6,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="S2W"
                                )
        
        if self._hps.pmi_window_width > -1:
            self.word2word = WSWGAT(in_dim=embed_size,
                                    out_dim=embed_size,
                                    num_heads=10,
                                    attn_drop_out=hps.atten_dropout_prob,
                                    ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                    ffn_drop_out=hps.ffn_dropout_prob,
                                    feat_embed_size=hps.feat_embed_size,
                                    layerType="W2W"
                                    )

        self.n_feature = hps.n_feature_size

        # sent dimension
        n_sent_dim = self.n_feature
        if hps.mean_paragraphs == 'mean':
            n_sent_dim = n_sent_dim
        elif hps.mean_paragraphs == 'mean_residual':
            n_sent_dim = n_sent_dim + hps.hidden_size

        if hps.pred_gated_fusion:
            # trainable gated weight
            if hps.bert_config is not None:
                self.bert_gt_w = nn.Linear(hps.bert_config.hidden_size + n_sent_dim, 1)
                self.down_bert = nn.Linear(hps.bert_config.hidden_size, n_sent_dim)

            self.n_feature = n_sent_dim
            hps.n_feature = n_sent_dim
            self._hps.n_feature = n_sent_dim
        else:
            final_n_dim = n_sent_dim
            if hps.bert_config is not None:
                final_n_dim = final_n_dim + hps.bert_config.hidden_size

            self.n_feature = final_n_dim
            hps.n_feature = final_n_dim
            self._hps.n_feature = final_n_dim

        if hps.test_final: # DEBUG
            self.final_attn = SelfAttention(hps.bert_config.hidden_size, n_sent_dim, use_dropout=True)
            # self.proj_final = nn.Linear(self.n_feature, n_sent_dim)

        if hps.baseline:
            self.n_feature = hps.bert_config.hidden_size

        if hps.head == 'linear':
            if hps.test_final: # DEBUG
                self.wh = nn.Linear(n_sent_dim, 6 if hps.problem_type == 'classification' else 1)
            else:
                self.wh = nn.Linear(self.n_feature, 6 if hps.problem_type == 'classification' else 1)
        elif hps.head == 'predictionhead':
            self.wh = PredictionHead(hps, 6 if hps.problem_type == 'classification' else 1)

    def forward(self, graph, graph_itvr, bert_input_ids):
        """
        :param graph: [batch_size] * DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
            edge:
                word2sent, sent2word:  tffrac=int, type=0
        :param graph_itvr: [batch_size] * DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
        :param bert_input_ids: [batch_size, max_positional_length]
        :return: result: [sentnum, 2]
        """

        # word node init
        word_feature = self.set_wnfeature(graph)    # [wnode, embed_size]

        sent_feature = self.n_feature_proj(self.set_snfeature(graph))    # [wnode, 2 * lstm_hidden_state] -> [snode, n_feature_size]
        
        # interviewer prompt as condition for the responses of interviewee
        if self._hps.interviewer:
            itvr_sent_feature = self.n_feature_proj(self.set_snfeature(graph_itvr))
            sent_feature = torch.cat((sent_feature, itvr_sent_feature), dim=1)
        
        # the start state
        word_state = word_feature
        sent_state = self.word2sent(graph, word_feature, sent_feature)

        # get baseline
        if self._hps.baseline:
            p = self._get_bert_inputs(bert_input_ids)
            result = self.wh(p)

            return result, sent_state, word_feature

        for i in range(self._n_iter):
            
            if self._hps.pmi_window_width > -1:
                # sent -> word
                word_state_from_sent = self.sent2word(graph, word_state, sent_state)
                # word -> word
                word_state_from_word = self.word2word(graph, word_state, word_state)
                word_state = word_state_from_sent + word_state_from_word
                # word -> sent
                sent_state = self.word2sent(graph, word_state, sent_state)
            else:
                # sent -> word
                word_state = self.sent2word(graph, word_state, sent_state)
                # word -> sent
                sent_state = self.word2sent(graph, word_state, sent_state)

        # update sent_state
        if self._hps.mean_paragraphs == 'mean_residual':
            mean_sent_state = self._mean_snfeature(graph, sent_state, repeat=True)
            sent_state = torch.cat((sent_state, mean_sent_state), dim=1) # add the information of self-mean
        elif self._hps.mean_paragraphs == 'mean':
            sent_state = self._mean_snfeature(graph, sent_state, repeat=True)
        else:
            sent_state = sent_state

        # BERT encoder
        if self._hps.bert_config is not None:
            p = self._get_bert_inputs(bert_input_ids)

            if self._hps.pred_gated_fusion:
                b_g_w = torch.sigmoid(self.bert_gt_w(torch.cat((sent_state, p), dim=1)))
                bert_state = b_g_w * self.down_bert(p)
            else:
                if self._hps.test_final: # DEBUG
                    b_sent_state = torch.cat((sent_state, p), dim=1)
                else:
                    sent_state = torch.cat((sent_state, p), dim=1)

        if self._hps.pred_gated_fusion:
            if self._hps.bert_config is not None:
                sent_state = sent_state + bert_state

        if self._hps.test_final and not self._hps.pred_gated_fusion: # DEBUG
            sent_state = self.final_attn(sent_state, p)

        result = self.wh(sent_state)

        return result, sent_state, word_feature

    def _init_sn_param(self):
        self.sent_pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self._hps.doc_max_timesteps + 1, self.embed_size, padding_idx=0),
            freeze=True)
        self.cnn_proj = nn.Linear(self.embed_size, self._hps.n_feature_size)
        self.lstm_hidden_state = self._hps.lstm_hidden_state
        self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden_state, num_layers=self._hps.lstm_layers, dropout=0.1,
                            batch_first=True, bidirectional=self._hps.bidirectional)
        if self._hps.bidirectional:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state * 2, self._hps.n_feature_size)
        else:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state, self._hps.n_feature_size)

        self.ngram_enc = sentEncoder(self._hps, self._embed)

    def _sent_cnn_feature(self, graph, snode_id):
        ngram_feature = self.ngram_enc.forward(graph.nodes[snode_id].data["words"])  # [snode, embed_size]
        graph.nodes[snode_id].data["sent_embedding"] = ngram_feature
        snode_pos = graph.nodes[snode_id].data["position"].view(-1)  # [n_nodes]
        position_embedding = self.sent_pos_embed(snode_pos)
        cnn_feature = self.cnn_proj(ngram_feature + position_embedding)
        return cnn_feature

    def _sent_lstm_feature(self, features, glen):
        pad_seq = rnn.pad_sequence(features, batch_first=True)
        lstm_input = rnn.pack_padded_sequence(pad_seq, glen, batch_first=True)
        lstm_output, _ = self.lstm(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        lstm_embedding = [unpacked[i][:unpacked_len[i]] for i in range(len(unpacked))]
        lstm_feature = self.lstm_proj(torch.cat(lstm_embedding, dim=0))  # [n_nodes, n_feature_size]
        return lstm_feature

    def set_wnfeature(self, graph):
        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"]==0)
        wsedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 0)   # for word to supernode(sent&doc)
        wid = graph.nodes[wnode_id].data["id"]  # [n_wnodes]
        w_embed = self._embed(wid)  # [n_wnodes, D]
        graph.nodes[wnode_id].data["embed"] = w_embed
        etf = graph.edges[wsedge_id].data["tffrac"]
        graph.edges[wsedge_id].data["tfidfembed"] = self._TFembed(etf)
        if self._hps.pmi_window_width > -1:
            wwedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 1)   # for word to word
            eww = graph.edges[wwedge_id].data["tffrac"]
            graph.edges[wwedge_id].data["tfidfembed"] = self._TFembed(eww)
        return w_embed

    def set_snfeature(self, graph):
        # node feature
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        cnn_feature = self._sent_cnn_feature(graph, snode_id)
        features, glen = get_snode_feat(graph, feat="sent_embedding")
        lstm_feature = self._sent_lstm_feature(features, glen)
        node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)  # [n_nodes, n_feature_size * 2]
        return node_feature

    def _mean_snfeature(self, graph, sent_state, repeat=False):
        repeat_cummulate_list = []
        tensors = []
        glist = dgl.unbatch(graph)
        for j in range(len(glist)):
            g = glist[j]
            snode_id = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 1)
            num_sents = len(snode_id)
            st_idx = j*num_sents
            cur_sent_state = sent_state[st_idx: st_idx + num_sents]

            sent_state_r = cur_sent_state.reshape(1, -1, self._hps.hidden_size)
            sent_state_m = torch.mean(sent_state_r, dim=1)
            repeat_cummulate_list.append(num_sents)
            tensors.append(sent_state_m)

        repeat_cummulate_list = torch.tensor(repeat_cummulate_list).to(self._hps.device)
        if repeat:
            return torch.cat(tensors, dim=0).repeat_interleave(repeat_cummulate_list, dim=0)
        return torch.cat(tensors, dim=0)
    
    def _get_bert_inputs(self, bert_input_ids):
        p = []
        for input_ids in bert_input_ids:
            input_ids = {k: v.to(self.bert_device) for k, v in input_ids.items()}
            self.bert = self.bert.to(self.bert_device)
            bert_output = self.bert(input_ids=input_ids.get('input_ids'),
                                    attention_mask=input_ids.get('attention_mask'),
                                    token_type_ids=input_ids.get('token_type_ids'))
            if self._hps.bert_mp:
                a = mean_pooling(bert_output.get('last_hidden_state'), attention_mask=input_ids['attention_mask'])
                b = bert_output.get('pooler_output')
                self.bert_pl_linear = self.bert_pl_linear.to(self.bert_device)
                c = self.bert_pl_linear(torch.cat((a, b), dim=1)).to(self._hps.device)
                p.append(c)
            else:
                p.append(bert_output.get('pooler_output').to(self._hps.device))
        return torch.cat(p, dim=0)


class HSumGraph(nn.Module):
    """ without sent2sent and add residual connection """
    def __init__(self, hps, embed):
        """

        :param hps: 
        :param embed: word embedding
        """
        super().__init__()

        self._hps = hps
        self._n_iter = hps.n_iter
        self._embed = embed
        self.embed_size = hps.word_emb_dim

        # BERT encoder
        if hps.bert_config is not None:
            self.bert_device = torch.device("cuda", hps.bert_gpu)
            self.bert = AutoModel.from_config(hps.bert_config).to(self.bert_device)

        # sent node mean
        if hps.mean_paragraphs == 'mean_residual':
            self.m_para_residual_linear = nn.Linear(hps.hidden_size * 2, hps.hidden_size)

        # sent node feature
        self._init_sn_param()
        self._TFembed = nn.Embedding(10, hps.feat_embed_size)   # box=10
        self.n_feature_proj = nn.Linear(hps.n_feature_size * 2, hps.hidden_size, bias=False)

        # word -> sent
        embed_size = hps.word_emb_dim
        self.word2sent = WSWGAT(in_dim=embed_size,
                                out_dim=hps.hidden_size,
                                num_heads=hps.n_head,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="W2S"
                                )

        # sent -> word
        self.sent2word = WSWGAT(in_dim=hps.hidden_size,
                                out_dim=embed_size,
                                num_heads=6,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="S2W"
                                )
        
        if self._hps.pmi_window_width > -1:
            self.word2word = WSWGAT(in_dim=embed_size,
                                    out_dim=embed_size,
                                    num_heads=10,
                                    attn_drop_out=hps.atten_dropout_prob,
                                    ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                    ffn_drop_out=hps.ffn_dropout_prob,
                                    feat_embed_size=hps.feat_embed_size,
                                    layerType="W2W"
                                    )

        self.n_feature = hps.hidden_size

        # sent dimension
        n_sent_dim = self.n_feature
        if hps.mean_paragraphs == 'mean':
            n_sent_dim = n_sent_dim
        elif hps.mean_paragraphs == 'mean_residual':
            n_sent_dim = n_sent_dim * 2

        # interviewer information
        if hps.interviewer:
            self.n_feature_similairty = nn.Linear(hps.hidden_size*2, n_sent_dim, bias=True) # down sampling at the same time

        if hps.pred_gated_fusion:
            # trainable gated weight
            if hps.bert_config is not None:
                self.bert_gt_w = nn.Linear(hps.bert_config.hidden_size + n_sent_dim, 1)
                self.down_bert = nn.Linear(hps.bert_config.hidden_size, n_sent_dim)
            if hps.interviewer:
                self.itvr_gt_w = nn.Linear(n_sent_dim * 2, 1)

            self.n_feature = n_sent_dim
            hps.n_feature = n_sent_dim
            self._hps.n_feature = n_sent_dim
        else:
            final_n_dim = n_sent_dim
            if hps.bert_config is not None:
                final_n_dim = final_n_dim + hps.bert_config.hidden_size
            if hps.interviewer:
                final_n_dim = final_n_dim + n_sent_dim

            self.n_feature = final_n_dim
            hps.n_feature = final_n_dim
            self._hps.n_feature = final_n_dim

        if hps.head == 'linear':
            self.wh = nn.Linear(self.n_feature, 6 if hps.problem_type == 'classification' else 1)
        elif hps.head == 'predictionhead':
            self.wh = PredictionHead(hps, 6 if hps.problem_type == 'classification' else 1)

    def forward(self, graph, graph_itvr, bert_input_ids):
        """
        :param graph: [batch_size] * DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
            edge:
                word2sent, sent2word:  tffrac=int, type=0
        :param graph_itvr: [batch_size] * DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
        :param bert_input_ids: [batch_size, max_positional_length]
        :return: result: [sentnum, 2]
        """

        # word node init
        word_feature = self.set_wnfeature(graph)    # [wnode, embed_size]

        sent_feature = self.n_feature_proj(self.set_snfeature(graph))    # [wnode, 2 * lstm_hidden_state] -> [snode, n_feature_size]
        
        # the start state
        word_state = word_feature
        sent_state = self.word2sent(graph, word_feature, sent_feature)

        for i in range(self._n_iter):
            
            if self._hps.pmi_window_width > -1:
                # sent -> word
                word_state_from_sent = self.sent2word(graph, word_state, sent_state)
                # word -> word
                word_state_from_word = self.word2word(graph, word_state, word_state)
                word_state = word_state_from_sent + word_state_from_word
                # word -> sent
                sent_state = self.word2sent(graph, word_state, sent_state)
            else:
                # sent -> word
                word_state = self.sent2word(graph, word_state, sent_state)
                # word -> sent
                sent_state = self.word2sent(graph, word_state, sent_state)

        # update sent_state
        if self._hps.mean_paragraphs == 'mean_residual':
            mean_sent_state = self._mean_snfeature(graph, sent_state, repeat=True)
            sent_state = torch.cat((sent_state, mean_sent_state), dim=1) # add the information of self-mean
        elif self._hps.mean_paragraphs == 'mean':
            sent_state = self._mean_snfeature(graph, sent_state, repeat=True)
        else:
            sent_state = sent_state

        # interviewer
        if self._hps.interviewer:
            itvr_sent_feature = self.n_feature_proj(self.set_snfeature(graph_itvr))
            itvr_set_snfeature = self.n_feature_similairty(
                torch.cat((itvr_sent_feature, sent_feature), dim=1)
            ) # similarity information via downsampling, and use the embeddings which have not enter GAT
            
            if self._hps.pred_gated_fusion:
                itvr_g_w = torch.sigmoid(self.itvr_gt_w(torch.cat((itvr_set_snfeature, sent_state), dim=1)))
                itvr_state = itvr_g_w * itvr_set_snfeature
            else:
                sent_state = torch.cat((sent_state, itvr_set_snfeature), dim=1)

        # BERT encoder
        if self._hps.bert_config is not None:
            p = self._get_bert_inputs(bert_input_ids)

            if self._hps.pred_gated_fusion:
                b_g_w = torch.sigmoid(self.bert_gt_w(torch.cat((sent_state, p), dim=1)))
                bert_state = b_g_w * self.down_bert(p)
            else:
                sent_state = torch.cat((sent_state, p), dim=1)

        if self._hps.pred_gated_fusion:
            if self._hps.interviewer:
                sent_state = sent_state + itvr_state
            if self._hps.bert_config is not None:
                sent_state = sent_state + bert_state

        result = self.wh(sent_state)

        if self._hps.oe:
            return result, sent_state

        return result

    def _init_sn_param(self):
        self.sent_pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self._hps.doc_max_timesteps + 1, self.embed_size, padding_idx=0),
            freeze=True)
        self.cnn_proj = nn.Linear(self.embed_size, self._hps.n_feature_size)
        self.lstm_hidden_state = self._hps.lstm_hidden_state
        self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden_state, num_layers=self._hps.lstm_layers, dropout=0.1,
                            batch_first=True, bidirectional=self._hps.bidirectional)
        if self._hps.bidirectional:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state * 2, self._hps.n_feature_size)
        else:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state, self._hps.n_feature_size)

        self.ngram_enc = sentEncoder(self._hps, self._embed)

    def _sent_cnn_feature(self, graph, snode_id):
        ngram_feature = self.ngram_enc.forward(graph.nodes[snode_id].data["words"])  # [snode, embed_size]
        graph.nodes[snode_id].data["sent_embedding"] = ngram_feature
        snode_pos = graph.nodes[snode_id].data["position"].view(-1)  # [n_nodes]
        position_embedding = self.sent_pos_embed(snode_pos)
        cnn_feature = self.cnn_proj(ngram_feature + position_embedding)
        return cnn_feature

    def _sent_lstm_feature(self, features, glen):
        pad_seq = rnn.pad_sequence(features, batch_first=True)
        lstm_input = rnn.pack_padded_sequence(pad_seq, glen, batch_first=True)
        lstm_output, _ = self.lstm(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        lstm_embedding = [unpacked[i][:unpacked_len[i]] for i in range(len(unpacked))]
        lstm_feature = self.lstm_proj(torch.cat(lstm_embedding, dim=0))  # [n_nodes, n_feature_size]
        return lstm_feature

    def set_wnfeature(self, graph):
        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"]==0)
        wsedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 0)   # for word to supernode(sent&doc)
        wid = graph.nodes[wnode_id].data["id"]  # [n_wnodes]
        w_embed = self._embed(wid)  # [n_wnodes, D]
        graph.nodes[wnode_id].data["embed"] = w_embed
        etf = graph.edges[wsedge_id].data["tffrac"]
        graph.edges[wsedge_id].data["tfidfembed"] = self._TFembed(etf)
        if self._hps.pmi_window_width > -1:
            wwedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 1)   # for word to word
            eww = graph.edges[wwedge_id].data["tffrac"]
            graph.edges[wwedge_id].data["tfidfembed"] = self._TFembed(eww)
        return w_embed

    def set_snfeature(self, graph):
        # node feature
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        cnn_feature = self._sent_cnn_feature(graph, snode_id)
        features, glen = get_snode_feat(graph, feat="sent_embedding")
        lstm_feature = self._sent_lstm_feature(features, glen)
        node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)  # [n_nodes, n_feature_size * 2]
        return node_feature

    def _mean_snfeature(self, graph, sent_state, repeat=False):
        repeat_cummulate_list = []
        tensors = []
        glist = dgl.unbatch(graph)
        for j in range(len(glist)):
            g = glist[j]
            snode_id = g.filter_nodes(lambda nodes: nodes.data['dtype'] == 1)
            num_sents = len(snode_id)
            st_idx = j*num_sents
            cur_sent_state = sent_state[st_idx: st_idx + num_sents]

            sent_state_r = cur_sent_state.reshape(1, -1, self._hps.hidden_size)
            sent_state_m = torch.mean(sent_state_r, dim=1)
            repeat_cummulate_list.append(num_sents)
            tensors.append(sent_state_m)

        repeat_cummulate_list = torch.tensor(repeat_cummulate_list).to(self._hps.device)
        if repeat:
            return torch.cat(tensors, dim=0).repeat_interleave(repeat_cummulate_list, dim=0)
        return torch.cat(tensors, dim=0)
    
    def _get_bert_inputs(self, bert_input_ids):
        p = []
        for input_ids in bert_input_ids:
            input_ids = {k: v.to(self.bert_device) for k, v in input_ids.items()}
            self.bert = self.bert.to(self.bert_device)
            bert_output = self.bert(input_ids=input_ids.get('input_ids'),
                                    attention_mask=input_ids.get('attention_mask'),
                                    token_type_ids=input_ids.get('token_type_ids'))
            p.append(bert_output.get('pooler_output').to(self._hps.device))
        return torch.cat(p, dim=0)
        


class HSumDocGraph(HSumGraph):
    """
        without sent2sent and add residual connection
        add Document Nodes
    """

    def __init__(self, hps, embed):
        super().__init__(hps, embed)
        self.dn_feature_proj = nn.Linear(hps.hidden_size, hps.hidden_size, bias=False)
        self.wh = nn.Linear(self.n_feature * 2, 6)

    def forward(self, graph):
        """
        :param graph: [batch_size] * DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
                document: unit=1, dtype=2
            edge:
                word2sent, sent2word: tffrac=int, type=0
                word2doc, doc2word: tffrac=int, type=0
                sent2doc: type=2
        :return: result: [sentnum, 2]
        """

        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        dnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        supernode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 1)

        # word node init
        word_feature = self.set_wnfeature(graph)    # [wnode, embed_size]
        sent_feature = self.n_feature_proj(self.set_snfeature(graph))    # [snode, n_feature_size]

        # sent and doc node init
        graph.nodes[snode_id].data["init_feature"] = sent_feature
        doc_feature, snid2dnid = self.set_dnfeature(graph)
        doc_feature = self.dn_feature_proj(doc_feature)
        graph.nodes[dnode_id].data["init_feature"] = doc_feature

        # the start state
        word_state = word_feature
        sent_state = graph.nodes[supernode_id].data["init_feature"]
        sent_state = self.word2sent(graph, word_state, sent_state)

        for i in range(self._n_iter):
            # sent -> word
            word_state = self.sent2word(graph, word_state, sent_state)
            # word -> sent
            sent_state = self.word2sent(graph, word_state, sent_state)

        graph.nodes[supernode_id].data["hidden_state"] = sent_state

        # extract sentence nodes
        s_state_list = []
        for snid in snode_id:
            d_state = graph.nodes[snid2dnid[int(snid)]].data["hidden_state"]
            s_state = graph.nodes[snid].data["hidden_state"]
            s_state = torch.cat([s_state, d_state], dim=-1)
            s_state_list.append(s_state)

        s_state = torch.cat(s_state_list, dim=0)
        result = self.wh(s_state)
        return result


    def set_dnfeature(self, graph):
        """ init doc node by mean pooling on the its sent node (connected by the edges with type=1) """
        dnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        node_feature_list = []
        snid2dnid = {}
        for dnode in dnode_id:
            snodes = [nid for nid in graph.predecessors(dnode) if graph.nodes[nid].data["dtype"]==1]
            doc_feature = graph.nodes[snodes].data["init_feature"].mean(dim=0)
            assert not torch.any(torch.isnan(doc_feature)), "doc_feature_element"
            node_feature_list.append(doc_feature)
            for s in snodes:
                snid2dnid[int(s)] = dnode
        node_feature = torch.stack(node_feature_list)
        return node_feature, snid2dnid


def get_snode_feat(G, feat):
    glist = dgl.unbatch(G)
    feature = []
    glen = []
    for g in glist:
        snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        feature.append(g.nodes[snode_id].data[feat])
        glen.append(len(snode_id))
    return feature, glen
