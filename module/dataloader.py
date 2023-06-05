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

"""This file contains code to read the train/eval/test data from file and process it, and read the vocab data from file and process it"""

import re
import os
from nltk.corpus import stopwords

import glob
import copy
import math
import random
import time
import json
import pickle
import nltk
import collections
from collections import Counter
from itertools import combinations
import numpy as np
from random import shuffle

import torch
import torch.utils.data
import torch.nn.functional as F

from tools.logger import *
from tools.utils import pikleOpen, pickleStore
from collections import Counter, defaultdict

import dgl
from dgl.data.utils import save_graphs, load_graphs

FILTERWORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
                '-', '--', '|', '\/']
FILTERWORD.extend(punctuations)


######################################### Example #########################################

class Example(object):
    """Class representing a train/val/test example for single-document extractive summarization."""

    def __init__(self, article_sents, abstract_sents, vocab, sent_max_len, label):
        """ Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

        :param article_sents: list(strings) for single document or list(list(string)) for multi-document; one per article sentence. each token is separated by a single space.
        :param abstract_sents: list(strings); one per abstract sentence. In each sentence, each token is separated by a single space.
        :param vocab: Vocabulary object
        :param sent_max_len: int, max length of each sentence
        :param label: list, the No of selected sentence, e.g. [1,3,5]
        """

        self.sent_max_len = sent_max_len
        self.enc_sent_len = []
        self.enc_sent_input = []
        self.enc_sent_input_pad = []

        # Store the original strings
        self.original_article_sents = article_sents
        self.original_abstract = "\n".join(abstract_sents)

        # Process the article
        if isinstance(article_sents, list) and isinstance(article_sents[0], list):  # multi document
            self.original_article_sents = []
            for doc in article_sents:
                self.original_article_sents.extend(doc)
        for sent in self.original_article_sents:
            article_words = sent.split()
            self.enc_sent_len.append(len(article_words))  # store the length before padding
            self.enc_sent_input.append([vocab.word2id(w.lower()) for w in article_words])  # list of word ids; OOVs are represented by the id for UNK token
        self._pad_encoder_input(vocab.word2id('[PAD]'))

        # Store the label
        self.label = label
        label_shape = (len(self.original_article_sents), 6)  # [N, 6] due to A1 to C2 (1-6) in CEFR
        label_index = int(label) - 1
        self.label_matrix = np.zeros(label_shape, dtype=int)
        if label:
            self.label_matrix[:, label_index] = 1  # label_matrix[i][j]=1 indicate the i-th sent will be selected in j-th step

    def _pad_encoder_input(self, pad_id):
        """
        :param pad_id: int; token pad id
        :return: 
        """
        max_len = self.sent_max_len
        for i in range(len(self.enc_sent_input)):
            article_words = self.enc_sent_input[i].copy()
            if len(article_words) > max_len:
                article_words = article_words[:max_len]
            if len(article_words) < max_len:
                article_words.extend([pad_id] * (max_len - len(article_words)))
            self.enc_sent_input_pad.append(article_words)


class Example2(Example):
    """Class representing a train/val/test example for multi-document extractive summarization."""

    def __init__(self, article_sents, abstract_sents, vocab, sent_max_len, label):
        """ Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

        :param article_sents: list(list(string)) for multi-document; one per article sentence. each token is separated by a single space.
        :param abstract_sents: list(strings); one per abstract sentence. In each sentence, each token is separated by a single space.
        :param vocab: Vocabulary object
        :param sent_max_len: int, max length of each sentence
        :param label: list, the No of selected sentence, e.g. [1,3,5]
        """

        super().__init__(article_sents, abstract_sents, vocab, sent_max_len, label)
        cur = 0
        self.original_articles = []
        self.article_len = []
        self.enc_doc_input = []
        for doc in article_sents:
            if len(doc) == 0:
                continue
            docLen = len(doc)
            self.original_articles.append(" ".join(doc))
            self.article_len.append(docLen)
            self.enc_doc_input.append(catDoc(self.enc_sent_input[cur:cur + docLen]))
            cur += docLen


######################################### ExampleSet #########################################

class ExamplePromptSet(torch.utils.data.Dataset):
    """ Constructor: Dataset of example(object) for single document summarization"""

    def __init__(self, data_path, interviewer_data_path, bert_data_path, vocab, doc_max_timesteps, sent_max_len, filter_word_path, w2s_path, pmi_window_width, tokenizer, hps):
        """ Initializes the ExampleSet with the path of data
        
        :param data_path: string; the path of data
        :param interviewer_data_path: string; the path of data from interviewer
        :param bert_data_path: string; the path of data from both interviewer and interviewee
        :param vocab: object;
        :param doc_max_timesteps: int; the maximum sentence number of a document, each example should pad sentences to this length
        :param sent_max_len: int; the maximum token number of a sentence, each sentence should pad tokens to this length
        :param filter_word_path: str; file path, the file must contain one word for each line and the tfidf value must go from low to high (the format can refer to script/lowTFIDFWords.py) 
        :param w2s_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2sTFIDF.py)
        :param pmi_window_width: widow with when calculatin NPMI
        :param tokenizer: WordPeice Tokenizer
        :param hps: parameters
        """

        self.hps = hps
        self.vocab = vocab
        self.sent_max_len = sent_max_len
        self.doc_max_timesteps = doc_max_timesteps

        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        start = time.time()
        self.example_list = readJson(data_path)
        if bert_data_path:
            self.bert_example_list = readJson(bert_data_path)
        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.example_list))
        if interviewer_data_path is not None:
            self.interviewer_example_list = readJson(interviewer_data_path)
            logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                        time.time() - start, len(self.interviewer_example_list))
        self.size = len(self.example_list)
        self.data_set = data_path.split('/')[-1].split('.')[0]
        self.doc_pmi_dir = os.path.join(self.hps.cache_dir, 'pmi')
        if not os.path.exists(self.doc_pmi_dir):
            os.makedirs(self.doc_pmi_dir)
        self.G_dir = os.path.join(self.hps.cache_dir, 'G')
        if not os.path.exists(self.G_dir):
            os.makedirs(self.G_dir)
            
        logger.info("[INFO] Loading filter word File %s", filter_word_path)
        tfidf_w = readText(filter_word_path)
        self.filterwords = FILTERWORD
        self.filterids = [vocab.word2id(w.lower()) for w in FILTERWORD]
        self.filterids.append(vocab.word2id("[PAD]"))   # keep "[UNK]" but remove "[PAD]"
        lowtfidf_num = 0
        pattern = r"^[0-9]+$"
        for w in tfidf_w:
            if vocab.word2id(w) != vocab.word2id('[UNK]'):
                self.filterwords.append(w)
                self.filterids.append(vocab.word2id(w))
                # if re.search(pattern, w) == None:  # if w is a number, it will not increase the lowtfidf_num
                    # lowtfidf_num += 1
                lowtfidf_num += 1
            if lowtfidf_num > 5000:
                break

        logger.info("[INFO] Loading word2sent TFIDF file from %s!" % w2s_path)
        self.w2s_tfidf = readJson(w2s_path)
        
        self.pmi_window_width = pmi_window_width
        if pmi_window_width > -1:
            logger.info("[INFO] Use N-PMI!")
        
        self.tokenizer = tokenizer

    def get_example(self, index):
        e = self.example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"])
        return example

    def get_bert_example(self, index):
        e = self.bert_example_list[index]
        t = e.get('text')
        f = []
        for p in t:
            ps = [self.tokenizer.cls_token]
            for s in p:
                for w in s.split():
                    wp = self.tokenizer.tokenize(w)
                    if len(wp) > 1:
                        wp = [self.tokenizer.unk_token]
                    ps.extend(wp)
                ps.append(self.tokenizer.sep_token)
            f.append(' '.join(ps[1:-1]))
        f = [self.tokenizer(f, padding="max_length", truncation=True)]
        return f

    def get_interviewer_example(self, index):
        e = self.interviewer_example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"])
        return example

    def pad_label_m(self, label_matrix):
        label_m = label_matrix[:self.doc_max_timesteps, :self.doc_max_timesteps]
        # N, m = label_m.shape
        # if m < self.doc_max_timesteps:
        #     pad_m = np.zeros((N, self.doc_max_timesteps - m))
        #     return np.hstack([label_m, pad_m])
        return label_m

    def AddWordNode(self, G, inputid):
        wid2nid = {}
        nid2wid = {}
        nid = 0
        for sentid in inputid:
            for wid in sentid:
                if wid not in self.filterids and wid not in wid2nid.keys():
                    wid2nid[wid] = nid
                    nid2wid[nid] = wid
                    nid += 1

        w_nodes = len(nid2wid)

        G.add_nodes(w_nodes)
        G.set_n_initializer(dgl.init.zero_initializer)
        G.ndata["unit"] = torch.zeros(w_nodes)
        G.ndata["id"] = torch.LongTensor(list(nid2wid.values()))
        G.ndata["dtype"] = torch.zeros(w_nodes)

        return wid2nid, nid2wid

    def AddCEFRNode(self, G, inputid):
        return

    def CreateGraph(self, input_pad, label, w2s_w, w2w_pmi_info):
        """ Create a graph for each document
        
        :param input_pad: list(list); [sentnum, wordnum]
        :param label: list(list); [sentnum, sentnum]
        :param w2s_w: dict(dict) {str: {str: float}}; for each sentence and each word, the tfidf between them
        :param w2w_pmi_info: dict(dict) {str: {str: int}}; for each word and each word, the n-pmi between them
        :return: G: dgl.DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
            edge:
                word2sent, sent2word:  tffrac=int, dtype=0
                word2word:             tffrac=int, dtype=1
        """
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, input_pad)
        w_nodes = len(nid2wid)

        N = len(input_pad)
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]

        G.set_e_initializer(dgl.init.zero_initializer)
        if self.pmi_window_width > -1:
            max_pmi = w2w_pmi_info.get('max_pmi')
            pmi_mat = w2w_pmi_info.get('pmi')
            for i in range(N):
                c = Counter(input_pad[i])
                sent_nid = sentid2nid[i]
                sent_tfw = w2s_w[str(i)]
                for s_wid in c.keys():
                    if s_wid in wid2nid.keys() and self.vocab.id2word(s_wid) in sent_tfw.keys():
                        for t_wid in c.keys():
                            if t_wid in wid2nid.keys() and self.vocab.id2word(t_wid) in sent_tfw.keys():
                                s2t = pmi_mat[self.vocab.id2word(s_wid)][self.vocab.id2word(t_wid)] / max_pmi
                                t2s = pmi_mat[self.vocab.id2word(t_wid)][self.vocab.id2word(s_wid)] / max_pmi
                                s2t = np.round(s2t * 9)
                                t2s = np.round(t2s * 9)
                                G.add_edges(wid2nid[s_wid], wid2nid[t_wid],
                                            data={"tffrac": torch.LongTensor([s2t]), "dtype": torch.Tensor([1])})
                                G.add_edges(wid2nid[t_wid], wid2nid[s_wid],
                                            data={"tffrac": torch.LongTensor([t2s]), "dtype": torch.Tensor([1])})
        for i in range(N):
            c = Counter(input_pad[i])
            sent_nid = sentid2nid[i]
            sent_tfw = w2s_w[str(i)]
            for wid in c.keys():
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw.keys():
                    tfidf = sent_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    G.add_edges(wid2nid[wid], sent_nid,
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edges(sent_nid, wid2nid[wid],
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
            
            # The two lines can be commented out if you use the code for your own training, since HSG does not use sent2sent edges. 
            # However, if you want to use the released checkpoint directly, please leave them here.
            # Otherwise it may cause some parameter corresponding errors due to the version differences.
            G.add_edges(sent_nid, sentid2nid, data={"dtype": torch.ones(N)})
            G.add_edges(sentid2nid, sent_nid, data={"dtype": torch.ones(N)})
        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]
        G.nodes[sentid2nid].data["label"] = torch.LongTensor(label)  # [N, doc_max]

        return G

    def CreateItvrGraph(self, input_pad):
        """ Create a graph for each document
        
        :param input_pad: list(list); [sentnum, wordnum]
        :param label: list(list); [sentnum, sentnum]
        :param w2s_w: dict(dict) {str: {str: float}}; for each sentence and each word, the tfidf between them
        :param w2w_pmi_info: dict(dict) {str: {str: int}}; for each word and each word, the n-pmi between them
        :return: G: dgl.DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
        """
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, input_pad)
        w_nodes = len(nid2wid)

        N = len(input_pad)
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]
        
        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]

        return G

    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return 
            G: graph for the example
            index: int; the index of the example in the dataset
        """
        item = self.get_example(index)
        input_pad = item.enc_sent_input_pad[:self.doc_max_timesteps]
        if self.tokenizer is not None:
            bert_input_ids = self.get_bert_example(index)
            bert_input_ids = self._merge_input_ids(bert_input_ids)
            for k, v in bert_input_ids.items():
                assert v.shape[0] == len(input_pad), 'Maybe try to add up the value of doc_max_timesteps arguments'
        label = self.pad_label_m(item.label_matrix)
        w2s_w = self.w2s_tfidf[index]

        doc_pmi_file_name = '.'.join(list(filter(None, 
            [self.data_set, str(index), ('pmi{}'.format(self.pmi_window_width) if self.pmi_window_width > -1 else ''), ('itvr' if self.hps.interviewer else None),
            ('cefrbd' if self.hps.cefr_word and (self.hps.cefr_info == 'embed_init') else ''), ('fpbd' if self.hps.filled_pauses_word and (self.hps.filled_pauses_info == 'embed_init') else '')
            ]
            ))
        )
        doc_pmi_path = os.path.join(self.doc_pmi_dir, doc_pmi_file_name)
        if os.path.exists(doc_pmi_path):
            w2w_pmi_info = pikleOpen(doc_pmi_path)
        else:
            w2w_pmi_info = self._calculate_pmi(item.original_article_sents, pmi_window_width=self.pmi_window_width)
            pickleStore(w2w_pmi_info, doc_pmi_path)

        G_file_name = '.'.join(list(filter(None,
            [self.data_set, str(index), ('pmi{}'.format(self.pmi_window_width) if self.pmi_window_width > -1 else ''), ('itvr' if self.hps.interviewer else None),
            ('cefrbd' if self.hps.cefr_word and (self.hps.cefr_info == 'embed_init') else ''), ('fpbd' if self.hps.filled_pauses_word and (self.hps.filled_pauses_info == 'embed_init') else '')
            ]
            ))
        )
        G_path = os.path.join(self.G_dir, G_file_name)
        if os.path.exists(G_path):
            G = pikleOpen(G_path)
        else:
            G = self.CreateGraph(input_pad, label, w2s_w, w2w_pmi_info)
            pickleStore(G, G_path)

        # interviewer
        itvr_G = None
        if self.hps.interviewer:
            itvr_item = self.get_interviewer_example(index)
            itvr_input_pad = itvr_item.enc_sent_input_pad[:self.doc_max_timesteps]
            assert len(itvr_input_pad) == len(input_pad), 'Problems in interviewers inputs, {} and {}'.format(len(itvr_input_pad), len(input_pad))
            itvr_G = self.CreateItvrGraph(itvr_input_pad)

        if self.tokenizer is not None:
            return G, index, bert_input_ids, itvr_G
        return G, index, None, itvr_G

    def _merge_input_ids(self, input_ids_list):
        rtn = {}
        for input_ids in input_ids_list:
            for k, v in input_ids.items():
                rtn.setdefault(
                    k,
                    []
                ).append(v)
        rtn = {k:torch.LongTensor(v_list).squeeze(0) for k, v_list in rtn.items()}
        return rtn

    def _iter_ngrams(self, words, n):
        """Iterate over all word n-grams in a list."""
        if len(words) < n:
            yield words

        for i in range(len(words) - n + 1):
            yield words[i:i+n]

    def _calculate_pmi(self, sents, pmi_window_width=2):
        doc_text = ' '.join(sents)
        word_counts = Counter()
        cooccur_counts = defaultdict(Counter)
        pmi = defaultdict(Counter)
        max_pmi = 0.
        for ngram in self._iter_ngrams(doc_text.split(), n=pmi_window_width):
            for i, src_word in enumerate(ngram):
                if src_word not in self.vocab.word_list():    # OOV
                    continue
                word_counts[src_word] += 1
                for j, tgt_word in enumerate(ngram):
                    if i != j and tgt_word != 0:
                        cooccur_counts[src_word][tgt_word] += 1

        log_total_counts = math.log(sum(word_counts.values()))
        for src_word, tgt_word_counts in cooccur_counts.items():
            for tgt_word, counts in tgt_word_counts.items():
                unconstrained_pmi = log_total_counts + math.log(counts)
                unconstrained_pmi -= math.log(word_counts[src_word] *
                                                word_counts[tgt_word])
                if unconstrained_pmi > 0.:
                    pmi[src_word][tgt_word] = unconstrained_pmi
                    max_pmi = max(max_pmi, unconstrained_pmi)
        return {'pmi': pmi, 'max_pmi': max_pmi}

    def __len__(self):
        return self.size


class ExampleSet(torch.utils.data.Dataset):
    """ Constructor: Dataset of example(object) for single document summarization"""

    def __init__(self, data_path, interviewer_data_path, vocab, doc_max_timesteps, sent_max_len, filter_word_path, w2s_path, pmi_window_width, tokenizer, hps):
        """ Initializes the ExampleSet with the path of data
        
        :param data_path: string; the path of data
        :param interviewer_data_path: string; the path of data from interviewer
        :param vocab: object;
        :param doc_max_timesteps: int; the maximum sentence number of a document, each example should pad sentences to this length
        :param sent_max_len: int; the maximum token number of a sentence, each sentence should pad tokens to this length
        :param filter_word_path: str; file path, the file must contain one word for each line and the tfidf value must go from low to high (the format can refer to script/lowTFIDFWords.py) 
        :param w2s_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2sTFIDF.py)
        :param pmi_window_width: widow with when calculatin NPMI
        :param tokenizer: WordPeice Tokenizer
        :param hps: parameters
        """

        self.hps = hps
        self.vocab = vocab
        self.sent_max_len = sent_max_len
        self.doc_max_timesteps = doc_max_timesteps

        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        start = time.time()
        self.example_list = readJson(data_path)
        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.example_list))
        if interviewer_data_path is not None:
            self.interviewer_example_list = readJson(interviewer_data_path)
            logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                        time.time() - start, len(self.interviewer_example_list))
        self.size = len(self.example_list)
        self.data_set = data_path.split('/')[-1].split('.')[0]
        self.doc_pmi_dir = os.path.join(self.hps.cache_dir, 'pmi')
        if not os.path.exists(self.doc_pmi_dir):
            os.makedirs(self.doc_pmi_dir)
        self.G_dir = os.path.join(self.hps.cache_dir, 'G')
        if not os.path.exists(self.G_dir):
            os.makedirs(self.G_dir)
            
        logger.info("[INFO] Loading filter word File %s", filter_word_path)
        tfidf_w = readText(filter_word_path)
        self.filterwords = FILTERWORD
        self.filterids = [vocab.word2id(w.lower()) for w in FILTERWORD]
        self.filterids.append(vocab.word2id("[PAD]"))   # keep "[UNK]" but remove "[PAD]"
        lowtfidf_num = 0
        pattern = r"^[0-9]+$"
        for w in tfidf_w:
            if vocab.word2id(w) != vocab.word2id('[UNK]'):
                self.filterwords.append(w)
                self.filterids.append(vocab.word2id(w))
                # if re.search(pattern, w) == None:  # if w is a number, it will not increase the lowtfidf_num
                    # lowtfidf_num += 1
                lowtfidf_num += 1
            if lowtfidf_num > 5000:
                break

        logger.info("[INFO] Loading word2sent TFIDF file from %s!" % w2s_path)
        self.w2s_tfidf = readJson(w2s_path)
        
        self.pmi_window_width = pmi_window_width
        if pmi_window_width > -1:
            logger.info("[INFO] Use N-PMI!")
        
        self.tokenizer = tokenizer

    def get_example(self, index):
        e = self.example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"])
        return example

    def get_bert_example(self, index):
        e = self.example_list[index]
        t = e.get('text')
        f = []
        for p in t:
            s = []
            for w in p.split():
                wp = self.tokenizer.tokenize(w)
                if len(wp) > 1:
                    wp = [self.tokenizer.unk_token]
                s.extend(wp)
            f.append(self.tokenizer(' '.join(s), padding="max_length", truncation=True))
        return f

    def get_interviewer_example(self, index):
        e = self.interviewer_example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"])
        return example

    def pad_label_m(self, label_matrix):
        label_m = label_matrix[:self.doc_max_timesteps, :self.doc_max_timesteps]
        # N, m = label_m.shape
        # if m < self.doc_max_timesteps:
        #     pad_m = np.zeros((N, self.doc_max_timesteps - m))
        #     return np.hstack([label_m, pad_m])
        return label_m

    def AddWordNode(self, G, inputid):
        wid2nid = {}
        nid2wid = {}
        nid = 0
        for sentid in inputid:
            for wid in sentid:
                if wid not in self.filterids and wid not in wid2nid.keys():
                    wid2nid[wid] = nid
                    nid2wid[nid] = wid
                    nid += 1

        w_nodes = len(nid2wid)

        G.add_nodes(w_nodes)
        G.set_n_initializer(dgl.init.zero_initializer)
        G.ndata["unit"] = torch.zeros(w_nodes)
        G.ndata["id"] = torch.LongTensor(list(nid2wid.values()))
        G.ndata["dtype"] = torch.zeros(w_nodes)

        return wid2nid, nid2wid

    def AddCEFRNode(self, G, inputid):
        return

    def CreateGraph(self, input_pad, label, w2s_w, w2w_pmi_info):
        """ Create a graph for each document
        
        :param input_pad: list(list); [sentnum, wordnum]
        :param label: list(list); [sentnum, sentnum]
        :param w2s_w: dict(dict) {str: {str: float}}; for each sentence and each word, the tfidf between them
        :param w2w_pmi_info: dict(dict) {str: {str: int}}; for each word and each word, the n-pmi between them
        :return: G: dgl.DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
            edge:
                word2sent, sent2word:  tffrac=int, dtype=0
                word2word:             tffrac=int, dtype=1
        """
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, input_pad)
        w_nodes = len(nid2wid)

        N = len(input_pad)
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]

        G.set_e_initializer(dgl.init.zero_initializer)
        if self.pmi_window_width > -1:
            max_pmi = w2w_pmi_info.get('max_pmi')
            pmi_mat = w2w_pmi_info.get('pmi')
            for i in range(N):
                c = Counter(input_pad[i])
                sent_nid = sentid2nid[i]
                sent_tfw = w2s_w[str(i)]
                for s_wid in c.keys():
                    if s_wid in wid2nid.keys() and self.vocab.id2word(s_wid) in sent_tfw.keys():
                        for t_wid in c.keys():
                            if t_wid in wid2nid.keys() and self.vocab.id2word(t_wid) in sent_tfw.keys():
                                s2t = pmi_mat[self.vocab.id2word(s_wid)][self.vocab.id2word(t_wid)] / max_pmi
                                t2s = pmi_mat[self.vocab.id2word(t_wid)][self.vocab.id2word(s_wid)] / max_pmi
                                s2t = np.round(s2t * 9)
                                t2s = np.round(t2s * 9)
                                G.add_edges(wid2nid[s_wid], wid2nid[t_wid],
                                            data={"tffrac": torch.LongTensor([s2t]), "dtype": torch.Tensor([1])})
                                G.add_edges(wid2nid[t_wid], wid2nid[s_wid],
                                            data={"tffrac": torch.LongTensor([t2s]), "dtype": torch.Tensor([1])})
        for i in range(N):
            c = Counter(input_pad[i])
            sent_nid = sentid2nid[i]
            sent_tfw = w2s_w[str(i)]
            for wid in c.keys():
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw.keys():
                    tfidf = sent_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    G.add_edges(wid2nid[wid], sent_nid,
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edges(sent_nid, wid2nid[wid],
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
            
            # The two lines can be commented out if you use the code for your own training, since HSG does not use sent2sent edges. 
            # However, if you want to use the released checkpoint directly, please leave them here.
            # Otherwise it may cause some parameter corresponding errors due to the version differences.
            G.add_edges(sent_nid, sentid2nid, data={"dtype": torch.ones(N)})
            G.add_edges(sentid2nid, sent_nid, data={"dtype": torch.ones(N)})
        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]
        G.nodes[sentid2nid].data["label"] = torch.LongTensor(label)  # [N, doc_max]

        return G

    def CreateItvrGraph(self, input_pad):
        """ Create a graph for each document
        
        :param input_pad: list(list); [sentnum, wordnum]
        :param label: list(list); [sentnum, sentnum]
        :param w2s_w: dict(dict) {str: {str: float}}; for each sentence and each word, the tfidf between them
        :param w2w_pmi_info: dict(dict) {str: {str: int}}; for each word and each word, the n-pmi between them
        :return: G: dgl.DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
        """
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, input_pad)
        w_nodes = len(nid2wid)

        N = len(input_pad)
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]
        
        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]

        return G

    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return 
            G: graph for the example
            index: int; the index of the example in the dataset
        """
        item = self.get_example(index)            
        input_pad = item.enc_sent_input_pad[:self.doc_max_timesteps]
        if self.tokenizer is not None:
            bert_input_ids = self.get_bert_example(index)
            bert_input_ids = self._merge_input_ids(bert_input_ids)
            for k, v in bert_input_ids.items():
                assert v.shape[0] == len(input_pad), 'Maybe try to add up the value of doc_max_timesteps arguments'
        label = self.pad_label_m(item.label_matrix)
        w2s_w = self.w2s_tfidf[index]

        doc_pmi_file_name = '.'.join(list(filter(None, 
            [self.data_set, str(index), ('pmi{}'.format(self.pmi_window_width) if self.pmi_window_width > -1 else ''), ('itvr' if self.hps.interviewer else None),
            ('cefrbd' if self.hps.cefr_word and (self.hps.cefr_info == 'embed_init') else ''), ('fpbd' if self.hps.filled_pauses_word and (self.hps.filled_pauses_info == 'embed_init') else '')
            ]
            ))
        )
        doc_pmi_path = os.path.join(self.doc_pmi_dir, doc_pmi_file_name)
        if os.path.exists(doc_pmi_path):
            w2w_pmi_info = pikleOpen(doc_pmi_path)
        else:
            w2w_pmi_info = self._calculate_pmi(item.original_article_sents, pmi_window_width=self.pmi_window_width)
            pickleStore(w2w_pmi_info, doc_pmi_path)

        G_file_name = '.'.join(list(filter(None,
            [self.data_set, str(index), ('pmi{}'.format(self.pmi_window_width) if self.pmi_window_width > -1 else ''), ('itvr' if self.hps.interviewer else None),
            ('cefrbd' if self.hps.cefr_word and (self.hps.cefr_info == 'embed_init') else ''), ('fpbd' if self.hps.filled_pauses_word and (self.hps.filled_pauses_info == 'embed_init') else '')
            ]
            ))
        )
        G_path = os.path.join(self.G_dir, G_file_name)
        if os.path.exists(G_path):
            G = pikleOpen(G_path)
        else:
            G = self.CreateGraph(input_pad, label, w2s_w, w2w_pmi_info)
            pickleStore(G, G_path)

        # interviewer
        itvr_G = None
        if self.hps.interviewer:
            itvr_item = self.get_interviewer_example(index)
            itvr_input_pad = itvr_item.enc_sent_input_pad[:self.doc_max_timesteps]
            # check_len = abs(len(itvr_input_pad) < len(input_pad))
            # if check_len:
            #     if len(itvr_input_pad) < len(input_pad):
            #         for i in range(0, check_len):
            #             itvr_input_pad.append(input_pad[len(input_pad)+i])
            #     else:
            #         itvr_input_pad = itvr_input_pad[:len(input_pad)]
            assert len(itvr_input_pad) == len(input_pad), 'Problems in interviewers inputs, {} and {}'.format(len(itvr_input_pad), len(input_pad))
            itvr_G = self.CreateItvrGraph(itvr_input_pad)

        if self.tokenizer is not None:
            return G, index, bert_input_ids, itvr_G
        return G, index, None, itvr_G

    def _merge_input_ids(self, input_ids_list):
        rtn = {}
        for input_ids in input_ids_list:
            for k, v in input_ids.items():
                rtn.setdefault(
                    k,
                    []
                ).append(v)
        rtn = {k:torch.LongTensor(v_list) for k, v_list in rtn.items()}
        return rtn

    def _iter_ngrams(self, words, n):
        """Iterate over all word n-grams in a list."""
        if len(words) < n:
            yield words

        for i in range(len(words) - n + 1):
            yield words[i:i+n]

    def _calculate_pmi(self, sents, pmi_window_width=2):
        doc_text = ' '.join(sents)
        word_counts = Counter()
        cooccur_counts = defaultdict(Counter)
        pmi = defaultdict(Counter)
        max_pmi = 0.
        for ngram in self._iter_ngrams(doc_text.split(), n=pmi_window_width):
            for i, src_word in enumerate(ngram):
                if src_word not in self.vocab.word_list():    # OOV
                    continue
                word_counts[src_word] += 1
                for j, tgt_word in enumerate(ngram):
                    if i != j and tgt_word != 0:
                        cooccur_counts[src_word][tgt_word] += 1

        log_total_counts = math.log(sum(word_counts.values()))
        for src_word, tgt_word_counts in cooccur_counts.items():
            for tgt_word, counts in tgt_word_counts.items():
                unconstrained_pmi = log_total_counts + math.log(counts)
                unconstrained_pmi -= math.log(word_counts[src_word] *
                                                word_counts[tgt_word])
                if unconstrained_pmi > 0.:
                    pmi[src_word][tgt_word] = unconstrained_pmi
                    max_pmi = max(max_pmi, unconstrained_pmi)
        return {'pmi': pmi, 'max_pmi': max_pmi}

    def __len__(self):
        return self.size


class MultiExampleSet(ExampleSet):
    """ Constructor: Dataset of example(object) for multiple document summarization"""
    def __init__(self, data_path, vocab, doc_max_timesteps, sent_max_len, filter_word_path, w2s_path, w2d_path):
        """ Initializes the ExampleSet with the path of data

        :param data_path: string; the path of data
        :param vocab: object;
        :param doc_max_timesteps: int; the maximum sentence number of a document, each example should pad sentences to this length
        :param sent_max_len: int; the maximum token number of a sentence, each sentence should pad tokens to this length
        :param filter_word_path: str; file path, the file must contain one word for each line and the tfidf value must go from low to high (the format can refer to script/lowTFIDFWords.py) 
        :param w2s_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2sTFIDF.py)
        :param w2d_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2dTFIDF.py)
        """

        super().__init__(data_path, vocab, doc_max_timesteps, sent_max_len, filter_word_path, w2s_path)

        logger.info("[INFO] Loading word2doc TFIDF file from %s!" % w2d_path)
        self.w2d_tfidf = readJson(w2d_path)

    def get_example(self, index):
        e = self.example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example2(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"])
        return example

    def MapSent2Doc(self, article_len, sentNum):
        sent2doc = {}
        doc2sent = {}
        sentNo = 0
        for i in range(len(article_len)):
            doc2sent[i] = []
            for j in range(article_len[i]):
                sent2doc[sentNo] = i
                doc2sent[i].append(sentNo)
                sentNo += 1
                if sentNo >= sentNum:
                    return sent2doc
        return sent2doc

    def CreateGraph(self, docLen, sent_pad, doc_pad, label, w2s_w, w2d_w):
        """ Create a graph for each document

        :param docLen: list; the length of each document in this example
        :param sent_pad: list(list), [sentnum, wordnum]
        :param doc_pad: list, [document, wordnum]
        :param label: list(list), [sentnum, sentnum]
        :param w2s_w: dict(dict) {str: {str: float}}, for each sentence and each word, the tfidf between them
        :param w2d_w: dict(dict) {str: {str: float}}, for each document and each word, the tfidf between them
        :return: G: dgl.DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
                document: unit=1, dtype=2
            edge:
                word2sent, sent2word: tffrac=int, dtype=0
                word2doc, doc2word: tffrac=int, dtype=0
                sent2doc: dtype=2
        """
        # add word nodes
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, sent_pad)
        w_nodes = len(nid2wid)

        # add sent nodes
        N = len(sent_pad)
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]
        ws_nodes = w_nodes + N

        # add doc nodes
        sent2doc = self.MapSent2Doc(docLen, N)
        article_num = len(set(sent2doc.values()))
        G.add_nodes(article_num)
        G.ndata["unit"][ws_nodes:] = torch.ones(article_num)
        G.ndata["dtype"][ws_nodes:] = torch.ones(article_num) * 2
        docid2nid = [i + ws_nodes for i in range(article_num)]

        # add sent edges
        for i in range(N):
            c = Counter(sent_pad[i])
            sent_nid = sentid2nid[i]
            sent_tfw = w2s_w[str(i)]
            for wid, cnt in c.items():
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw.keys():
                    tfidf = sent_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    # w2s s2w
                    G.add_edge(wid2nid[wid], sent_nid,
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edge(sent_nid, wid2nid[wid],
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
            # s2d
            docid = sent2doc[i]
            docnid = docid2nid[docid]
            G.add_edge(sent_nid, docnid, data={"tffrac": torch.LongTensor([0]), "dtype": torch.Tensor([2])})

        # add doc edges
        for i in range(article_num):
            c = Counter(doc_pad[i])
            doc_nid = docid2nid[i]
            doc_tfw = w2d_w[str(i)]
            for wid, cnt in c.items():
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in doc_tfw.keys():
                    # w2d d2w
                    tfidf = doc_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    G.add_edge(wid2nid[wid], doc_nid,
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edge(doc_nid, wid2nid[wid],
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})

        G.nodes[sentid2nid].data["words"] = torch.LongTensor(sent_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]
        G.nodes[sentid2nid].data["label"] = torch.LongTensor(label)  # [N, doc_max]

        return G

    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return 
            G: graph for the example
            index: int; the index of the example in the dataset
        """
        item = self.get_example(index)
        sent_pad = item.enc_sent_input_pad[:self.doc_max_timesteps]
        enc_doc_input = item.enc_doc_input
        article_len = item.article_len
        label = self.pad_label_m(item.label_matrix)

        G = self.CreateGraph(article_len, sent_pad, enc_doc_input, label, self.w2s_tfidf[index], self.w2d_tfidf[index])

        return G, index


class LoadHiExampleSet(torch.utils.data.Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        self.gfiles = [f for f in os.listdir(self.data_root) if f.endswith("graph.bin")]
        logger.info("[INFO] Start loading %s", self.data_root)

    def __getitem__(self, index):
        graph_file = os.path.join(self.data_root, "%d.graph.bin" % index)
        g, label_dict = load_graphs(graph_file)
        # print(graph_file)
        return g[0], index

    def __len__(self):
        return len(self.gfiles)


######################################### Tools #########################################


import dgl


def catDoc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res


def readJson(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def readText(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data


def graph_collate_fn(samples):
    '''
    :param batch: (G, input_pad)
    :return: 
    '''
    graphs, index, input_ids, itvr_graphs = map(list, zip(*samples))
    graph_len = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graphs]  # sent node of graph
    sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
    batched_bert_input_ids = None
    if input_ids is not None:
        batched_bert_input_ids = [input_ids[idx.item()] for idx in sorted_index]
        assert len(batched_bert_input_ids) == len(input_ids), '{} _ {}'.format(len(batched_bert_input_ids), len(input_ids))
    batched_graph = dgl.batch([graphs[idx] for idx in sorted_index]) 
    batched_itvr_graph = dgl.batch([itvr_graphs[idx] for idx in sorted_index]) if list(filter(None, itvr_graphs)) else None
    return batched_graph, [index[idx] for idx in sorted_index], batched_bert_input_ids, batched_itvr_graph
