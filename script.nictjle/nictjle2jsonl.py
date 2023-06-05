#!/bin/env python
#coding:utf-8

import os
import csv
import sys
import json
import argparse
import numpy as np


def GetType(path):
    filename = path.split("/")[-2]
    return filename

def args_init():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_text_tsv_file_path', type=str, default='data/CNNDM/train.label.jsonl', help='File to deal with')
    parser.add_argument('--output_dir_path', type=str, default='CNNDM', help='dataset name')
    parser.add_argument('--sentaspara', type=str, default='sent', choices=['sent', 'para'], help='dataset name')
    parser.add_argument('--interviewer', action='store_true', default=False, help='Interviewer')
    parser.add_argument('--combine', action='store_true', default=False, help='Combine responses of nterviewer and interviewee')
    parser.add_argument('--bert', action='store_true', default=False, help='Feature used for BERT')

    
    args = parser.parse_args()
    return args

def readText(input_file, output_col_names=False, quotechar=None):
    """Reads a tab separated value file."""
    columns = []
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for i, line in enumerate(reader):
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            if i == 0:
                columns = line
                continue
            lines.append(line)
        if output_col_names:
            return columns, lines
        return lines

def parse_nictjle_content(content, interviewer=False, combine=False):
    
    # special tokens
    unk_token = '[UNK]'
    splitter_qa = '[SEP_QA]' # it is a pesudo label for distinct <A> from <B>
    splitter_pair = '[PAIR]' # it is a pesudo label for distinct pairs from each other
    newline_token = '[NEW]'
    
    token_list = content.split()
    special_tokens_list = [splitter_qa, splitter_pair, newline_token]
    
    collect_point = False if not interviewer else True
    sentence  = []
    sentences = []
    
    if combine:
        for i, t in enumerate(token_list):
            if t in special_tokens_list:
                sentences.append(' '.join(sentence))
                sentence = []
                continue
            elif t == splitter_pair:
                sentences.append(' '.join(sentence))
                sentence = []
                continue
            if t == 'grammarerrorword':
                t = unk_token
            sentence.append(t)
            if len(token_list)-1 == i:
                sentences.append(' '.join(sentence))
        return sentences
    
    for i, t in enumerate(token_list):
        if t == splitter_qa:
            if interviewer:
                sentences.append(' '.join(sentence))
                sentence = []
            collect_point = True if not interviewer else False
            continue
        elif t == splitter_pair:
            if not interviewer:
                sentences.append(' '.join(sentence))
                sentence = []
            collect_point = False if not interviewer else True
            continue
        if not collect_point:
            continue
        if t == 'grammarerrorword':
            t = unk_token
        sentence.append(t)
        if len(token_list)-1 == i:
            sentences.append(' '.join(sentence))
    return sentences

def read_nictjle_format(columns, lines, interviewer=False, combine=False):
    rtn = list()
    content_idx = columns.index('pre_token')
    label_idx   = columns.index('score')
    utt_idx  = columns.index('id')
    utt_id_idx  = columns.index('utt_index') if 'utt_index' in columns else 'X'
    speaker_idx = columns.index('speaker')
    speaker_id_idx = columns.index('speaker_index') if 'speaker_index' in columns else 'X'
    for line in lines:
        id_idx = line[utt_id_idx] if utt_id_idx != 'X' else 'X'
        speaker_id = line[speaker_id_idx] if speaker_id_idx != 'X' else 'X'
        id      = line[utt_idx]
        content = [t for t in parse_nictjle_content(line[content_idx], interviewer=interviewer, combine=combine) if t]
        label   = line[label_idx]
        speaker = line[speaker_idx]
        rtn.append({'id': id, 'id_idx': id_idx, 'label': label, 'speaker': speaker, 'speaker_id': speaker_id, 'text': content})
    return rtn

def combine_into_documents(nictjle_data_list, combine=False, bert=False):
    documents = []
    converts = {}
    converts2 = {}
    if combine and bert:
        for e in nictjle_data_list:
            speaker = e.get('speaker')
            sents   = e.get('text')
            converts.setdefault(
                speaker,
                []
            ).append(sents)
            if speaker not in converts2:
                converts2[speaker] = {'speaker_id': e.get('speaker_id'), 'label': e.get('label')}

        for speaker, document in converts.items():
            e = {}
            e.update(converts2[speaker])
            e['text'] = document
            e['speaker'] = speaker
            documents.append(e)
    else:
        for e in nictjle_data_list:
            speaker = e.get('speaker')
            sents   = e.get('text')
            converts.setdefault(
                speaker,
                []
            ).append(' '.join(sents))
            if speaker not in converts2:
                converts2[speaker] = {'speaker_id': e.get('speaker_id'), 'label': e.get('label')}

        for speaker, document in converts.items():
            e = {}
            e.update(converts2[speaker])
            e['text'] = document
            e['speaker'] = speaker
            documents.append(e)
        
    return documents


if __name__ == '__main__':
    
    args = args_init()
    input_text_tsv_file_path = args.input_text_tsv_file_path
    output_dir_path = args.output_dir_path
    sentaspara = args.sentaspara
    interviewer = args.interviewer
    combine = args.combine
    bert = args.bert
    
    if not os.path.exists(output_dir_path): os.makedirs(output_dir_path)

    columns, text_lines = readText(input_text_tsv_file_path, output_col_names=True)
    nictjle_data_list   = read_nictjle_format(columns, text_lines, interviewer=interviewer, combine=combine)
    if sentaspara == 'para':
        nictjle_data_list = combine_into_documents(nictjle_data_list, combine=combine, bert=bert)
    
    fname = GetType(input_text_tsv_file_path) + ".{}.label.jsonl".format(
        sentaspara if not interviewer else '{}.{}'.format(sentaspara, 'interviewer') \
        if not combine else '{}.{}'.format(sentaspara, 'combine')
    )
    saveFile = os.path.join(output_dir_path, fname)

    f = open(saveFile, "w")
    for e in nictjle_data_list:
        f.write(json.dumps(e) + "\n")