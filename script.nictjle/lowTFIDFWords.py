#!/bin/env python
#coding:utf-8

import os
import json
import argparse
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


def catDoc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res

def calTFidf(text):
    vectorizer = CountVectorizer(lowercase=True)
    wordcount = vectorizer.fit_transform(text)
    tf_idf_transformer = TfidfTransformer()
    tfidf_matrix = tf_idf_transformer.fit_transform(wordcount)
    return vectorizer, tfidf_matrix

def combine_into_documents(data_path):
    documents = {}
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            documents.setdefault(
                e.get('speaker'),
                []
            ).extend([e.get('text')])
    return documents

            
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str, default='data/CNNDM/train.label.jsonl', help='File to deal with')
    parser.add_argument('--dataset', type=str, default='CNNDM', help='dataset name')
    parser.add_argument('--sentaspara', type=str, default='sent', choices=['sent', 'para'], help='dataset name')
    
    args = parser.parse_args()
    
    sentaspara = args.sentaspara

    save_dir = os.path.join("cache", args.dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    saveFile = os.path.join(save_dir, "filter_word.{}.txt".format(sentaspara))
    print("Save low tfidf words in dataset %s to %s" % (args.dataset, saveFile))

    documents = []
    documents_lines = combine_into_documents(args.data_path)
    for speaker, e in documents_lines.items():
        if isinstance(e, list) and isinstance(e[0], list):
            text = catDoc(e)
        else:
            text = e
        documents.append(" ".join(text))

    # with open(args.data_path, "r", encoding="utf-8") as f:
    #     for line in f:
    #         e = json.loads(line)
    #         if isinstance(e["text"], list) and isinstance(e["text"][0], list):
    #             text = catDoc(e["text"])
    #         else:
    #             text = e["text"]
    #         print('text: ', text)
    #         input()
    #         documents.append(text)
            
    vectorizer, tfidf_matrix = calTFidf(documents)
    print("The number of example is %d, and the TFIDF vocabulary size is %d" % (len(documents), len(vectorizer.vocabulary_)))
    word_tfidf = np.array(tfidf_matrix.mean(0))
    del tfidf_matrix
    word_order = np.argsort(word_tfidf[0])

    id2word = vectorizer.get_feature_names()
    with open(saveFile, "w") as fout:
        for idx in word_order:
            w = id2word[idx]
            string = w + "\n"
            try:
                fout.write(string)
            except:
                pass
                # print(string.encode("utf-8"))

if __name__ == '__main__':
    main()
    

        
        
        