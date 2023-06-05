import os
import torch
import random

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 

from module.embedding import Word_Embedding
from module.INFOembedding import CEFREmbed, FILLEDEmbed
from module.vocabulary import Vocab
from tools.utils import (
    BERT2ABB,
    pikleOpen,
)
from tools.args import get_eval_args


def fix_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if not args.use_amp:
    #     torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def visualize_layerwise_embeddings(labels, embeds, title):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    labels = np.array(labels)
    embeds = torch.cat(embeds, dim=0).detach().numpy()
    layer_dim_reduced_embeds = dim_reducer.fit_transform(embeds)
    df = pd.DataFrame.from_dict({'x':layer_dim_reduced_embeds[:,0],'y':layer_dim_reduced_embeds[:,1],'label':labels})
    sns.scatterplot(data=df,x='x',y='y',hue='label', ax=ax)
    save_path = os.path.join(alsy_dir, '{}.tsne.png'.format(title))
    plt.savefig(save_path, format='png', pad_inches=0)
    print('Image is saved at {}'.format(save_path))

def deal_mul_cefrs(taglist, cefr_loader, mode='minmean'):
    if taglist == [0]:
        return '0'
    assert mode in ['minmean', 'maxmean']
    CEFR2INT = cefr_loader.get_CEFR2INT()
    t_taglist = np.array([CEFR2INT[t] for t in taglist])
    if mode == 'minmean':
        m = np.array([np.mean(t_taglist).item()]*len(CEFR2INT))
        n = list(m - np.array(list(CEFR2INT.values())))
        min_v = min(n)
        mm_v = max(filter(lambda x: x == min_v, n))
        g_idx = n.index(mm_v)
        if 0. in n:
            g_idx = n.index(0.)
        return list(CEFR2INT.keys())[g_idx]
    elif mode == 'maxmean':
        m = np.array([np.mean(t_taglist).item()]*len(taglist))
        n = list(m - np.array(list(CEFR2INT.values())))
        mn_v = min(filter(lambda x: x >= 0, n))
        g_idx = n.index(mn_v)
        if 0. in n:
            g_idx = n.index(0.)
        return list(CEFR2INT.keys())[g_idx]

hps = get_eval_args()
hps.save_dir_name = "{}".format(
    '.'.join(
        [hps.model, hps.sentaspara] + \
        (['reweight{}'.format('' if hps.rw_alpha == 1. else hps.rw_alpha)] if hps.reweight else []) + \
        ([] if hps.mean_paragraphs is None else [hps.mean_paragraphs]) + \
        [hps.problem_type] + \
        [hps.head] + \
        ([BERT2ABB[hps.bert_model_path]] if hps.bert else []) + \
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

alsy_dir = os.path.join(hps.save_root, hps.save_dir_name, "analysis")
pnode_file_path = os.path.join(alsy_dir, 'slabels.sembeds')
(labels, embeds) = pikleOpen(pnode_file_path)
dim_reducer = TSNE(n_components=5, method='exact', init='pca', learning_rate='auto')

if hps.eval_speaker_wise:
    embeds = [torch.mean(e, dim=0).unsqueeze(0) for e in embeds]
    
## Paragraph Nodes t-SNE
visualize_layerwise_embeddings(labels=labels,
                               embeds=embeds,
                               title='train_data',
                              )


# CEFR node and Filled Pauses node
cefr_loader, filled_pauses_loader = None, None
if hps.cefr_word and (hps.cefr_info == 'embed_init'):
    VOCABPROFILE_FILE = os.path.join(hps.data_dir, 'cefrj1.6_c1c2.final.txt')
    cefr_loader = CEFREmbed(hps.word_emb_dim, VOCABPROFILE_FILE)
if hps.filled_pauses_word and (hps.filled_pauses_info == 'embed_init'):
    FLUENCYPAUSE_FILE = os.path.join(hps.data_dir, 'all.filled_pauses.txt')
    filled_pauses_loader = FILLEDEmbed(hps.word_emb_dim, FLUENCYPAUSE_FILE)
VOCAL_FILE = os.path.join(hps.cache_dir, "vocab.combine" if hps.interviewer else 'vocab')
vocab = Vocab(VOCAL_FILE, hps.vocab_size)
assert cefr_loader is not None

embedd = torch.nn.Embedding(vocab.size(), hps.word_emb_dim, padding_idx=0)
embed_loader = Word_Embedding(hps.embedding_path, vocab)
word2cefr, vectors = embed_loader.get_word_cefr_list_and_vectors(cefr_loader, filled_pauses_loader, k=hps.word_emb_dim)
pretrained_weight, word2cefr = embed_loader.add_unknown_words_by_avg(vectors, hps.word_emb_dim, dic=word2cefr)
embedd.weight.data.copy_(torch.Tensor(pretrained_weight))
# if not hps.word_embedding:
#     embedd = torch.nn.Embedding(vocab.size(), hps.word_emb_dim, padding_idx=0)

# get learned cefr2embeds (cefr from referenced words. if a word mapped to multiple cefrs, get the minimum mean in the list)
pnode_file_path = os.path.join(alsy_dir, 'wlabels.wembeds')
(labels, embeds) = pikleOpen(pnode_file_path)
collect_w = {}
for data, ebd in zip(labels, embeds):
    for i, wid in enumerate(data):
        bd = ebd[i]
        collect_w.setdefault(
            wid,
            []
        ).append(bd)

wid2lembed = {wid: torch.mean(torch.cat([e.unsqueeze(0) for e in c_bds], dim=0), dim=0).unsqueeze(0) for wid, c_bds in collect_w.items()}
lembed_cefrs = [deal_mul_cefrs(word2cefr[vocab.id2word(wid)], cefr_loader, mode='minmean') for wid, _ in wid2lembed.items()]

wid2embed = {wid: embedd(torch.tensor([wid])) for wid, _ in collect_w.items()}
embed_cefrs = [deal_mul_cefrs(word2cefr[vocab.id2word(wid)], cefr_loader, mode='minmean') for wid, _ in wid2embed.items()]

## Paragraph Nodes t-SNE
visualize_layerwise_embeddings(labels=lembed_cefrs,
                               embeds=list(wid2lembed.values()),
                               title='wnode.wenc',
                              )

visualize_layerwise_embeddings(labels=embed_cefrs,
                               embeds=list(wid2embed.values()),
                               title='wnode.init',
                              )