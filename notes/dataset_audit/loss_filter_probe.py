"""Screen a random web-training sample using old loss and paired loss differences.

This only scores and ranks text; it never edits dataset shards. A large positive
old-minus-sanitized loss measures corpus specialization, not quality by itself.
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from analyze import ROOT, OUT, RAW, MODELS
from inspect_tokens import TokenArtifact, load_tokenizer
from cross_eval import load_model


def summarize(rows, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    old = np.array([r['old_loss'] for r in rows])
    new = np.array([r['sanitized_loss'] for r in rows])
    delta = old - new
    edges = np.unique(np.quantile(old, np.linspace(0, 1, args.baseline_bins + 1)))
    bins = np.clip(np.searchsorted(edges, old, side='right') - 1, 0, len(edges) - 2)
    centers = {i: float(np.median(delta[bins == i])) for i in np.unique(bins)}
    residual = delta - np.array([centers[i] for i in bins])
    for i,r in enumerate(rows):
        r['delta'] = float(delta[i])
        r['conditional_delta'] = float(residual[i])
    scores = {'old_loss': old, 'delta': delta, 'conditional_delta': residual}
    topics = np.array([bool(r['finance_or_vaping']) for r in rows])
    result = dict(settings=vars(args), blocks=len(rows), token_sample=len(rows)*args.sequence_length,
                  models={k: MODELS[k] for k in ['old', 'sanitized']},
                  dataset='artifacts/datasets/balanced_sanitized_v2_16k/train_balanced',
                  source='web', tokenizer='artifacts/tokenizers/production_16k/tokenizer.model',
                  mean_old_loss=float(old.mean()), mean_sanitized_loss=float(new.mean()),
                  quantiles={}, ranking={},
                  note='Topic flag includes legitimate text. Ranking enrichment is not detector precision.')
    for method,values in scores.items():
        result['quantiles'][method] = {str(q): float(np.quantile(values,q)) for q in [.01,.05,.5,.9,.95,.99]}
        order = np.argsort(-values, kind='stable')
        result['ranking'][method] = {}
        for fraction in [.01,.05,.10,.20]:
            n = max(1,round(len(rows)*fraction)); selected=order[:n]
            result['ranking'][method][str(fraction)] = dict(blocks=n, cutoff=float(values[selected[-1]]),
                finance_or_vaping_fraction=float(topics[selected].mean()),
                mean_old_loss=float(old[selected].mean()), mean_sanitized_loss=float(new[selected].mean()),
                mean_delta=float(delta[selected].mean()),
                fraction_of_total_positive_delta=float(np.maximum(delta[selected],0).sum()/np.maximum(delta,0).sum()))
    result['baseline_topic_fraction'] = float(topics.mean())
    # Save exact offsets and full text for manual review of distinct rankings.
    review=[]
    for method,values in scores.items():
        for rank,i in enumerate(np.argsort(-values,kind='stable')[:args.review_per_method],1):
            review.append(dict(method=method, rank=rank, **rows[int(i)]))
    random = np.random.default_rng(args.seed+1).choice(len(rows),size=args.review_per_method,replace=False)
    for rank,i in enumerate(random,1):review.append(dict(method='random',rank=rank,**rows[int(i)]))
    RAW.mkdir(parents=True,exist_ok=True)
    (RAW/'loss_filter_probe_blocks.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    (RAW/'loss_filter_probe_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
    (OUT/'loss_filter_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    axes[0].hist(old,bins=50,color='#3478aa',alpha=.75,label='Old model')
    axes[0].hist(new,bins=50,color='#e28a2f',alpha=.65,label='Sanitized model')
    axes[0].set_xlabel('Loss (nats/token)');axes[0].set_ylabel('Blocks');axes[0].legend()
    axes[0].set_title('Sanitized training web sample')
    for flag,color,label in [(False,'#82939c','Other topics'),(True,'#b65a56','Finance / vaping')]:
        axes[1].scatter(old[topics==flag],new[topics==flag],s=7,alpha=.35,color=color,label=label)
    axes[1].plot([0,6],[0,6],color='black',linewidth=.7)
    axes[1].set_xlim(0,6);axes[1].set_ylim(0,6);axes[1].set_xlabel('Old model loss');axes[1].set_ylabel('Sanitized model loss');axes[1].legend(fontsize=8)
    axes[1].set_title('Specialization changes the relationship')
    for flag,color,label in [(False,'#82939c','Other topics'),(True,'#b65a56','Finance / vaping')]:
        axes[2].hist(delta[topics==flag],bins=np.linspace(-1,3,50),color=color,alpha=.65,label=label)
    axes[2].set_xlabel('Old loss − sanitized loss');axes[2].set_ylabel('Blocks');axes[2].legend(fontsize=8)
    axes[2].set_title('Positive tail is a screening candidate')
    fig.savefig(OUT/'loss_filter_probe.png',dpi=160)
    print(json.dumps(result,indent=2),flush=True)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks',type=int,required=True)
    parser.add_argument('--sequence-length',type=int,required=True)
    parser.add_argument('--batch-size',type=int,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--baseline-bins',type=int,required=True)
    parser.add_argument('--review-per-method',type=int,required=True)
    parser.add_argument('--reuse',action='store_true')
    args=parser.parse_args()
    if args.reuse:
        rows=json.loads((RAW/'loss_filter_probe_blocks.json').read_text())
        if len(rows)!=args.blocks:raise ValueError('cached sample size differs')
        summarize(rows,args);return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('requires CUDA/BF16')
    torch.set_num_threads(4)
    artifact=TokenArtifact(ROOT/'artifacts/datasets/balanced_sanitized_v2_16k/train_balanced')
    tokenizer=load_tokenizer(ROOT/'artifacts/tokenizers/production_16k/tokenizer.model',artifact)
    web=next(r for r in artifact.ranges if r['source']=='web')
    count=(web['tokens']-1)//args.sequence_length
    offsets=web['start_token']+np.random.default_rng(args.seed).choice(count,size=args.blocks,replace=False)*args.sequence_length
    topic=re.compile(r'\b(?:prestit[oi]|finanziament[oi]|noipa)\b|cessione del quinto|sigarett[ae] elettronich[ae]',re.I)
    rows=[]
    for offset in offsets:
        text=tokenizer.decode(artifact.read(int(offset),args.sequence_length).tolist())
        rows.append(dict(start_token=int(offset),text=text,finance_or_vaping=bool(topic.search(text))))
    for label in ['old','sanitized']:
        model,checkpoint=load_model(MODELS[label],torch.device('cuda'))
        if args.sequence_length>checkpoint['hyperparameters']['SEQ_LEN']:raise ValueError('context exceeds model limit')
        started=time.monotonic()
        for left in range(0,len(rows),args.batch_size):
            batch=rows[left:left+args.batch_size]
            tokens=np.stack([artifact.read(r['start_token'],args.sequence_length+1) for r in batch]).astype(np.int64)
            ids=torch.from_numpy(tokens).to('cuda')
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                logits=model(ids[:,:-1])
                loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none').view(len(batch),-1)
            for row,value in zip(batch,loss.mean(dim=1).cpu().tolist(),strict=True):row[label+'_loss']=value
            if left%(args.batch_size*100)==0:print(label,left,'/',len(rows),round(time.monotonic()-started),'seconds',flush=True)
        del model,checkpoint;torch.cuda.empty_cache()
    summarize(rows,args)


if __name__=='__main__':main()
