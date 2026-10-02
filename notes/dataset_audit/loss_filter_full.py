"""Exhaustively score paired checkpoint losses into resumable compressed Parquet.

Includes the final partial context: every available next-token transition is
scored once. Decoded text stays in the original shards and is recovered by offset.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from analyze import ROOT, MODELS
from inspect_tokens import TokenArtifact, load_tokenizer
from cross_eval import load_model


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def token_losses(logits, targets):
    return F.cross_entropy(logits.float(), targets, reduction='none')


def compatible_metadata(cached, expected):
    """Batch size may change on resume; model, data, and loss settings may not."""
    if set(cached or {}) != set(expected or {}):
        return False
    previous=json.loads(cached[b'audit'])
    current=json.loads(expected[b'audit'])
    previous.pop('requested_batch_size',None)
    current.pop('requested_batch_size',None)
    return previous==current


@torch.inference_mode()
def score(model, ids, head_tokens, loss_function):
    # Exactly DecoderTransformer.forward, splitting only the final linear head.
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        hidden = model.embeddings(ids[:, :-1])
        for layer in model.layers:
            hidden = layer(hidden)
        hidden = model.norm(hidden).reshape(-1, model.embeddings.embedding_dim)
        targets = ids[:, 1:].reshape(-1)
        losses = torch.empty(targets.numel(), device=ids.device, dtype=torch.float32)
        for left in range(0, targets.numel(), head_tokens):
            right = min(left + head_tokens, targets.numel())
            logits = F.linear(hidden[left:right], model.embeddings.weight, bias=None)
            # Avoid compiling many shapes for the last partial chunk/context.
            fn = loss_function if right-left == head_tokens else token_losses
            losses[left:right] = fn(logits, targets[left:right])
            del logits
        return losses.reshape(ids.shape[0], ids.shape[1]-1).mean(dim=1).cpu().numpy()


def attempt(models, artifact, starts, length, head_tokens, loss_function):
    """Scope CUDA temporaries so an OOM can be retried after they are freed."""
    try:
        tokens = np.stack([artifact.read(int(s), length+1) for s in starts]).astype(np.int64)
        ids = torch.from_numpy(tokens).to('cuda')
        values = [score(model, ids, head_tokens, loss_function) for model in models]
        return values, None
    except torch.cuda.OutOfMemoryError as error:
        return None, str(error).split('\n')[0]


def table_for(artifact, split, starts, lengths, old, new, batch_sizes, metadata):
    source = [artifact.source_at(int(s)+1) for s in starts]
    source_end = [artifact.source_at(int(s)+int(n)) for s,n in zip(starts,lengths,strict=True)]
    return pa.table({
        'split': pa.array([split]*len(starts)).dictionary_encode(),
        'start_token': pa.array(starts, type=pa.int64()),
        'target_tokens': pa.array(lengths, type=pa.int16()),
        'source': pa.array(source).dictionary_encode(),
        'source_end': pa.array(source_end).dictionary_encode(),
        'crosses_source_boundary': pa.array([a!=b for a,b in zip(source,source_end,strict=True)]),
        'old_loss': pa.array(old, type=pa.float32()),
        'sanitized_loss': pa.array(new, type=pa.float32()),
        'delta': pa.array(old-new, type=pa.float32()),
        'batch_size': pa.array(batch_sizes, type=pa.int16()),
    }).replace_schema_metadata(metadata)


def summarize(path, artifact, split, metadata):
    table = pq.read_table(path)
    starts = table['start_token'].to_numpy()
    weights = table['target_tokens'].to_numpy().astype(np.int64)
    if not np.array_equal(starts, np.arange(len(starts))*metadata['sequence_length']):
        raise ValueError('missing or duplicate offsets in final Parquet')
    if weights.sum() != artifact.total_tokens-1:
        raise ValueError('not all next-token transitions were scored')
    rows=[]
    sources=np.array(table['source'].to_pylist())
    crossing=table['crosses_source_boundary'].to_numpy()
    for group in ['all',*sorted(set(sources))]:
        selected=np.ones(len(starts),dtype=bool) if group=='all' else (sources==group)&~crossing
        for column in ['old_loss','sanitized_loss','delta']:
            values=table[column].to_numpy()[selected]
            if not np.isfinite(values).all():raise ValueError('nonfinite loss')
            statistics={'token_weighted_mean':np.average(values,weights=weights[selected]),
                        'blocks':int(selected.sum()),'target_tokens':int(weights[selected].sum())}
            statistics.update({f'q{q:g}':np.quantile(values,q) for q in [.01,.05,.5,.9,.95,.99]})
            for metric,value in statistics.items():
                rows.append(dict(split=split,source=group,score=column,metric=metric,value=float(value)))
    print(f'{split}: verified {len(starts):,} blocks, {weights.sum():,} target tokens; '
          f'old={np.average(table["old_loss"].to_numpy(),weights=weights):.6f} '
          f'sanitized={np.average(table["sanitized_loss"].to_numpy(),weights=weights):.6f}',flush=True)
    return rows


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--splits',nargs='+',choices=['train','validation','test'],required=True)
    parser.add_argument('--batch-size',type=int,required=True)
    parser.add_argument('--sequence-length',type=int,required=True)
    parser.add_argument('--head-tokens',type=int,required=True)
    parser.add_argument('--part-blocks',type=int,required=True)
    parser.add_argument('--threads',type=int,required=True)
    parser.add_argument('--progress-seconds',type=float,required=True)
    parser.add_argument('--compile-loss',action='store_true')
    args=parser.parse_args()
    for name in ['batch_size','sequence_length','head_tokens','part_blocks','threads','progress_seconds']:
        if getattr(args,name)<=0:parser.error(f'{name} must be positive')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA/BF16 is required')
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    # Keep compilation caches inside the workspace, including on a local replay.
    cache=args.output_dir/'compiler_cache'
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(cache/'inductor'))
    os.environ.setdefault('TRITON_CACHE_DIR',str(cache/'triton'))
    loss_function=torch.compile(token_losses,fullgraph=True) if args.compile_loss else token_losses
    loaded=[load_model(MODELS[k],torch.device('cuda')) for k in ['old','sanitized']]
    models=[item[0] for item in loaded]
    for _,checkpoint in loaded:
        if args.sequence_length>checkpoint['hyperparameters']['SEQ_LEN']:
            raise ValueError('context exceeds checkpoint limit')
    checkpoints={k:dict(name=MODELS[k],sha256=digest(ROOT/'artifacts/models'/f'{MODELS[k]}.pt'),
                       global_step=loaded[i][1]['global_step']) for i,k in enumerate(['old','sanitized'])}
    paths={'train':'train_balanced','validation':'validation','test':'test'}
    batch_size=args.batch_size
    summary_rows=[]
    total_started=time.monotonic()
    for split in args.splits:
        artifact=TokenArtifact(args.dataset_root/paths[split])
        load_tokenizer(ROOT/'artifacts/tokenizers/production_16k/tokenizer.model',artifact)
        run=dict(sequence_length=args.sequence_length,head_tokens=args.head_tokens,
                 requested_batch_size=args.batch_size,compile_loss=args.compile_loss,
                 part_blocks=args.part_blocks,precision='FP32 parameters, BF16 autocast, FP32 cross entropy',
                 torch_version=str(torch.__version__),gpu=torch.cuda.get_device_name(),
                 checkpoints=checkpoints,dataset=str(artifact.directory.resolve()),
                 metadata_sha256=digest(artifact.directory/'metadata.json'),
                 tokenizer_sha256=artifact.metadata['tokenizer']['sha256'],
                 definition='delta = old_loss - sanitized_loss; losses are nats/target-token',
                 source_summary='cross-source blocks included in all, excluded from source-specific summaries')
        metadata={b'audit':json.dumps(run,sort_keys=True).encode()}
        final=args.output_dir/f'{split}.parquet'
        parts=args.output_dir/'parts'/split
        parts.mkdir(parents=True,exist_ok=True)
        full,tail=divmod(artifact.total_tokens-1,args.sequence_length)
        blocks=full+bool(tail)
        last_progress=0.0
        print(f'{split}: {artifact.total_tokens:,} tokens / {blocks:,} blocks; batch={batch_size}',flush=True)
        # Validate the chunked forward against the ordinary checkpoint forward.
        example=torch.from_numpy(artifact.read(0,2*(args.sequence_length+1)).astype(np.int64)
                                 .reshape(2,args.sequence_length+1)).to('cuda')
        for label,model in zip(['old','sanitized'],models,strict=True):
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                logits=model(example[:,:-1])
                reference=token_losses(logits.reshape(-1,logits.shape[-1]),example[:,1:].reshape(-1))
                reference=reference.reshape(2,-1).mean(dim=1).cpu().numpy()
                if args.compile_loss and split==args.splits[0] and label=='old':
                    flat=logits.reshape(-1,logits.shape[-1])
                    repeat=(args.head_tokens+len(flat)-1)//len(flat)
                    probe=flat.repeat(repeat,1)[:args.head_tokens]
                    targets=example[:,1:].reshape(-1).repeat(repeat)[:args.head_tokens]
                    expected=token_losses(probe,targets)
                    actual=loss_function(probe,targets)
                    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
                    del probe,targets,expected,actual,flat
                    print('Compiled/eager cross entropy agreement verified',flush=True)
            del logits
            measured=score(model,example,args.head_tokens,loss_function)
            np.testing.assert_allclose(measured,reference,rtol=1e-6,atol=1e-6)
            print(f'{label}: ordinary/chunked loss agreement verified',flush=True)
        del example
        for left in range(0,blocks,args.part_blocks):
            right=min(left+args.part_blocks,blocks)
            part=parts/f'part-{left:09d}.parquet'
            if part.exists():
                cached=pq.read_table(part)
                if not compatible_metadata(cached.schema.metadata,metadata) or cached.num_rows!=right-left:
                    raise ValueError(f'incompatible cached part: {part}')
                if not np.array_equal(cached['start_token'].to_numpy(),np.arange(left,right)*args.sequence_length):
                    raise ValueError(f'incorrect cached offsets: {part}')
                print(f'{split}: resume through block {right:,}',flush=True)
                continue
            starts=np.arange(left,right,dtype=np.int64)*args.sequence_length
            lengths=np.minimum(args.sequence_length,artifact.total_tokens-1-starts).astype(np.int16)
            old=np.empty(len(starts),dtype=np.float32);new=np.empty_like(old)
            sizes=np.empty(len(starts),dtype=np.int16)
            position=0
            while position<len(starts):
                # A partial tail has a different length and gets its own batch.
                length=int(lengths[position])
                take=min(batch_size,len(starts)-position)
                if lengths[position+take-1]!=length:take-=1
                take=max(1,take)
                values,error=attempt(models,artifact,starts[position:position+take],length,
                                     args.head_tokens,loss_function)
                if error:
                    gc.collect();torch.cuda.empty_cache()
                    if take==1:raise RuntimeError(f'batch size 1 OOM: {error}')
                    reduced=max(1,take//2)
                    print(f'OOM at batch {take}; reducing batch to {reduced}: {error}',flush=True)
                    batch_size=reduced
                    continue
                old[position:position+take],new[position:position+take]=values
                sizes[position:position+take]=take
                position+=take
                now=time.monotonic()
                if now-last_progress>=args.progress_seconds:
                    done=left+position
                    elapsed=now-total_started
                    print(f'{split}: {done:,}/{blocks:,} blocks ({done/blocks:.2%}), '
                          f'batch={batch_size}, elapsed={elapsed/60:.1f} min, '
                          f'GPU peak={torch.cuda.max_memory_allocated()/2**30:.2f} GiB',flush=True)
                    last_progress=now
            if not (np.isfinite(old).all() and np.isfinite(new).all()):raise ValueError('nonfinite loss')
            table=table_for(artifact,split,starts,lengths,old,new,sizes,metadata)
            temporary=part.with_suffix('.parquet.inprogress')
            pq.write_table(table,temporary,compression='zstd',use_dictionary=True)
            temporary.replace(part)
        temporary=final.with_suffix('.parquet.inprogress')
        part_paths=sorted(parts.glob('part-*.parquet'))
        requested_sizes=sorted({json.loads(pq.read_schema(p).metadata[b'audit'])['requested_batch_size']
                                for p in part_paths})
        final_audit=dict(run)
        final_audit['consolidation_requested_batch_size']=final_audit.pop('requested_batch_size')
        final_audit['requested_batch_sizes_scored']=requested_sizes
        final_metadata={b'audit':json.dumps(final_audit,sort_keys=True).encode()}
        schema=pq.read_schema(parts/'part-000000000.parquet').with_metadata(final_metadata)
        with pq.ParquetWriter(temporary,schema,compression='zstd') as writer:
            for part in part_paths:
                writer.write_table(pq.read_table(part).replace_schema_metadata(final_metadata))
        temporary.replace(final)
        summary_rows.extend(summarize(final,artifact,split,run))
        pq.write_table(pa.Table.from_pylist(summary_rows),args.output_dir/'summary.parquet',compression='zstd')
    print(f'COMPLETE: {time.monotonic()-total_started:.1f} seconds; results: {args.output_dir}',flush=True)


if __name__=='__main__':
    main()
