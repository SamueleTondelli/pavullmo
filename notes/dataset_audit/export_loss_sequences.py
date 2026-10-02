"""Export the largest signed old-minus-sanitized losses with decoded text."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'dataset'))
from inspect_tokens import TokenArtifact, load_tokenizer


def decode_marked(tokenizer, ids, special):
    parts=[]
    content=[]
    for token in ids:
        if token in special:
            if content:
                parts.append(tokenizer.decode(content))
                content=[]
            parts.append(f'\n<{special[token].upper()}>\n')
        else:
            content.append(token)
    if content:parts.append(tokenizer.decode(content))
    return ''.join(parts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores',type=Path,required=True)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--top-k',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    table=pq.read_table(args.scores)
    audit=json.loads(table.schema.metadata[b'audit'])
    artifact=TokenArtifact(args.artifact)
    tokenizer=load_tokenizer(args.tokenizer,artifact)
    if Path(audit['dataset']).resolve()!=artifact.directory.resolve():
        raise ValueError('scores belong to a different artifact')
    if hashlib.sha256((artifact.directory/'metadata.json').read_bytes()).hexdigest()!=audit['metadata_sha256']:
        raise ValueError('artifact metadata has changed since scoring')
    if audit['tokenizer_sha256']!=artifact.metadata['tokenizer']['sha256']:
        raise ValueError('score and artifact tokenizers differ')
    if not 0<args.top_k<=table.num_rows:parser.error('top-k must be between 1 and the scored row count')
    starts=table['start_token'].to_numpy()
    lengths=table['target_tokens'].to_numpy().astype(np.int64)
    sequence_length=audit['sequence_length']
    if not np.array_equal(starts,np.arange(table.num_rows)*sequence_length):
        raise ValueError('scored artifact has missing or duplicate windows')
    if not np.array_equal(lengths,np.minimum(sequence_length,artifact.total_tokens-1-starts)):
        raise ValueError('invalid scored window lengths')
    if lengths.sum()!=artifact.total_tokens-1:raise ValueError('scored artifact is incomplete')
    delta=table['delta'].to_numpy()
    if not np.isfinite(delta).all():raise ValueError('nonfinite differences')
    # Sort signed differences descending; break ties by ascending global offset.
    selected=np.lexsort((starts,-delta))[:args.top_k]
    result=table.take(pa.array(selected,type=pa.int64()))
    special={v:k for k,v in artifact.metadata['tokenizer']['special_token_ids'].items()}
    texts=[]
    for rank,index in enumerate(selected,1):
        ids=artifact.read(int(starts[index]),int(lengths[index])).tolist()
        texts.append(decode_marked(tokenizer,ids,special))
        if rank%1000==0:print(f'Decoded {rank:,}/{args.top_k:,}',flush=True)
    result=result.add_column(0,'rank',pa.array(np.arange(1,args.top_k+1),type=pa.int32()))
    result=result.append_column('text',pa.array(texts,type=pa.string()))
    metadata=dict(table.schema.metadata)
    metadata[b'export']=json.dumps(dict(
        scores=str(args.scores.resolve()),top_k=args.top_k,
        scores_sha256=hashlib.sha256(args.scores.read_bytes()).hexdigest(),
        ranking='delta descending, start_token ascending for ties; delta = old_loss - sanitized_loss',
        text='input tokens [start_token, start_token + target_tokens); control-token markers made visible',
        final_target='The final target at start_token + target_tokens is scored but is outside text.',
    ),sort_keys=True).encode()
    result=result.replace_schema_metadata(metadata)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.parquet.inprogress')
    pq.write_table(result,temporary,compression='zstd',compression_level=9,row_group_size=1000,
                   use_dictionary=['split','source','source_end','batch_size'])
    restored=pq.read_table(temporary)
    if not restored.equals(result,check_metadata=True):raise ValueError('Parquet round-trip changed the export')
    if not (np.diff(restored['delta'].to_numpy())<=0).all():raise ValueError('export is not ranked')
    if any(not t for t in restored['text'].to_pylist()):raise ValueError('empty decoded text')
    temporary.replace(args.output)
    print(f'Wrote {args.output}: {result.num_rows:,} rows, {args.output.stat().st_size:,} bytes; '
          f'delta {float(delta[selected[-1]]):.6f}–{float(delta[selected[0]]):.6f}',flush=True)


if __name__=='__main__':main()
