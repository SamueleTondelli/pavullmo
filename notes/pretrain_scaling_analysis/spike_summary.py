import csv,json
from pathlib import Path
import numpy as np
out=Path(__file__).resolve().parent
results=[]
for p in sorted((out/'logs').glob('*.csv')):
 d={}
 for r in csv.DictReader(p.open()):d.setdefault(r['tag'],[]).append((int(r['step']),float(r['value'])))
 g=np.array(d['train/gradient_norm']);l=np.array(d['train/loss']);v=np.array(d['validation/loss'])
 assert np.array_equal(g[:,0],l[:,0])
 med=np.median(np.lib.stride_tricks.sliding_window_view(np.pad(l[:,1],(100,100),mode='edge'),201),axis=1)
 resid=l[:,1]-med;m=g[:,0]>1000;sp=np.flatnonzero(m & (g[:,1]>1))
 after=[float(np.mean(resid[i+1:i+21])) for i in sp if i+21<=len(resid)]
 row={'run':p.stem,'steps':len(g),'after_step':1000,'median_grad':float(np.median(g[m,1])),'p99_grad':float(np.quantile(g[m,1],.99)),'max_grad':float(g[m,1].max()),'clipped_steps':len(sp),'clipped_percent':100*len(sp)/m.sum(),'loss_residual_p95':float(np.quantile(resid[m],.95)),'loss_residual_p99':float(np.quantile(resid[m],.99)),'loss_residual_at_clipped_median':float(np.median(resid[sp])) if len(sp) else None,'next20_loss_residual_mean':float(np.mean(after)) if after else None,'largest_validation_increase':float(np.diff(v[:,1]).max()),'validation_increases':int((np.diff(v[:,1])>0).sum())}
 results.append(row)
 print(json.dumps(row))
(out/'spike_summary.json').write_text(json.dumps(results,indent=2)+'\n')
