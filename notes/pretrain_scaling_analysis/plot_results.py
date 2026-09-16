import csv,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
rows=list(csv.DictReader((root/'src/pavullmo/pretrain_runs.csv').open()))
base='35m_balanced_100m_muon_lrwd_20260913T183237Z-51200_lr110_mwd150'
names=[base,'35m_balanced_300m','35m_balanced_1b','35m_web_100m','35m_web_300m','35m_knowledge_100m','35m_knowledge_300m']
selected=[r for r in rows if r['experiment_name'] in names]
assert len(selected)==7
for r in selected:
 r['train_loss']=float(r['train_loss']);r['validation_loss']=float(r['validation_loss'])
 r['perplexity']=math.exp(r['validation_loss'])
(out/'selected_results.json').write_text(json.dumps(selected,indent=2)+'\n')
colors={'web':'#0072B2','balanced':'#009E73','knowledge':'#D55E00'}
fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for mix,color in colors.items():
 group=sorted([r for r in selected if r['dataset_variant'].startswith(mix+'_')],key=lambda r: 1000 if r['dataset_variant'].endswith('1b') else int(r['dataset_variant'].split('_')[-1][:-1]))
 xs=[1000 if r['dataset_variant'].endswith('1b') else int(r['dataset_variant'].split('_')[-1][:-1]) for r in group]
 for ax,key in zip(axs,['validation_loss','train_loss']):
  ys=[r[key] for r in group]
  ax.plot(xs,ys,'o-',color=color,label=mix.capitalize(),lw=1.8)
  for x,y in zip(xs,ys):ax.annotate(f'{y:.3f}',(x,y),xytext=(0,(-17 if mix=='knowledge' else 10) if key=='train_loss' else (12 if mix=='knowledge' else -17 if mix=='web' else 10)),textcoords='offset points',ha='center',fontsize=9)
for ax in axs:
 ax.set_xscale('log');ax.set_xticks([100,300,1000],['100M','300M','1B']);ax.set_xlim(80,1250)
 ax.set_xlabel('Training dataset size (tokens)');ax.set_ylabel('Cross-entropy (nats/token)');ax.grid(alpha=.18);ax.margins(y=.2)
axs[0].set_title('Recorded validation loss · lower is better')
axs[1].set_title('Final training batch · not a dataset average')
axs[0].legend(frameon=False)
fig.suptitle('35M model: dataset scaling and mixture comparison',fontsize=15)
fig.get_layout_engine().set(rect=(0,.12,1,.78))
fig.text(.5,.025,'Single seed; balanced 100M uses matching sweep settings. Validation: 7 × 16 × 1,024 = 114,688 tokens.\nDataset artifacts are absent locally: actual evaluation composition remains unverified.',ha='center',fontsize=9)
fig.savefig(out/'final_results.png',dpi=180)
fig.savefig(out/'final_results.svg')
for r in selected:print(r['dataset_variant'],round(r['validation_loss'],4),round(r['perplexity'],2))
