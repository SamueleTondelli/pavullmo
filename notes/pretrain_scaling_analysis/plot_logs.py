import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'logs'
OUT.mkdir(exist_ok=True)
TAGS=['train/loss','validation/loss','train/gradient_norm','train/learning_rate','train/muon_learning_rate','train/tokens_seen']
runs=[];inventory={}
for folder in sorted((ROOT/'tmp/logs').iterdir()):
 e=EventAccumulator(str(folder),size_guidance={'scalars':0});e.Reload()
 data={tag:e.Scalars(tag) for tag in TAGS if tag in e.Tags()['scalars']}
 inventory[folder.name]={tag:len(e.Scalars(tag)) for tag in e.Tags()['scalars']}
 label=folder.name.removeprefix('35m_').split('_muon_')[0].replace('_',' ').capitalize()
 runs.append((folder.name,label,data))
 with (OUT/(folder.name+'.csv')).open('w') as f:
  w=csv.writer(f);w.writerow(['tag','step','wall_time','value'])
  for tag,events in data.items():
   w.writerows((tag,v.step,v.wall_time,v.value) for v in events)
 fig,axs=plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained')
 for tag,color,name in [('train/loss','#0072B2','Training'),('validation/loss','#D55E00','Validation')]:
  v=data[tag];axs[0].plot([q.step for q in v],[q.value for q in v],label=name,color=color,lw=.65 if tag=='train/loss' else 1.3)
 axs[0].set_ylabel('Loss (nats/token)');axs[0].legend(frameon=False)
 for ax,tag,ylabel in [(axs[1],'train/gradient_norm','Gradient norm'),(axs[2],'train/learning_rate','Learning rate')]:
  v=data[tag];ax.plot([q.step for q in v],[q.value for q in v],lw=.65,color='#0072B2');ax.set_ylabel(ylabel)
 for ax in axs:ax.grid(alpha=.2)
 axs[-1].set_xlabel('Optimizer step');fig.suptitle(label+' · raw logged values',fontsize=14)
 fig.savefig(OUT/(folder.name+'.png'),dpi=150);plt.close(fig)
 print(folder.name,len(data['train/loss']),data['train/loss'][-1].step,flush=True)
(OUT/'metric_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
fig,axs=plt.subplots(3,3,figsize=(14,9),sharey=True,layout='constrained')
lookup={label:(name,data) for name,label,data in runs}
for i,mix in enumerate(['Web','Balanced','Knowledge']):
 for j,size in enumerate(['100m','300m','1b']):
  ax=axs[i,j];label=mix+' '+size
  ax.set_title(label.replace('100m','100M').replace('300m','300M').replace('1b','1B'))
  if label not in lookup:
   ax.text(.5,.5,'No log supplied',ha='center',va='center',transform=ax.transAxes)
  else:
   _,data=lookup[label]
   for tag,color,name in [('train/loss','#0072B2','Training'),('validation/loss','#D55E00','Validation')]:
    v=data[tag];ax.plot([q.step/1000 for q in v],[q.value for q in v],color=color,lw=.55 if tag=='train/loss' else 1.2,label=name)
   ax.grid(alpha=.2)
  if j==0:ax.set_ylabel('Loss (nats/token)')
  if i==2:ax.set_xlabel('Optimizer steps (thousands)')
axs[0,0].legend(frameon=False)
fig.suptitle('Training and validation loss · all recorded points, no smoothing',fontsize=15)
fig.savefig(OUT/'loss_overview.png',dpi=160);plt.close(fig)
