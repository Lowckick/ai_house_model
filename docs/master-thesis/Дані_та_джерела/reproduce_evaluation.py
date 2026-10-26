"""Run from the matching repository checkout; leaves the checkpoint unchanged."""
import json,sys,statistics
from pathlib import Path
import torch
sys.path.insert(0,str(Path.cwd()))
from smart_house.analytics.comparison import load_agent,compare_controllers,model_digest
R=Path(__file__).resolve().parent;model=Path.cwd()/'data/models/planned-v1.pt'
torch.set_num_threads(2)
expected='a525ffcc0486ed5f8613229680333b12764fcf485adadfed2f3aceef1eb5337d'
assert model_digest(model)==expected,'Checkpoint differs from reported model'
results=[]
for seed in [61000,71000,81000]:
 result=compare_controllers(load_agent(model),20,144,seed)
 (R/f'reproduced-test-{seed}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 results.append(result)
summary={c:{k:statistics.mean(x['summary'][c][k] for x in results) for k in ['avg_comfort','energy_kwh','total_reward','unsafe_steps']} for c in ['model','deterministic']}
(R/'reproduced-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
