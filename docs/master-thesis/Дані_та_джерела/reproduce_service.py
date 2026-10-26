import sys,json,platform,time,statistics
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch,numpy as np
from smart_house.analytics.comparison import model_digest
from smart_house.analytics.lab import ModelLab
from smart_house.analytics.history import History
R=Path(__file__).resolve().parent;model=Path.cwd()/'data/models/planned-v1.pt'
torch.set_num_threads(2)
lab=ModelLab(model,History(R/'latency-reproduced.db'))
reading={'timestamp':'2026-10-26T12:00:00+02:00','temperature_c':22,'humidity_pct':50,'total_occupants':2}
cold=lab.predict({'reading':reading});latencies=[]
for i in range(40):
 r=lab.predict({'reading':reading|{'temperature_c':21+i*.02}});assert not r['cache_hit'];latencies.append(r['latency_ms'])
hits=[]
for i in range(40):
 r=lab.predict({'reading':reading});assert r['cache_hit'];hits.append(r['latency_ms'])
safety=lab._safety_checks(lab._service('model-0'))
agent=lab._service('model-0')._agent
D={'measured_at':'2026-10-26','checkpoint_sha256':model_digest(model),'state_dim':54,'action_mode':agent.action_mode,'actions':agent.n_actions,'parameters':sum(p.numel() for p in agent.policy_net.parameters()),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'torch_threads':torch.get_num_threads(),'cold_ms':cold['latency_ms'],'uncached_ms':latencies,'cached_ms':hits,'uncached_median_ms':statistics.median(latencies),'cached_median_ms':statistics.median(hits),'uncached_p95_ms':float(np.percentile(latencies,95)),'cached_p95_ms':float(np.percentile(hits,95)),'safety_checks':safety}
(R/'reproduced-service.json').write_text(json.dumps(D,ensure_ascii=False,indent=2));lab.close();print(json.dumps({k:v for k,v in D.items() if k not in ['uncached_ms','cached_ms']},ensure_ascii=False))
