import sys,json,time,platform,hashlib,statistics
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch,numpy as np
from smart_house.analytics.comparison import load_agent,compare_controllers,model_digest
from smart_house.analytics.lab import ModelLab
from smart_house.analytics.history import History
from smart_house.environment.indoor_state import IndoorState,DeviceState
from smart_house.ml.agent import STATE_DIM
root=Path(__file__).resolve().parent
torch.set_num_threads(2)
model=Path.cwd()/'data/models/improved-v2.pt'
agent=load_agent(model)
start=time.perf_counter()
comparison=compare_controllers(agent,12,144,20000)
(root/'evaluation.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2))
lab=ModelLab(model,History(root/'lab.db'))
reading={'timestamp':'2026-10-26T12:00:00+02:00','temperature_c':22,'humidity_pct':50,'total_occupants':2}
cold=lab.predict({'reading':reading})
latencies=[]
for i in range(40):
 r=lab.predict({'reading':reading|{'temperature_c':21+i*.02}})
 latencies.append(r['latency_ms'])
hits=[lab.predict({'reading':reading})['latency_ms'] for _ in range(40)]
safety=lab._safety_checks(lab._service('model-0'))
metadata={'checkpoint_sha256':model_digest(model),'state_dim':STATE_DIM,'action_mode':agent.action_mode,'actions':agent.n_actions,'parameters':sum(p.numel() for p in agent.policy_net.parameters()),'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'torch_threads':torch.get_num_threads(),'cold_ms':cold['latency_ms'],'uncached_ms':latencies,'cached_ms':hits,'uncached_median_ms':statistics.median(latencies),'cached_median_ms':statistics.median(hits),'uncached_p95_ms':float(np.percentile(latencies,95)),'cached_p95_ms':float(np.percentile(hits,95)),'safety_checks':safety,'evaluation_duration_s':time.perf_counter()-start}
(root/'measurements.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
lab.close()
print(json.dumps({k:v for k,v in metadata.items() if k not in ('cached_ms','uncached_ms')},ensure_ascii=False,indent=2))
print(json.dumps(comparison['summary'],ensure_ascii=False,indent=2))
