import sys,json,csv,hashlib,platform,time,random
from pathlib import Path
from collections import Counter
from dataclasses import asdict
from unittest.mock import patch
sys.path.insert(0,str(Path.cwd()))
import torch,numpy as np
from smart_house.analytics.comparison import load_agent,model_digest,compare_controllers,assumed_weather
from smart_house.ml.agent import encode_state
from smart_house.environment.indoor_state import IndoorPhysicsSimulator,DeviceState
from smart_house.environment.scenarios import make_scenario
from smart_house.ml.controller import deterministic_devices
from smart_house.ml.rule_layer import RuleLayer
ROOT=Path(__file__).resolve().parent;torch.set_num_threads(2)
MODEL=Path.cwd()/'data/models/improved-v2.pt';agent=load_agent(MODEL)
digest_before=model_digest(MODEL)
original_select=agent.select_action;original_decode=agent.decode_action;original_step=IndoorPhysicsSimulator.step
allrows=[];runs=[]
def canonical(x):return json.dumps(x,sort_keys=True,default=str,allow_nan=False,separators=(',',':'))
for seed,episodes in [(20000,12),(42,20),(101,20),(2026,20),(31415,20),(987654,20)]:
 hist=Counter();raw_diffs=0;final_diffs=0;step_count=0;traces=[];hashes={};examples=[]
 def select(state):
  a=original_select(state);hist[a]+=1;return a
 def decode(a,current,indoor,weather):
  global raw_diffs,final_diffs
  raw=original_decode(a,current,indoor,weather);baseline=deterministic_devices(indoor,weather)
  final=RuleLayer().apply(raw,indoor,weather)
  raw_diffs+=int(asdict(raw)!=asdict(baseline));different=asdict(final)!=asdict(baseline);final_diffs+=int(different)
  if different and len(examples)<10:examples.append({'state':asdict(indoor),'action':a,'final':asdict(final),'baseline':asdict(baseline)})
  return raw
 def step(self,weather,devices):
  global step_count
  state=original_step(self,weather,devices)
  episode=step_count//288;controller='model' if step_count%288<144 else 'deterministic';position=step_count%144
  key=(episode,controller)
  if key not in hashes:hashes[key]=hashlib.sha256()
  hashes[key].update(canonical({'state':asdict(state),'devices':asdict(devices)}).encode())
  traces.append({'seed':seed,'episode':episode+1,'controller':controller,'step':position+1,'timestamp':state.timestamp.isoformat(),'temperature_c':state.temperature_c,'humidity_pct':state.humidity_pct,'co2_ppm':state.co2_ppm,'pm25_ugm3':state.pm25_ugm3,'comfort':state.comfort_score,'power_kw':state.power_consumption_kw,'devices_json':canonical(asdict(devices))})
  step_count+=1;return state
 started=time.perf_counter()
 with patch.object(agent,'select_action',side_effect=select),patch.object(agent,'decode_action',side_effect=decode),patch.object(IndoorPhysicsSimulator,'step',new=step):
  result=compare_controllers(agent,episodes,144,seed)
 same=[hashes[(e,'model')].hexdigest()==hashes[(e,'deterministic')].hexdigest() for e in range(episodes)]
 diagnostic={'action_histogram':dict(hist),'raw_command_differences_on_same_model_states':raw_diffs,'final_command_differences_on_same_model_states':final_diffs,'full_trajectory_hashes_equal':same,'trajectory_hashes':[{'episode':e+1,'model':hashes[(e,'model')].hexdigest(),'deterministic':hashes[(e,'deterministic')].hexdigest()} for e in range(episodes)],'difference_examples':examples,'seconds':time.perf_counter()-started}
 (ROOT/f'comparison-seed-{seed}.json').write_text(json.dumps(result|{'diagnostics':diagnostic},ensure_ascii=False,indent=2,default=str))
 with (ROOT/f'trajectories-seed-{seed}.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=traces[0].keys());writer.writeheader();writer.writerows(traces)
 runs.append({'seed':seed,'episodes':episodes,'summary':result['summary'],'delta':result['delta_model_minus_deterministic'],'diagnostics':diagnostic})
 for i,t in enumerate(result['trials'],1):
  for name in ['model','deterministic']:allrows.append({'seed':seed,'episode':i,'controller':name,**t[name]})
 print('seed',seed,'completed; actions',dict(hist),'final mismatches',final_diffs,'equal trajectories',sum(same),'/',episodes,flush=True)
# Diagnostic probe beyond the closed-loop trajectories; not a performance benchmark.
rng=random.Random(712026);probe_hist=Counter();probe_diffs=0
for i in range(1000):
 scenario=make_scenario('test',i,712026);state=scenario.environment().state
 state.temperature_c=rng.uniform(-10,45);state.room_temps={room:state.temperature_c for room in state.room_temps}
 state.humidity_pct=rng.uniform(10,95);state.co2_ppm=rng.uniform(300,5000);state.pm25_ugm3=rng.uniform(0,150);state.occupant_activity=rng.choice(['idle','sleeping','cooking','exercising'])
 weather=scenario.weather_trace(1)[0];a=agent.select_action(encode_state(state,weather,DeviceState()));probe_hist[a]+=1
 final=RuleLayer().apply(agent.decode_action(a,DeviceState(),state,weather),state,weather)
 probe_diffs+=int(asdict(final)!=asdict(deterministic_devices(state,weather)))
summary={'checkpoint_sha256':digest_before,'checkpoint_unchanged':digest_before==model_digest(MODEL),'model':'data/models/improved-v2.pt','action_mode':agent.action_mode,'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'threads':2,'steps':144,'runs':runs,'probe':{'count':1000,'seed':712026,'action_histogram':dict(probe_hist),'final_command_differences':probe_diffs,'note':'Independent synthetic state probe, not closed-loop performance and not physical measurements.'}}
(ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
with (ROOT/'per-episode.csv').open('w',newline='') as stream:
 w=csv.DictWriter(stream,fieldnames=allrows[0].keys());w.writeheader();w.writerows(allrows)
print('ALL DONE',flush=True)
