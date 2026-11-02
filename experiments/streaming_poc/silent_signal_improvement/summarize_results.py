"""Create aggregate machine-readable results from completed per-video traces."""
import csv, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiment_results/streaming_poc/silent_signal_improvement'
ORDER=['negative_174439','negative_174454','candidate_175204','candidate_174955','candidate_175021','candidate_175132','candidate_175228']
GT={'negative_174439':'Human-reviewed negative','negative_174454':'Human-reviewed negative'}
MODES={'baseline':'baselineTriggered','same_hand_repeated':'strictTriggeredTracks','same_hand_stable_open2':'stableTriggeredTracks','same_hand_tolerant':'robustTriggeredTracks'}
rows=[]
for name in ORDER:
    trace=json.loads((OUT/'traces'/f'{name}.json').read_text(encoding='utf-8'))
    hits={mode:[item['frame'] for item in trace if item[key]] for mode,key in MODES.items()}
    rows.append({'case':name,'groundTruth':GT.get(name,'Candidate; GT unconfirmed'),'frames':len(trace),'hits':hits})
(OUT/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['case','groundTruth','frames',*MODES])
    for row in rows:w.writerow([row['case'],row['groundTruth'],row['frames'],*[','.join(map(str,row['hits'][m])) for m in MODES]])
