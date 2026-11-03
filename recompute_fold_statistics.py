"""Recompute the two approximate paired t-tests from observed five-fold F1."""
import argparse,csv,math,statistics
from pathlib import Path

def calculate(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 assert len(rows)==5,'Exactly five outer folds are required'
 assert len({r['outer_fold'] for r in rows})==5,'Repeated outer fold'
 results=[]
 for baseline in ['ShuffleNetV2','MobileNetV3-Large']:
  diff=[float(r['TF-ShuffleNetV2-Small'])-float(r[baseline]) for r in rows]
  mean=statistics.mean(diff);sd=statistics.stdev(diff)
  se=sd/math.sqrt(5);t=mean/se
  # Two-sided Student-t survival for df=4, evaluated in closed form.
  s=abs(t)/math.sqrt(t*t+4)
  p=(1-s)**2*(s+2)/2
  critical=2.7764451051977987
  results.append(dict(comparison='Small - '+baseline,difference_pp=mean,difference_sd_pp=sd,t=t,df=4,p_raw=p,ci95_low_pp=mean-critical*se,ci95_high_pp=mean+critical*se))
 order=sorted(range(2),key=lambda i:results[i]['p_raw']);prev=0
 for rank,i in enumerate(order):
  prev=max(prev,min(1,(2-rank)*results[i]['p_raw']));results[i]['p_Holm']=prev
 return results

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 rows=calculate(a.input)
 with Path(a.output).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('Saved two fold-level approximate comparisons; df=4; Holm family size=2.')
