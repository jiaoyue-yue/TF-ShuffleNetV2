"""Validate real five-fold assignments against the fixed 17-group record map."""
import argparse,csv,collections,json
from pathlib import Path

def read(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def check(records,assignments):
 meta=read(records);byid={r['record_id']:r for r in meta}
 assert len(meta)==len(byid)==2485,'Record count or duplicate ID'
 assert {r['source_group'] for r in meta}=={'SG%02d'%i for i in range(1,18)},'Expected SG01-SG17'
 assert collections.Counter(r['y_true'] for r in meta)=={'0':1754,'1':731}
 a=read(assignments);folds=sorted({r['outer_fold'] for r in a})
 assert folds==['1','2','3','4','5'],'Expected outer folds 1-5'
 assert len(a)==12425
 outer=collections.Counter();summary=[]
 for fold in folds:
  rows=[r for r in a if r['outer_fold']==fold]
  ids=[r['record_id'] for r in rows]
  assert len(ids)==len(set(ids))==2485,'Each fold must assign all records exactly once'
  assert set(ids)==set(byid),'Record IDs do not match'
  sets={role:set() for role in ['train','inner_val','outer_test']}
  labels={role:set() for role in sets};counts=collections.Counter()
  for row in rows:
   role=row['role'];assert role in sets,'Invalid role'
   m=byid[row['record_id']]
   for key in ['source_group','relative_path','label']:assert row[key]==m[key],key+' mapping differs'
   assert m['label']==('clean' if m['y_true']=='0' else 'infested')
   if role=='outer_test':assert row['outer_test_fold']==fold
   sets[role].add(m['source_group']);labels[role].add(m['y_true']);counts[role]+=1
   if role=='outer_test':outer[row['record_id']]+=1
  roles=list(sets)
  for i,x in enumerate(roles):
   assert labels[x]=={'0','1'},'Both classes required in '+x
   for y in roles[i+1:]:assert sets[x].isdisjoint(sets[y]),'Source group leakage'
  summary.append({'outer_fold':fold,'counts':dict(counts),'groups':{k:sorted(v) for k,v in sets.items()}})
 assert set(outer)==set(byid) and set(outer.values())=={1},'Each record must be tested once'
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--assignments',required=True);a=p.parse_args()
 print(json.dumps(check(a.records,a.assignments),ensure_ascii=False,indent=2))
