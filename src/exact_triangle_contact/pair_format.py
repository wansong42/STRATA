from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'kernel_r3'))
from wire import Invalid, exact, fingerprint, need


def vector(raw):
    need(type(raw) is list and len(raw)==3,'three world coordinates required')
    return tuple(exact(x) for x in raw)


def unpack(m):
    need(type(m) is dict and set(m)=={'schema','snapshot_id','mode_id','pieces','t_closed'} and m['schema']=='rigid-pair/v1','pair schema')
    need(all(type(m[k]) is str and m[k] for k in ('snapshot_id','mode_id')),'missing identity')
    need(type(m['t_closed']) is list and len(m['t_closed'])==2,'closed interval required')
    lo,hi=map(exact,m['t_closed']); need(lo<=hi,'inverted interval')
    need(type(m['pieces']) is list and len(m['pieces'])==2,'pair required')
    out=[]; ids=set()
    for obj in m['pieces']:
        need(type(obj) is dict and set(obj)=={'id','vertices','velocity'},'piece schema')
        name=obj['id']; need(type(name) is str and name and name not in ids,'piece identity'); ids.add(name)
        need(type(obj['vertices']) is list and len(obj['vertices']) in (1,3),'triangle or fixed endpoint required')
        v=tuple(vector(p) for p in obj['vertices']); w=vector(obj['velocity'])
        need(len(v)!=1 or not any(w),'moving endpoint unsupported')
        out.append((v,w))
    need(any(len(v)==3 for v,w in out),'two endpoints do not create an edge')
    return (lo,hi),out
