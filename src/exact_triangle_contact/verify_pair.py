from fractions import Fraction as F
from itertools import combinations
from pair_format import fingerprint,need,unpack
from wire import exact,Invalid,Exhausted


def reduce_rows(rows,n):
    a=[list(row) for row in rows]; r=0
    for k in range(n):
        p=next((i for i in range(r,len(a)) if a[i][k]),None)
        if p is None: continue
        a[r],a[p]=a[p],a[r]; val=a[r][k]; a[r]=[z/val for z in a[r]]
        for i in range(len(a)):
            if i!=r and a[i][k]:
                coeff=a[i][k]; a[i]=[v-coeff*w for v,w in zip(a[i],a[r])]
        r+=1
    if any(row[n] for row in a[r:]): return None
    return a[:r]


def vertices(m):
    (lo,hi),((a,u),(b,v))=unpack(m)
    for pts in (a,b):
        if len(pts)==3:
            diffs=[[y-x for x,y in zip(pts[0],p)]+[F(0)] for p in pts[1:]]
            need(len(reduce_rows(diffs,3))==2,'affine rank is below two')
    p,q=len(a),len(b); n=p+q+1; z=F(0); one=F(1)
    equations=[[one]*p+[z]*q+[z,one],[z]*p+[one]*q+[z,one]]
    for k in range(3): equations.append([x[k] for x in a]+[-x[k] for x in b]+[u[k]-v[k],z])
    base=reduce_rows(equations,n)
    if base is None: return None
    active=[]
    for k in range(p+q):
        row=[z]*(n+1); row[k]=one; active.append(row)
    for x in (lo,hi): active.append([z]*(n-1)+[one,x])
    extrema={}
    for subset in combinations(active,n-len(base)):
        rr=reduce_rows(base+list(subset),n)
        if rr is None or len(rr)!=n: continue
        vals=[row[-1] for row in rr]; t=vals[-1]
        if not lo<=t<=hi or min(vals[:-1])<0: continue
        extrema[t]=dict(t=str(t),left_weights=list(map(str,vals[:p])),right_weights=list(map(str,vals[p:p+q])))
    if not extrema: return None
    first,last=min(extrema),max(extrema)
    return dict(t_closed=[str(first),str(last)],endpoint_witnesses=[extrema[first],extrema[last]])


def grade(m,claim):
    try:
        need(type(claim) is dict and set(claim)=={'schema','input_hash','t_closed'} and claim['schema']=='rigid-pair-window/v1','window schema')
        need(claim['input_hash']==fingerprint(m),'stale model')
        ref=vertices(m); raw=claim['t_closed']
        if ref is None: need(raw is None,'false nonempty contact')
        else:
            need(type(raw) is list and len(raw)==2,'missing interval')
            need([exact(x,'proof_bits') for x in raw]==list(map(F,ref['t_closed'])),'wrong extremum')
        return {'ok':True,'reference':ref}
    except (Invalid,Exhausted) as e: return {'ok':False,'reason':str(e)}
