from pair_format import fingerprint,need,unpack


def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def project_pair(m):
    (lo,hi),((left,lv),(right,rv))=unpack(m)
    axes=set(); edges=[]
    def axis(x):
        divisor=next((a for a in x if a),None)
        if divisor: axes.add(tuple(a/divisor for a in x))
    for v in (left,right):
        ee=[sub(b,a) for a,b in zip(v,v[1:]+v[:1])] if len(v)==3 else []
        if ee:
            n=cross(ee[0],ee[1]); need(any(n),'degenerate triangle'); axis(n)
            for edge in ee: axis(cross(n,edge))
        edges.append(ee)
    for a in edges[0]:
        for b in edges[1]: axis(cross(a,b))
    window=True
    for direction in sorted(axes):
        a=[dot(direction,p) for p in left]; b=[dot(direction,p) for p in right]; s=dot(direction,sub(lv,rv))
        for c,d in ((s,max(b)-min(a)),(-s,max(a)-min(b))):
            if c>0: hi=min(hi,d/c)
            elif c<0: lo=max(lo,d/c)
            elif d<0: window=False
        if lo>hi or not window:
            window=False; break
    return dict(schema='rigid-pair-window/v1',input_hash=fingerprint(m),t_closed=[str(lo),str(hi)] if window else None)


def compile_checked(m):
    from verify_pair import grade
    proposed=project_pair(m)
    checked=grade(m,proposed)
    if not checked['ok']: return {'status':'CERTIFICATE_INVALID','window':None,'reason':checked['reason']}
    return {'status':'VERIFIED_CLOSED_CONTACT_WINDOW','window':proposed,'independent_reference':checked['reference']}
