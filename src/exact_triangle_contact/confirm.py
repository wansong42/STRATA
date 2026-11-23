import json,sys,subprocess
from pathlib import Path
from copy import deepcopy
from fractions import Fraction as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from run_io import PYTHON,now,digest,read,save


def cases():
    xy=[[0,0,0],[2,0,0],[0,2,0]]
    def m(name,a,b,va=(0,0,0),vb=(0,0,0),span=(-1,1)):
        return {'schema':'rigid-pair/v1','snapshot_id':name,'mode_id':'declared','t_closed':list(span),'pieces':[{'id':i,'vertices':v,'velocity':list(w)} for i,v,w in [('left',a,va),('right',b,vb)]]}
    skew=m('oblique_nonorthogonal_point_event',xy,[[2,0,0],[3,0,1],[3,1,1]],vb=(0,1,0))
    slide=m('coplanar_slide',xy,[[3,0,0],[5,0,0],[3,2,0]],vb=(-1,0,0),span=(0,6))
    rotate=deepcopy(skew); rotate['snapshot_id']='winding_swap_and_translation'; rotate['pieces'].reverse()
    for obj in rotate['pieces']:
        obj['vertices']=[[z+9,-x+3,y-4] for x,y,z in reversed(obj['vertices'])]; x,y,z=obj['velocity']; obj['velocity']=[z,-x,y]
    return [(skew,['0','0']), (m('skew_nonparallel_permanent_segment',xy,[[0,0,-1],[2,0,1],[0,2,-1]]),['-1','1']), (slide,['1','5']),
            (m('coplanar_overlap_boxes_but_disjoint_triangles',xy,[[2,2,0],[2,'1/2',0],['1/2',2,0]]),None),
            (m('parallel_small_gap',xy,[[x,y,'1/100000000000000000000'] for x,y,z in xy]),None),
            (m('endpoint_enters_sliding_triangle',[[0,0,0]],[[1,0,0],[3,0,0],[1,2,0]],vb=(-1,0,0),span=(0,4)),['1','3']), (rotate,['0','0']),
            (m('degenerate_rejection',xy,[[0,0,0],[1,1,1],[2,2,2]]),'INVALID')]


def producer():
    from compile_pair import compile_checked
    from wire import Invalid
    result=[]
    for m,expected in cases():
        try: out=compile_checked(m)
        except Invalid as e: out={'status':'INVALID_INPUT','reason':str(e)}
        result.append({'model':m,'result':out})
    print(json.dumps(result))


def main():
    from verify_pair import grade,vertices
    from pair_format import fingerprint
    q=read(HERE.parent/'kernel_r3/kernel_qualification.json'); protocol=read(HERE/'protocol.json')
    if q['grade']!='OFFLINE_KERNEL_QUALIFIED': raise SystemExit('KERNEL_QUALIFICATION_BLOCKED')
    cmd=[PYTHON,'-B','-X','utf8',str(Path(__file__).resolve()),'--producer']; ret=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW,shell=False)
    if ret.returncode: raise SystemExit(ret.stderr.decode('utf8','replace'))
    rows=json.loads(ret.stdout); expected=cases(); ids=[m['snapshot_id'] for m,_ in expected]
    if ids!=protocol['cases'] or len(rows)!=len(expected) or not rows: raise SystemExit('WRONG_OR_EMPTY_CASES')
    controls=[]; passed=True
    for row,(m,ans) in zip(rows,expected):
        out=row['result']; row['expected']=ans; ok=row['model']==m
        if ans=='INVALID':
            rejection=grade(m,{'schema':'rigid-pair-window/v1','input_hash':fingerprint(m),'t_closed':None}); row['fresh_check']=rejection
            ok=ok and out['status']=='INVALID_INPUT' and not rejection['ok'] and rejection['reason']=='affine rank is below two'
        else:
            claim=out.get('window'); checked=grade(m,claim); row['fresh_check']=checked
            ok=ok and out['status']=='VERIFIED_CLOSED_CONTACT_WINDOW' and claim['t_closed']==ans and checked['ok']
            if ans is not None and checked['ok']:
                for witness in checked['reference']['endpoint_witnesses']:
                    points=[]; t=F(witness['t'])
                    for obj,key in zip(m['pieces'],('left_weights','right_weights')):
                        w=list(map(F,witness[key])); ok=ok and len(w)==len(obj['vertices']) and sum(w)==1 and all(x>=0 for x in w)
                        points.append([sum(weight*F(v[k]) for weight,v in zip(w,obj['vertices']))+t*F(obj['velocity'][k]) for k in range(3)])
                    ok=ok and points[0]==points[1]
            bad={**claim,'t_closed':None if ans is not None else ['0','0']}; stale=deepcopy(m); stale['mode_id']='new-mode'; corrupt={**claim,'t_closed':['-989','989']}
            for label,model,c in [('inverted_presence',m,bad),('stale_mode',stale,claim),('wrong_extrema',m,corrupt)]:
                g=grade(model,c); controls.append({'case':m['snapshot_id'],'mutation':label,'actual':g}); ok=ok and not g['ok']
        row['ok']=ok; passed=passed and ok
    passed=passed and 'compile_pair' not in sys.modules
    data=dict(at_utc=now(),scope='Generated rational primitive pairs in protocol.json, complete shared parameter intervals only; no real input consumed and no whole-network or physical guarantee',enumeration='All registered pairs and named corrupt presence/endpoint/staleness variants, one independent verifier-only parent, actual SAT producer child',protocol_sha256=digest((HERE/'protocol.json').read_bytes()),code_sha256={p.name:digest(p.read_bytes()) for p in HERE.glob('*.py')},case_count=len(rows),tamper_controls=len(controls),checks_passed=passed,rows=rows,controls=controls,verifier_imported_proposer='compile_pair' in sys.modules,producer_command=cmd,producer_exit_code=ret.returncode,exit_code=int(not passed),status='SCOPED_PAIR_COMPILER_DEVELOPMENT_MATCH' if passed else 'PAIR_COMPILER_BLOCKED',mathematics='The separating axes of two 3D triangles are their face normals and cross products of edge directions; in the coplanar face add in-plane edge normals. Point membership uses face and edge normals. With pure translations every fixed projection shifts affinely. The independent polytope uses two simplices of barycentric weights, equality of their translated world positions, and a closed bounded t interval. The polytope is bounded: nonempty implies a vertex, and extrema occur at vertices even in a lower-dimensional affine hull. Enumerate all full-rank active subsets by exact elimination; no vertices means empty. A positive window includes feasible endpoint weights. Shared code: schema/exact number parsing/hash only.',limitations=['fixed shape/orientation and pure translation; not the general affine halfspace family','no inference of candidate completeness or certified triangulation of observations','not wired to rectangular network solver; no network L/U from this module','independent implementation but internal known-answer development cases, not held-out confirmation or originality evidence'])
    save(HERE/'result.json',data); print(json.dumps({k:data[k] for k in ('status','case_count','tamper_controls','checks_passed')})); raise SystemExit(int(not passed))


if __name__=='__main__':
    if '--producer' in sys.argv: producer()
    else: main()
