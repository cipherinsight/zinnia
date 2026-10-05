from pathlib import Path
import textwrap,json
p=Path(__file__).parent
c={
'clip':'return np.clip(2 * x + 5, 0, 255)',
'minmax':'return np.minimum(np.maximum(2*x+5, 0), 255)',
'where':'y=2*x+5\nreturn np.where(y < 0, 0, np.where(y > 255, 255, y))',
'mask_update':'y=2*x+5\ny[y < 0]=0\ny[y > 255]=255\nreturn y',
'indexed_if':'''out=np.zeros_like(x)
for i in range(len(x)):
    y=2*x[i]+5
    if y < 0:
        out[i]=0
    elif y > 255:
        out[i]=255
    else:
        out[i]=y
return out''',
'while':'out=np.zeros_like(x)\ni=0\nwhile i<len(x):\n    out[i]=min(255,max(0,2*x[i]+5))\n    i+=1\nreturn out',
'comprehension':'return np.array([min(255,max(0,2*v+5)) for v in x])',
'append':'out=[]\nfor v in x:\n    out.append(min(255,max(0,2*v+5)))\nreturn np.asarray(out)',
'local_helper':'''def clamp(v):
    if v < 0:
        return 0
    if v > 255:
        return 255
    return v
return np.array([clamp(2*v+5) for v in x])''',
'map_lambda':'return np.array(list(map(lambda v: min(255,max(0,2*v+5)), x)))',
'fromiter':'return np.fromiter((min(255,max(0,2*v+5)) for v in x), dtype=np.int64, count=len(x))',
'vectorize':'f=np.vectorize(lambda v: min(255,max(0,2*v+5)))\nreturn f(x)',
'piecewise':'y=2*x+5\nreturn np.piecewise(y, [y<0, y>255], [0,255,lambda v:v])',
'select':'y=2*x+5\nreturn np.select([y<0,y>255],[0,255],default=y)',
'copyto':'y=2*x+5\nnp.copyto(y,0,where=y<0)\nnp.copyto(y,255,where=y>255)\nreturn y',
'putmask':'y=2*x+5\nnp.putmask(y,y<0,0)\nnp.putmask(y,y>255,255)\nreturn y',
'ufunc_out':'y=2*x+5\nnp.maximum(y,0,out=y)\nnp.minimum(y,255,out=y)\nreturn y',
'clip_method':'return (2*x+5).clip(min=0,max=255)',
'fmin_fmax':'return np.fmin(np.fmax(2*x+5,0),255)',
'abs_identity':'y=2*x+5\nreturn (255+np.abs(y)-np.abs(y-255))//2',
'median_stack':'y=2*x+5\nreturn np.median(np.stack([np.zeros_like(x),y,np.full_like(x,255)]),axis=0).astype(int)',
'boolean_arithmetic':'y=2*x+5\nreturn (y>=0)*(y<=255)*y+(y>255)*255',
'enumerate_ternary':'out=np.empty_like(x)\nfor i,v in enumerate(x):\n    y=2*v+5\n    out[i]=0 if y<0 else 255 if y>255 else y\nreturn out',
'continue_loop':'''out=np.zeros_like(x)
for i in range(len(x)):
    y=2*x[i]+5
    if y < 0:
        continue
    if y > 255:
        out[i]=255
        continue
    out[i]=y
return out''',
}
rows=[]
for name,body in c.items():
    source='def kernel(x):\n'+textwrap.indent(body,'    ')+'\n'
    (p/(name+'.py')).write_text(source)
    rows.append({'name':name,'path':name+'.py','intent':'clip(2*x+5,0,255), same shape','status':'untested'})
(p/'inventory.json').write_text(json.dumps(rows,indent=2)+'\n')
