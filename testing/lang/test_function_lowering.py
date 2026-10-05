"""Semantic regression tests for ordinary source helpers and finite callbacks."""
import ast
import copy
import textwrap

import numpy as np
import pytest

from zinnia import ZKCircuit, ZKChip
from zinnia.compile.function_lowering import FunctionLoweringError, lower_function_source


def run_python(source, args, lowered=False):
    source=textwrap.dedent(source)
    env={'np':np}
    if lowered:
        entry,helpers=lower_function_source(source,'kernel')
        for _,helper in helpers:exec(helper,env)
        exec(entry,env)
    else:exec(source,env)
    return env['kernel'](*args)


def check_python(source,args):
    expected=run_python(source,copy.deepcopy(args))
    actual=run_python(source,copy.deepcopy(args),True)
    np.testing.assert_equal(actual,expected)
    return actual


def check_circuit(source,args,expected):
    """Check result and rejected output without requiring compiler integration."""
    tree=ast.parse(textwrap.dedent(source));fn=next(f for f in tree.body if isinstance(f,ast.FunctionDef) and f.name=='kernel')
    for argument,value in zip(fn.args.args,args):
        shape=' '.join(map(str,np.asarray(value).shape))
        argument.annotation=ast.parse(f'Integer[Array,"{shape}"]',mode='eval').body
    expected=np.asarray(expected,dtype=np.int64)
    shape=' '.join(map(str,expected.shape))
    fn.args.args.append(ast.arg(arg='check_expected',annotation=ast.parse(f'Integer[Array,"{shape}"]',mode='eval').body))
    class Returns(ast.NodeTransformer):
        def visit_FunctionDef(self,node):return node
        def visit_Return(self,node):
            result=ast.parse('check_actual=np.asarray(None)\nassert np.all(check_actual==check_expected)\nreturn').body
            result[0].value.args[0]=node.value
            return result
    visitor=Returns();body=[]
    for statement in fn.body:
        result=visitor.visit(statement);body.extend(result if isinstance(result,list) else [result])
    fn.body=body
    typed=ast.unparse(ast.fix_missing_locations(tree))
    entry,helpers=lower_function_source(typed,'kernel')
    chips=[ZKChip.from_source(name,source) for name,source in helpers]
    circuit=ZKCircuit.from_source('kernel',entry,chips=chips);circuit.compile()
    assert circuit.prove(*copy.deepcopy(args),expected,backend='mock').proof_bytes_hex=='mock_satisfied'
    wrong=expected.copy();wrong.flat[0]+=1
    assert circuit.prove(*copy.deepcopy(args),wrong,backend='mock').proof_bytes_hex!='mock_satisfied'


def test_helper_captures_are_read_at_call_and_arguments_evaluated_once():
    source='''
    def kernel(x):
        offset=3
        def add(v, scale=2):
            return v*scale+offset
        offset=7
        return np.array([add(v,scale=3) for v in x])
    '''
    args=[np.array([-3,0,5])]
    expected=check_python(source,args)
    check_circuit(source,args,expected)


def test_nested_capture_and_helper_alias():
    source='''
    def kernel(x):
        offset=9
        def outer(v):
            def inner(w):
                return w+v+offset
            return inner(2)
        alias=outer
        return np.array([alias(v) for v in x])
    '''
    args=[np.array([-5,1,4])]
    expected=check_python(source,args);check_circuit(source,args,expected)


def test_global_helper_uses_global_constant_not_same_named_entry_local():
    source='''
    offset=4
    def add(v):
        return v+offset
    def kernel(x):
        offset=99
        return np.array([add(v) for v in x])
    '''
    args=[np.array([1,2,3])]
    expected=check_python(source,args);check_circuit(source,args,expected)


def test_global_helper_can_call_another_global_helper():
    source='''
    def double(v):
        return 2*v
    def add(v):
        return double(v)+1
    def kernel(x):
        return np.array([add(v) for v in x])
    '''
    args=[np.array([1,2])]
    expected=check_python(source,args);check_circuit(source,args,expected)


@pytest.mark.parametrize('expression',[
    '(lambda v: 3*v+2)(x)',
    'np.array(list(map(lambda v: 3*v+2, x)))',
    'np.array(list(map(lambda a,b: a+b, x, x)))',
])
def test_lambda_and_map(expression):
    source='def kernel(x):\n    return '+expression
    args=[np.array([-4,1,5])]
    expected=check_python(source,args);check_circuit(source,args,expected)


def test_filter_map_fusion_preserves_late_predicate_capture():
    source='''
    def kernel(x):
        threshold=0
        pairs=filter(lambda p:p[0]>threshold,zip(x,x))
        threshold=2
        return np.array([sum(map(lambda p:p[0]*p[1],pairs))])
    '''
    args=[np.array([-3,1,4,5])]
    expected=check_python(source,args);check_circuit(source,args,expected)


def test_iterator_outer_iterable_is_bound_before_reassignment():
    source='''
    def kernel(x):
        pairs=filter(lambda p:p[0]>0,zip(x,x))
        x=np.array([20,30])
        return np.array([sum(map(lambda p:p[0],pairs))])
    '''
    args=[np.array([-3,4])]
    expected=check_python(source,args);check_circuit(source,args,expected)


def test_fromiter_stops_at_count_without_extra_callback_evaluation():
    source='''
    def kernel(x):
        seen=[]
        def record(v):
            seen.append(v)
            return v+1
        result=np.fromiter((record(v) for v in x),dtype=np.int64,count=2)
        return result,seen
    '''
    result,seen=check_python(source,[np.array([1,2,3,4])])
    assert list(seen)==[1,2]
    np.testing.assert_array_equal(result,[2,3])


@pytest.mark.parametrize('count',[0,1,3,-1])
def test_fromiter_count(count):
    source=f'def kernel(x):\n    return np.fromiter((v*2 for v in x),dtype=np.int64,count={count})'
    args=[np.array([2,3,4])]
    expected=check_python(source,args)
    if count!=0:check_circuit(source,args,expected)


@pytest.mark.parametrize('cache',[False,True])
def test_vectorize_dtype_and_evaluation_count(cache):
    source=f'''
    def kernel(x):
        seen=[]
        def record(v):
            seen.append(v)
            return int(v) if v==1 else v/2
        f=np.vectorize(record,cache={cache})
        result=f(x)
        return result,seen
    '''
    result,seen=check_python(source,[np.array([1,3,5])])
    assert result.dtype.kind=='i'
    assert len(seen)==(3 if cache else 4)


def test_vectorize_numeric_circuit():
    source='''
    def kernel(x):
        f=np.vectorize(lambda v:min(255,max(0,2*v+5)))
        return f(x)
    '''
    args=[np.array([-512,-3,-2,125,4096])]
    expected=check_python(source,args);check_circuit(source,args,expected)


def test_piecewise_last_condition_wins_and_default_callback_gets_only_uncovered():
    source='''
    def kernel(x):
        seen=[]
        def record(v):
            seen.append(v.copy())
            return v+1
        result=np.piecewise(x,[x<2,x>0],[10,20,record])
        return result,seen
    '''
    result,seen=check_python(source,[np.array([-2,0,1,3])])
    np.testing.assert_array_equal(result,[10,10,20,20]);assert seen==[]


def test_piecewise_numeric_circuit():
    source='''
    def kernel(x):
        y=2*x+5
        return np.piecewise(y,[y<0,y>255],[0,255,lambda v:v])
    '''
    args=[np.array([-512,-3,-2,0,125,4096])]
    expected=check_python(source,args);check_circuit(source,args,expected)


@pytest.mark.parametrize('axis',[0,1,-1])
@pytest.mark.parametrize('callback',['np.sum(row)','np.array([np.sum(row),np.max(row)])'])
def test_apply_along_axis_shape_and_axis_placement(axis,callback):
    source=f'''
    def kernel(x):
        def transform(row):
            return {callback}
        return np.apply_along_axis(transform,{axis},x)
    '''
    args=[np.arange(6).reshape(2,3)]
    expected=check_python(source,args);check_circuit(source,args,expected)


def test_comprehension_target_can_shadow_helper_name():
    source='''
    def kernel(x):
        f=lambda v:v+1
        return np.array([f for f in x])
    '''
    check_python(source,[np.array([1,2])])


@pytest.mark.parametrize('source,match',[
    ('def kernel(x):\n    v=0\n    def f():\n        nonlocal v\n        v+=1\n    f()\n    return x','nonlocal'),
    ('def kernel(x):\n    f=lambda v:v+1\n    f=lambda v:v+2\n    return f(x)','Rebinding'),
    ('def kernel(x):\n    def f(v):\n        return f(v-1)\n    return f(x)','Recursive'),
    ('def kernel(x):\n    def f(v=[]):\n        return v\n    return f()','Mutable'),
    ('def kernel(x):\n    def f(v=x):\n        return v\n    return f()','definition-time'),
    ('def kernel(x):\n    f=lambda v:v\n    return f','escape|Function values'),
])
def test_unsupported_function_semantics_are_explicit(source,match):
    with pytest.raises(FunctionLoweringError,match=match):lower_function_source(source,'kernel')


def test_direct_vectorize_and_explicit_output_dtype():
    source='''
    def kernel(x):
        return np.vectorize(lambda v:v+1,otypes='i')(x)
    '''
    args=[np.array([1,3,5])]
    expected=check_python(source,args);check_circuit(source,args,expected)


def test_piecewise_scalar_choices_can_be_bound_names():
    source='''
    def kernel(x):
        low=0
        high=255
        y=2*x+5
        return np.piecewise(y,[y<0,y>255],[low,high,lambda v:v])
    '''
    args=[np.array([-512,-2,125,4096])]
    expected=check_python(source,args);check_circuit(source,args,expected)


def test_module_constant_does_not_replace_comprehension_target():
    source='''
    v=99
    def kernel(x):
        return np.array([v+1 for v in x])
    '''
    args=[np.array([1,2,3])]
    expected=check_python(source,args);check_circuit(source,args,expected)


def test_entry_requires_decorator_survives_normalization():
    source='''
    @zk_circuit(requires=[lambda x: x[0] > 0])
    def kernel(x: Integer[Array, "1"]):
        assert x[0] > 0
    '''
    entry,_=lower_function_source(source,'kernel')
    assert ast.dump(ast.parse(entry).body[0].decorator_list[0])==ast.dump(ast.parse(textwrap.dedent(source)).body[0].decorator_list[0])


def test_legacy_result_marker_in_a_captured_helper():
    source='''
    def kernel(x):
        offset=4
        def add(v):
            _zinnia_result=v+offset
        return np.array([add(v) for v in x])
    '''
    check_circuit(source,[np.array([1,2])],np.array([5,6]))


def test_local_capturing_helper_can_call_explicit_chip():
    source='''
    def kernel(x: Integer[Array, "2"]):
        offset=4
        def add(v):
            return double(v)+offset
        assert add(x[0])==6
        assert add(x[1])==8
    '''
    entry,helpers=lower_function_source(source,'kernel')
    chips=[ZKChip.from_source('double','def double(v):\n    return v*2')]
    chips.extend(ZKChip.from_source(name,body) for name,body in helpers)
    circuit=ZKCircuit.from_source('kernel',entry,chips=chips);circuit.compile()
    assert circuit.prove(np.array([1,2]),backend='mock').proof_bytes_hex=='mock_satisfied'
    assert circuit.prove(np.array([2,3]),backend='mock').proof_bytes_hex!='mock_satisfied'


@pytest.mark.parametrize('prefix,body', [
    ('import numpy as numpy', 'return numpy.clip(2*x+5,0,255)'),
    ('import numpy', 'return numpy.clip(2*x+5,0,255)'),
    ('from numpy import clip', 'return clip(2*x+5,0,255)'),
    ('from numpy import clip as clamp', 'return clamp(2*x+5,0,255)'),
    ('', 'import numpy as arraylib\nreturn arraylib.clip(2*x+5,0,255)'),
    ('', 'from numpy import clip as clamp\nreturn clamp(2*x+5,0,255)'),
    ('import numpy as arraylib', 'def clamp(v):\n    return arraylib.clip(v,0,255)\nreturn clamp(2*x+5)'),
    ('', 'def clamp(v):\n    from numpy import clip\n    return clip(v,0,255)\nreturn clamp(2*x+5)'),
])
def test_numpy_import_aliases_compile(prefix,body):
    source=prefix+'\ndef kernel(x):\n'+textwrap.indent(body,'    ')
    args=[np.array([-512,-3,-2,0,125,4096])]
    expected=check_python(source,args);check_circuit(source,args,expected)


@pytest.mark.parametrize('body',[
    'clip=4\nreturn x+clip',
    'return np.array([clip+1 for clip in x])',
    'def helper(clip):\n    return clip+1\nreturn helper(x)',
    'return (lambda clip:clip+1)(x)',
])
def test_numpy_import_does_not_override_shadowing(body):
    source='from numpy import clip\ndef kernel(x):\n'+textwrap.indent(body,'    ')
    args=[np.array([1,2,3])]
    expected=check_python(source,args);check_circuit(source,args,expected)


@pytest.mark.parametrize('source',[
    'from numpy import *\ndef kernel(x):\n    return x',
    'def kernel(x):\n    if len(x)>0:\n        import numpy as num\n    return num.asarray(x)',
])
def test_unsupported_imports_are_explicit(source):
    with pytest.raises(FunctionLoweringError,match='import'):
        lower_function_source(source,'kernel')

@pytest.mark.parametrize('source',[
    'import random\ndef kernel(x):\n    return x',
    'def kernel(x):\n    import random\n    return x',
    'def kernel(x):\n    if len(x)>0:\n        import random\n    return x',
])
def test_out_of_scope_imports_are_rejected(source):
    from zinnia.compile.import_scope import OutOfScopeImportError
    with pytest.raises(OutOfScopeImportError,match='out of scope: imports random'):
        lower_function_source(source,'kernel')

@pytest.mark.parametrize('prefix,body',[
    ('from numpy.lib.stride_tricks import sliding_window_view as win','return win(x,3).sum(axis=1)'),
    ('import numpy.lib.stride_tricks as strides','return strides.sliding_window_view(x,3).sum(axis=1)'),
    ('import numpy.lib.stride_tricks','return numpy.lib.stride_tricks.sliding_window_view(x,3).sum(axis=1)'),
    ('from numpy.lib import stride_tricks as strides','return strides.sliding_window_view(x,3).sum(axis=1)'),
    ('','from numpy.lib.stride_tricks import sliding_window_view\nreturn sliding_window_view(x,3).sum(axis=1)'),
    ('','def window_sum(a):\n    from numpy.lib.stride_tricks import sliding_window_view as win\n    return win(a,3).sum(axis=1)\nreturn window_sum(x)'),
])
def test_sliding_window_imports(prefix,body):
    source=prefix+'\ndef kernel(x):\n'+textwrap.indent(body,'    ')
    args=[np.array([1,2,3,4,5,6])]
    check_circuit(source,args,check_python(source,args))


def test_sliding_window_import_does_not_override_local_parameter():
    source='''from numpy.lib.stride_tricks import sliding_window_view as win
def kernel(x):
    def helper(win):
        return win + 1
    return helper(x)
'''
    args=[np.array([1,2,3])]
    check_circuit(source,args,check_python(source,args))
