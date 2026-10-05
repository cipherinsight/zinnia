"""Bilingual participant contracts for revision 2. No reference solutions served."""
import numpy as np
from .tasks import TASKS, fixtures, signatures, oracle

ORDER=[t.name for t in TASKS]
BUDGETS=dict(zip(ORDER,[600,720,900,1080,1080,1200]))
TITLES=dict(zip(ORDER,[
 ['Calibrate batches and measure channel energy','批量校准并计算通道能量'],
 ['Centered products between channels','计算通道间的中心化乘积'],
 ['Weighted distances within batches','计算批次内的加权距离'],
 ['Apply multichannel image filters','应用多通道图像滤波器'],
 ['Channel covariance within image blocks','计算图像区块内的通道协方差'],
 ['Update masked grids and mix channels','按掩码更新网格并混合通道'],
]))
DESCRIPTIONS=dict(zip(ORDER,[
 ["x has shape [B, T, C]: B batches, T readings per batch, and C channels. gain and bias each contain C values. First compute a[b,t,c] = clip(x[b,t,c] * gain[c] + bias[c], 0, 255).\n\nFor each batch and channel separately, find the minimum of these calibrated readings over T. Subtract that minimum from each calibrated reading, square the differences, and sum over T. Return one energy per batch and channel, with shape [B, C].\n\nClip before finding the minimum. Do not combine different batches or channels. A batch with one reading has zero energy in every channel.",
 "x 的形状为 [B, T, C]：共有 B 个批次，每批有 T 次读数，每次包含 C 个通道。gain 和 bias 各有 C 个值。先计算 a[b,t,c] = clip(x[b,t,c] * gain[c] + bias[c], 0, 255)，即把结果限制在 0 到 255。\n\n对每个批次、每个通道，分别找出这 T 个校准后读数的最小值。每个校准后读数减去该最小值，将差平方，再沿 T 求和。返回形状为 [B, C] 的数组，每个位置是对应批次、对应通道的能量。\n\n先限制数值范围，再求最小值。不同批次、不同通道不要混合。只有一次读数的批次，各通道能量都为零。"],
 ["x has shape [B, T, C]: B batches of T observations, each with C channels. For each batch and channel compute its sum over T, then define z[b,t,c] = T * x[b,t,c] - sum_t(x[b,t,c]). This is integer-scaled centering; no division is needed.\n\nReturn a [B, C, C] array where result[b,c,d] = sum_t(z[b,t,c] * z[b,t,d]). Compute every channel pair, including the diagonal, within the same batch. Do not divide the result by T or T-1. With T=1 the entire output is zero.\n\nFor one batch with rows [1,2] and [3,6], the centered rows are [-2,-4] and [2,4], giving [[8,16],[16,32]].",
 "x 的形状为 [B, T, C]：B 个批次，每批 T 次观测，每次有 C 个通道。先对每个批次、每个通道沿 T 求和，再计算 z[b,t,c] = T * x[b,t,c] - sum_t(x[b,t,c])。这是用整数缩放的中心化，不需要除法。\n\n返回形状为 [B, C, C] 的数组，其中 result[b,c,d] = sum_t(z[b,t,c] * z[b,t,d])。在同一批次内计算所有通道对，包括对角线。结果不要除以 T 或 T-1。T=1 时，输出全部为零。\n\n例如，一个批次的两行为 [1,2] 和 [3,6]，中心化后为 [-2,-4] 和 [2,4]，结果是 [[8,16],[16,32]]。"],
 ["x has shape [B, N, D] and y has shape [B, M, D]. Each batch contains N and M points respectively; each point has D coordinates. weight contains D nonnegative coordinate weights. D is not fixed to 3.\n\nReturn shape [B, N, M], with result[b,i,j] = sum_d(weight[d] * (x[b,i,d] - y[b,j,d])²). Compare points only within the same batch, and sum all weighted squared coordinate differences into one value per point pair. Do not take a square root. N and M may differ; zero weights and repeated points are allowed.\n\nFor points [1,2] and [3,5] with weights [2,0], the result is 2*(1-3)² + 0*(2-5)² = 8.",
 "x 的形状为 [B, N, D]，y 的形状为 [B, M, D]。每个批次分别包含 N 个和 M 个点，每个点有 D 个坐标。weight 包含 D 个非负坐标权重。D 不固定为 3。\n\n返回形状 [B, N, M]，其中 result[b,i,j] = sum_d(weight[d] * (x[b,i,d] - y[b,j,d])²)。只比较同一批次内的点，把所有坐标的加权平方差相加，每对点得到一个值。不要开平方。N 和 M 可以不同；权重可以为零，点可以重复。\n\n例如，[1,2] 与 [3,5] 的权重为 [2,0] 时，结果为 2*(1-3)² + 0*(2-5)² = 8。"],
 ["x has shape [B, H, W, C]: B images with H rows, W columns, and C channels. filters has shape [K, U, V, C]: K filters, each covering U rows, V columns, and all C channels. U<=H and V<=W.\n\nSlide each filter one pixel at a time over each image. At each position multiply matching filter and image values and sum over both spatial axes and all channels. Use only positions where the entire filter fits; no padding. Use filter coefficients in their given order; do not flip the filter.\n\nReturn [B, H-U+1, W-V+1, K], where result[b,i,j,k] = sum over u,v,c of x[b,i+u,j+v,c] * filters[k,u,v,c]. Keep images and output filters separate. This operation is also called valid cross-correlation.",
 "x 的形状为 [B, H, W, C]，表示 B 张 H 行、W 列、C 通道的图像。filters 的形状为 [K, U, V, C]：K 个滤波器，每个覆盖 U 行、V 列和全部 C 个通道。U<=H，V<=W。\n\n在每张图像上，以一个像素为步长滑动每个滤波器。每个位置上，将滤波器与图像对应的值相乘，再对两个空间轴和全部通道求和。只计算滤波器完全落在图像内的位置，不填充边界。滤波器系数按给定顺序使用，不要翻转。\n\n返回形状 [B, H-U+1, W-V+1, K]，其中 result[b,i,j,k] 是 x[b,i+u,j+v,c] * filters[k,u,v,c] 对 u、v、c 的总和。不同图像、不同输出滤波器分别计算。这也称为有效互相关。"],
 ["x has shape [B, H, W, C]. H and W are positive even numbers. Split each image into non-overlapping 2 by 2 blocks starting at the top-left. Treat the four pixels in each block as four observations of C channels.\n\nWithin each block, let s[c] be the sum of its four values for channel c, and z[p,c] = 4*x[p,c] - s[c], where p identifies a pixel in that block. For every channel pair (c,d), sum z[p,c]*z[p,d] over the four pixels.\n\nReturn shape [B, H/2, W/2, C, C], preserving image and block order. Include diagonal and off-diagonal entries. Do not mix blocks or divide the result; these are scaled covariance values. A constant channel within a block has a zero row and column in that block's result.",
 "x 的形状为 [B, H, W, C]，H 和 W 都是正偶数。从左上角开始，把每张图像划分为互不重叠的 2×2 区块。把每个区块的四个像素看作 C 个通道的四次观测。\n\n在每个区块内，s[c] 是通道 c 的四个值之和，z[p,c] = 4*x[p,c] - s[c]，p 表示区块中的一个像素。对每一对通道 (c,d)，将四个像素的 z[p,c]*z[p,d] 相加。\n\n返回形状 [B, H/2, W/2, C, C]，保持图像和区块的顺序。包括对角线和非对角线元素。不同区块不要混合，结果不做除法；这是缩放后的协方差。如果某个通道在区块内为常数，其对应的行和列都应为零。"],
 ["x has shape [B, H, W, C] and values from 0 to 255. mask is an [H, W] array of zeros and ones, shared by all batches. mix is a [C, C] matrix: mix[c,d] weights source channel c into destination channel d. Perform exactly three rounds.\n\nKeep the outermost rows and columns unchanged. Also keep every cell with mask[i,j]=0 unchanged. For each remaining cell, first compute one spatial average per source channel: (4*center + above + below + left + right) // 8. Round down before mixing channels.\n\nFor destination channel d, sum average[c]*mix[c,d] over c and clip that sum to [0,255]. All spatial reads in a round must use the values at the start of that round, including frozen neighbors. Apply the new values together after the round. Return the full [B,H,W,C] array after three rounds. If H<3 or W<3, return the input values unchanged.",
 "x 的形状为 [B, H, W, C]，值在 0 到 255 之间。mask 是所有批次共享的 [H, W] 数组，只含 0 和 1。mix 为 [C, C] 矩阵，mix[c,d] 是从源通道 c 到目标通道 d 的权重。恰好执行三轮。\n\n最外侧的行和列始终保持不变。mask[i,j]=0 的单元也保持不变。对其余单元，先分别计算每个源通道的空间平均值：(4*自身 + 上 + 下 + 左 + 右) // 8。先向下取整，再混合通道。\n\n目标通道 d 的值是所有 average[c]*mix[c,d] 的和，再限制到 [0,255]。同一轮的所有空间读取都必须使用本轮开始时的值，包括被冻结的邻居。整轮结束后一起应用新值。返回三轮后的完整 [B,H,W,C] 数组。如果 H<3 或 W<3，输入值全部保持不变。"],
]))
HINT_TEXT=[
 ['Calibrate using channel broadcasting, then reduce the time axis for the minimum and the squared sum. Keep the time axis when subtracting the minimum.','先沿通道广播校准，再沿时间轴求最小值和平方和。相减时保留时间轴以便广播。'],
 ['Compute channel sums along the observation axis first. Then use an outer product reduction, a matrix product for each batch, or direct accumulation.','先沿观测轴求通道总和，再对外积归约、逐批矩阵相乘，或直接累加。'],
 ['Insert a point axis into each input to form point pairs, then reduce over coordinates. Processing one pair at a time avoids a large intermediate tensor.','分别插入点的轴来组合点对，再沿坐标轴归约。也可以逐对处理，避免较大的中间张量。'],
 ['Extract each valid image patch and reduce its product with the filters. Alternatively, accumulate one filter offset at a time over the output grid.','可提取每个有效图像区块，与滤波器相乘后归约；也可每次固定一个滤波器偏移，在输出网格上累加。'],
 ['Group the four pixels of each block using slices or reshape and transpose. Center within each block, then reduce channel products over its four pixels.','可通过切片，或 reshape 与 transpose，把每个区块的四个像素分组。在区块内中心化，再对四个像素的通道乘积求和。'],
 ['Keep separate current and next arrays for each round. Compute spatial averages before channel mixing, and copy frozen cells into the next array.','每轮使用独立的当前数组和下一轮数组。先计算空间平均值，再混合通道；被冻结的单元复制到下一轮数组。'],
]
TOPICS=[['broadcast','minmax','sum','clamp'],['sum','broadcast','matmul'],['expand','broadcast','sum','matmul'],['slicing','sum','matmul'],['reshape','transpose','matmul','sum'],['updates','clamp','matmul']]
# Topic IDs refer to the shared bilingual manual.
HINTS={n:([text[0]+' A straightforward implementation using only loops (nested where needed) is also accepted.',text[1]+' 也可以采用最直接的纯循环实现，需要时使用嵌套循环。'],topics) for n,text,topics in zip(ORDER,HINT_TEXT,TOPICS)}
REQUIREMENTS=dict(zip(ORDER,[
 [['Clip before finding the minimum.','shape [B, C]','Do not combine different batches or channels.'],['先限制数值范围，再求最小值。','形状为 [B, C]','不同批次、不同通道不要混合。']],
 [['no division is needed','including the diagonal','Do not divide the result by T or T-1.'],['不需要除法','包括对角线','结果不要除以 T 或 T-1。']],
 [['D is not fixed to 3.','only within the same batch','Do not take a square root.'],['D 不固定为 3。','只比较同一批次内的点','不要开平方。']],
 [['no padding','do not flip the filter','all channels','[B, H-U+1, W-V+1, K]'],['不填充边界','不要翻转','全部通道','[B, H-U+1, W-V+1, K]']],
 [['non-overlapping','[B, H/2, W/2, C, C]','Do not mix blocks or divide the result'],['互不重叠','[B, H/2, W/2, C, C]','不同区块不要混合，结果不做除法']],
 [['exactly three rounds','Keep the outermost rows and columns unchanged.','mask[i,j]=0 unchanged','Round down before mixing channels.','values at the start of that round','If H<3 or W<3, return the input values unchanged.'],['恰好执行三轮','最外侧的行和列始终保持不变。','mask[i,j]=0 的单元也保持不变。','先向下取整，再混合通道。','本轮开始时的值','如果 H<3 或 W<3，输入值全部保持不变。']],
]))

def contract(task):return next(t for t in TASKS if t.name==task)
def names(task):return list(contract(task).arguments)
RANKS={t.name:[len(s) for s in signatures(t,t.profiles[0])[0].values()] for t in TASKS}
OUT_RANK={t.name:t.output_rank for t in TASKS}

def starter(task,condition):
    if condition=='zinnia':return f"def kernel({', '.join(names(task))}):\n    # Write your implementation here.\n    raise NotImplementedError(\"Implement kernel\")\n"
    args=', '.join(f'{n}: &Array{r}<i64>' for n,r in zip(names(task),RANKS[task]))
    return f'fn kernel({args}) -> Array{OUT_RANK[task]}<i64> {{\n    // Write your implementation here.\n    todo!("Implement kernel")\n}}\n'

def cases(task,hidden=False):
    t=contract(task)
    select=[(0,i) for i in ([1,3,4,5,7] if hidden else [0,2,6])]
    if hidden:select += [(p,i) for p in range(1,len(t.profiles)) for i in (2,6)]
    return [dict(zip(t.arguments,[v.tolist() for v in fixtures(t,t.profiles[p])[i]])) for p,i in select]

def validate_input(task,data):
    t=contract(task)
    if not isinstance(data,dict) or set(data)!=set(t.arguments):raise ValueError('Input must contain exactly: '+', '.join(t.arguments))
    arrays=[]
    for n,r in zip(t.arguments,RANKS[task]):
        a=np.asarray(data[n])
        if a.dtype.kind not in 'iu' or a.ndim!=r or not all(a.shape) or a.size>1024 or max(a.shape)>16:raise ValueError(f'{n}: expected a nonempty rank-{r} integer array, dimensions <=16, at most 1024 values')
        low,high=((-16,16) if n not in ('gain','weight','mask','mix') else {'gain':(0,4),'weight':(0,4),'mask':(0,1),'mix':(-2,2)}[n])
        if task=='channel_energy_v2' and n=='x':low,high=-64,64
        if task=='masked_diffusion_v2' and n=='x':low,high=0,255
        if np.any(a<low) or np.any(a>high):raise ValueError(f'{n} values must be in [{low}, {high}]')
        arrays.append(a.astype(np.int64))
    x=arrays[0];p={}
    if task=='channel_energy_v2':p=dict(zip(('B','T','C'),x.shape))
    elif task=='centered_gram_v2':p=dict(zip(('B','T','C'),x.shape))
    elif task=='weighted_distances_v2':p=dict(zip(('B','N','D'),x.shape));p['M']=arrays[1].shape[1]
    elif task=='convolution_v2':
        p=dict(zip(('B','H','W','C'),x.shape));p.update(zip(('K','U','V'),arrays[1].shape[:3]))
        if p['U']>p['H'] or p['V']>p['W']:raise ValueError('Filters must fit inside the image')
    else:
        p=dict(zip(('B','H','W','C'),x.shape))
        if task=='block_covariance_v2' and (p['H']%2 or p['W']%2):raise ValueError('Image height and width must be even')
    shapes,out=signatures(t,p)
    if any(a.shape!=s for a,s in zip(arrays,shapes.values())):raise ValueError('Batch/channel/weight/mask dimensions do not match')
    if np.prod(out)>4096:raise ValueError('Output must contain at most 4096 values')
    return arrays

def answer(task,data):
    t=contract(task);a=validate_input(task,data)
    # Oracles derive loop bounds from the actual arrays, not profile constants.
    x=a[0]
    if task in ('channel_energy_v2','centered_gram_v2'):p=dict(zip(('B','T','C'),x.shape))
    elif task=='weighted_distances_v2':p=dict(zip(('B','N','D'),x.shape));p['M']=a[1].shape[1]
    else:
        p=dict(zip(('B','H','W','C'),x.shape))
        if task=='convolution_v2':p.update(zip(('K','U','V'),a[1].shape[:3]))
    return oracle(t,p,a).tolist()

def task_payload(task,condition):
    t=contract(task)
    bounds={n:'[-16,16]' for n in t.arguments}
    for n,r in [('gain','[0,4]'),('weight','[0,4]'),('mask','{0,1}'),('mix','[-2,2]')]:
        if n in bounds:bounds[n]=r
    if task=='masked_diffusion_v2':bounds['x']='[0,255]'
    if task=='channel_energy_v2':bounds['x']='[-64,64]'
    ranges=', '.join(n+': '+r for n,r in bounds.items())
    details=['All inputs are integer arrays. Value ranges: '+ranges+'. Each dimension is positive and at most 16; each input has at most 1,024 values. The supplied inputs satisfy the shape constraints. Return the result from kernel.',
             '输入均为整数数组。数值范围：'+ranges+'。各维度为 1 到 16，每个输入最多 1,024 个元素。输入满足题目规定的形状约束。请从 kernel 返回结果。']
    return {'id':task,'titles':TITLES[task],'descriptions':DESCRIPTIONS[task],'hints':HINTS[task][0],'manual_topics':HINTS[task][1],'requirements':REQUIREMENTS[task],'input_details':details,'starter':starter(task,condition),'examples':[{'name':f'Case {i+1}','input':v,'expected':answer(task,v)} for i,v in enumerate(cases(task))],'ranks':RANKS[task],'output_rank':OUT_RANK[task],'arguments':names(task)}
