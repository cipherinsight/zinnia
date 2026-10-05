"""Simpler task contracts with explicit implementation guides and scalar oracles."""
import numpy as np

ORDER=['channel_energy_v5','cross_products_v5','weighted_distances_v5','image_filter_v5','block_energy_v5','masked_diffusion_v5']
BUDGETS=dict.fromkeys(ORDER,600)
ARGS=[['x','gain','bias'],['x','y'],['x','y','weight'],['x','filter'],['x'],['x','mask']]
RANKS=dict(zip(ORDER,[[2,1,1],[2,2],[2,2,1],[2,2],[3],[2,2]]))
OUT_RANK=dict(zip(ORDER,[1,2,2,2,3,2]))
TITLES=dict(zip(ORDER,[['Calibrate and sum squared differences','校准并汇总平方差'],['Centered channel products','中心化通道乘积'],['Weighted distances between 3D points','三维点的加权平方距离'],['Apply an image filter','应用图像滤波器'],['Squared differences in image blocks','图像区块内的平方差'],['Update a masked grid','更新带掩码的网格']]))
DESCRIPTIONS=dict(zip(ORDER,[
[
'''x has shape [T,C], with T readings and C channels. gain and bias each have shape [C], with one value per channel. The output is a vector [C] with one result per channel.

1. Calculate v = x * gain + bias using broadcasting. For element [t,c], this means v[t,c] = x[t,c] * gain[c] + bias[c]. v has shape [T,C].
2. Replace values below 0 in v with 0 and values above 255 with 255; leave other values unchanged. Call this new matrix a, with shape [T,C].
3. Find the minimum along axis 0 of a, giving m with shape [C]. Use m to calculate b = a - m by broadcasting. b has shape [T,C].
4. Square b elementwise, then sum along axis 0. Return the resulting vector [C].

Example: for a single channel with readings [1,3,5], gain 2 and bias -3, v=[-1,3,7], a=[0,3,7], and m=0. The output is [0²+3²+7²]=[58].''',
'''x 的形状为 [T,C]，包含 T 次读数、C 个通道。gain 和 bias 的形状均为 [C]，为每个通道分别提供一个乘数和一个偏移量。输出是形状为 [C] 的向量，每个通道对应一个结果。

1. 利用广播计算 v = x * gain + bias。对于位置 [t,c]，具体计算为 v[t,c] = x[t,c] * gain[c] + bias[c]。得到的 v 形状为 [T,C]。
2. 将 v 中小于 0 的改为 0，大于 255 的改为 255，其余值不变。将处理后的矩阵命名为 a，形状仍为 [T,C]。
3. 对 a 沿轴 0 求最小值，得到形状为 [C] 的 m。用这个 m，利用广播计算 b = a - m，b 的形状为 [T,C]。
4. 将 b 逐元素平方，再沿轴 0 求和。返回得到的 [C] 向量。

例如，只有一个通道，读数为 [1,3,5]，gain=2，bias=-3，则 v=[-1,3,7]，a=[0,3,7]，m=0，输出为 [0²+3²+7²]=[58]。'''
],
[
'''x has shape [T,C] and y has shape [T,D]. Both matrices contain readings at the same T time points, with C channels in x and D channels in y. C and D may differ. First calculate two intermediate matrices z and w, then combine them into an output of shape [C,D].

1. Sum x along axis 0 to obtain sx of shape [C]. Use sx to calculate z = T*x - sx by broadcasting. z has shape [T,C].
2. Sum y along axis 0 to obtain sy of shape [D]. Use sy to calculate w = T*y - sy by broadcasting. w has shape [T,D].
3. Transpose z to shape [C,T], then calculate the matrix product of this transposed matrix and w. Return the product directly, with shape [C,D].

Example: x=[[1,2],[3,6]], y=[[2],[5]] gives sx=[4,8], sy=[7], z=[[-2,-4],[2,4]], and w=[[-3],[3]]. The matrix product is [[12],[24]].''',
'''x 的形状为 [T,C]，y 的形状为 [T,D]。两个矩阵记录同样 T 个时刻的读数；x 有 C 个通道，y 有 D 个通道，C 和 D 可以不同。先分别得到中间矩阵 z 和 w，再用它们计算形状为 [C,D] 的输出。

1. 对 x 沿轴 0 求和，得到形状为 [C] 的 sx。用这个 sx，利用广播计算 z = T*x - sx，z 的形状为 [T,C]。
2. 对 y 沿轴 0 求和，得到形状为 [D] 的 sy。用这个 sy，利用广播计算 w = T*y - sy，w 的形状为 [T,D]。
3. 将 z 转置为形状 [C,T]，然后计算转置后的矩阵与 w 的矩阵乘积。直接返回这个形状为 [C,D] 的乘积。

例如 x=[[1,2],[3,6]]，y=[[2],[5]]，则 sx=[4,8]，sy=[7]，z=[[-2,-4],[2,4]]，w=[[-3],[3]]，矩阵乘积为 [[12],[24]]。'''
],
[
'''Each row of x is a three-dimensional point; x has shape [N,3]. Each row of y is another three-dimensional point; y has shape [M,3]. N and M may differ. weight is a vector of length 3 with a nonnegative weight for each coordinate. The output has shape [N,M]: output[i,j] is the weighted squared distance between point x[i] and point y[j].

1. Add a size-1 dimension to x to obtain x3 with shape [N,1,3]. Add a size-1 dimension to y to obtain y3 with shape [1,M,3]. Calculate d = x3 - y3 using broadcasting. d has shape [N,M,3]; d[i,j,k] is the difference in coordinate k between the two points.
2. Square d elementwise to obtain sq. Calculate q = sq * weight using broadcasting: q[i,j,k] = sq[i,j,k] * weight[k]. Coordinate 0 uses weight[0], coordinate 1 uses weight[1], and coordinate 2 uses weight[2]. q has shape [N,M,3].
3. Sum q along axis 2. Return the resulting matrix [N,M].

Example: for points [1,2,0] and [3,5,4] with weight=[2,0,1], the result is 2×(-2)² + 0×(-3)² + 1×(-4)² = 24.''',
'''x 的每行是一个三维点，形状为 [N,3]；y 的每行也是一个三维点，形状为 [M,3]，N 和 M 可以不同。weight 是长度为 3 的向量，为每个坐标提供一个非负权重。输出形状为 [N,M]，其中 output[i,j] 表示 x 的第 i 个点与 y 的第 j 个点之间的加权平方距离。

1. 给 x 增加一个长度为 1 的维度，得到形状为 [N,1,3] 的 x3；给 y 增加一个长度为 1 的维度，得到形状为 [1,M,3] 的 y3。利用广播计算 d = x3 - y3，得到形状为 [N,M,3] 的 d，其中 d[i,j,k] 是这两个点在第 k 个坐标上的差。
2. 将 d 逐元素平方，得到 sq。利用广播计算 q = sq * weight，即 q[i,j,k] = sq[i,j,k] * weight[k]。第 0 个坐标乘 weight[0]，第 1 个坐标乘 weight[1]，第 2 个坐标乘 weight[2]。q 的形状为 [N,M,3]。
3. 对 q 沿轴 2 求和，返回得到的 [N,M] 矩阵。

例如，两个点为 [1,2,0] 和 [3,5,4]，weight=[2,0,1]，则结果为 2×(-2)² + 0×(-3)² + 1×(-4)² = 24。'''
],
[
'''x is a single-channel image with H rows and W columns. filter is a matrix with U rows and V columns, where U<=H and V<=W. Each output[i,j] is calculated from the U-by-V region of x whose top-left corner is (i,j).

1. Create output with shape [H-U+1,W-V+1]. Visit each output position (i,j), where 0 <= i < H-U+1 and 0 <= j < W-V+1. Perform steps 2–3 for this position.
2. Read patch = x[i:i+U,j:j+V] from the original x. Slice stops are exclusive, so patch contains U rows and V columns and has the same shape as filter.
3. Multiply patch and filter elementwise into a new array products: products[u,v] = patch[u,v] * filter[u,v]. Sum all elements of products and store the result in output[i,j]. Keep x unchanged so the next output position reads the original values.
4. After processing every output position, return output.

Example: x=[[1,2,3],[4,5,6]], filter=[[1,0],[0,2]] gives output [[11,14]].''',
'''x 是有 H 行、W 列的单通道图像。filter 是有 U 行、V 列的矩阵，U<=H、V<=W。每个输出值 output[i,j]，都使用 x 中以 (i,j) 为左上角、大小为 U 行 V 列的区域来计算。

1. 创建形状为 [H-U+1,W-V+1] 的 output。遍历每个输出位置 (i,j)，范围为 0 <= i < H-U+1、0 <= j < W-V+1，对这个位置执行第 2–3 步。
2. 从原始 x 中读取 patch = x[i:i+U,j:j+V]。切片终点不包含在内，因此 patch 包含 U 行、V 列，形状与 filter 相同。
3. 将 patch 与 filter 逐元素相乘，存入新数组 products，即 products[u,v] = patch[u,v] * filter[u,v]。对 products 的全部元素求和，将结果存入 output[i,j]。保持 x 不变，以便下一个输出位置仍使用原始数值。
4. 处理完所有输出位置后，返回 output。

例如 x=[[1,2,3],[4,5,6]]，filter=[[1,0],[0,2]]，输出为 [[11,14]]。'''
],
[
'''x is an image with shape [H,W,C]: H rows, W columns, and C values at each pixel. H and W are positive even numbers. Divide the image into non-overlapping 2-by-2 blocks of pixels. Each block produces one output position containing C values, so output has shape [H/2,W/2,C].

1. Create the output array output with shape [H/2,W/2,C]. Calculate each output[i,j,:], meaning all C channel values at row i, column j of output. Use two loops with 0 <= i < H/2 and 0 <= j < W/2; perform steps 2–4 for each pair (i,j).
2. To calculate the current output[i,j,:], first read a 2-by-2 region from the input array x. Its top-left corner is row 2*i, column 2*j of x. Create a matrix p of shape [4,C] and fill it as follows for each channel c:
   p[0,c] = x[2*i,   2*j,   c]
   p[1,c] = x[2*i,   2*j+1, c]
   p[2,c] = x[2*i+1, 2*j,   c]
   p[3,c] = x[2*i+1, 2*j+1, c]
   Now p contains one pixel per row and one channel per column. Step 3 uses this p to calculate the current output[i,j,:].
3. Find the minimum along axis 0 of p, giving m with shape [C]. Use m to calculate b = p - m by broadcasting. Square b elementwise, then sum along axis 0 to obtain a vector r of shape [C].
4. Store r in output at position (i,j): output[i,j,c] = r[c] for every channel c.
5. After processing every output position, return output.

Example: for a single-channel block [[1,2],[3,4]], p has rows [1], [2], [3], [4] and m=[1]. Its output value is 0²+1²+2²+3²=14.''',
'''x 是形状为 [H,W,C] 的图像：有 H 行、W 列，每个像素包含 C 个通道值。H 和 W 均为正偶数。将图像划分为互不重叠的 2×2 像素区域，下面称为“区块”。每个区块计算出 C 个结果，放在一个输出位置，因此 output 的形状为 [H/2,W/2,C]。

1. 创建形状为 [H/2,W/2,C] 的输出数组 output。接下来逐个计算 output[i,j,:]，也就是 output 第 i 行、第 j 列的全部 C 个通道值。用两层循环遍历 i、j，范围为 0 <= i < H/2、0 <= j < W/2；每次循环执行第 2–4 步。
2. 为计算当前的 output[i,j,:]，先从输入数组 x 读取一个 2×2 区域。这个区域的左上角是 x 的第 2*i 行、第 2*j 列，包含这两行和这两列交叉处的四个像素。创建形状为 [4,C] 的矩阵 p，并按下面的关系填入数值（对每个通道 c 执行）：
   p[0,c] = x[2*i,   2*j,   c]
   p[1,c] = x[2*i,   2*j+1, c]
   p[2,c] = x[2*i+1, 2*j,   c]
   p[3,c] = x[2*i+1, 2*j+1, c]
   填好后，p 的每行对应一个像素，每列对应一个通道。第 3 步使用这个 p 计算当前 output[i,j,:] 的值。
3. 对 p 沿轴 0 求最小值，得到形状为 [C] 的 m。用这个 m，利用广播计算 b = p - m。将 b 逐元素平方，再沿轴 0 求和，得到形状为 [C] 的向量 r。
4. 将 r 存入 output 的位置 (i,j)：对每个通道 c，令 output[i,j,c] = r[c]。
5. 处理完所有输出位置后，返回 output。

例如，单通道区块为 [[1,2],[3,4]] 时，p 的四行依次为 [1]、[2]、[3]、[4]，m=[1]。这个区块的输出值为 0²+1²+2²+3²=14。'''
],
[
'''x is a nonnegative integer grid [H,W]. mask has the same shape and contains 0 or 1. The output is the grid after three rounds of updates. current holds the grid at the start of a round; next_grid holds its updated values.

1. Copy x into current. Repeat steps 2–4 for three rounds.
2. At the start of each round, make an independent copy of current called next_grid. During this round, read values from current and write updates to next_grid.
3. Visit every cell (i,j). If 1 <= i < H-1 and 1 <= j < W-1 and mask[i,j]=1, calculate total = 4*current[i,j] + current[i-1,j] + current[i+1,j] + current[i,j-1] + current[i,j+1]. Use all four neighbor values, including border neighbors and neighbors whose mask is 0. Divide the entire total by 8, round down, and store the result in next_grid[i,j]. Otherwise, leave next_grid[i,j] at its copied value. When H<3 or W<3, no cell satisfies the condition, so next_grid remains unchanged.
4. After visiting every cell, replace current with next_grid.
5. After all three rounds have finished, return current with shape [H,W].

Example: an updated cell with center 10 and neighbors 1,2,3,4 receives floor(50/8)=6 for that round.''',
'''x 是非负整数网格 [H,W]，mask 形状相同，只包含 0 和 1。输出是经过三轮更新后的网格。下面用 current 保存每轮开始时的网格，用 next_grid 保存这一轮更新后的网格。

1. 将 x 复制为 current。重复第 2–4 步，共进行三轮。
2. 每轮开始时，将 current 独立复制为 next_grid。本轮从 current 读取数值，将更新结果写入 next_grid。
3. 遍历每个单元 (i,j)。如果满足 1 <= i < H-1 且 1 <= j < W-1，且 mask[i,j]=1，则计算 total = 4*current[i,j] + current[i-1,j] + current[i+1,j] + current[i,j-1] + current[i,j+1]。四个邻居的值都参与计算，包括位于边界或 mask 为 0 的邻居。将整个 total 除以 8 后向下取整，把结果存入 next_grid[i,j]。否则，next_grid[i,j] 保留复制时的值。H<3 或 W<3 时，没有单元满足上述条件，next_grid 保持不变。
4. 遍历完所有单元后，用 next_grid 替换 current。
5. 三轮全部结束后，返回形状为 [H,W] 的 current。

例如，一个需要更新的单元自身为 10，四个邻居为 1、2、3、4，该轮结果为 50 除以 8 向下取整，即 6。'''
]
]))
REQUIREMENTS=dict(zip(ORDER, [[['v = x * gain + bias',
   'Replace values below 0 in v with 0 and values above 255 with 255',
   'minimum along axis 0 of a',
   'b = a - m',
   'sum along axis 0'],
  ['v = x * gain + bias', '小于 0 的改为 0，大于 255 的改为 255，其余值不变', '对 a 沿轴 0 求最小值', 'b = a - m', '沿轴 0 求和']],
 [['Sum x along axis 0',
   'z = T*x - sx',
   'Sum y along axis 0',
   'w = T*y - sy',
   'Transpose z to shape [C,T]',
   'matrix product',
   'shape [C,D]'],
  ['对 x 沿轴 0 求和', 'z = T*x - sx', '对 y 沿轴 0 求和', 'w = T*y - sy', '将 z 转置为形状 [C,T]', '矩阵乘积', '形状为 [C,D]']],
 [['three-dimensional point',
   'vector of length 3',
   'd = x3 - y3',
   'q[i,j,k] = sq[i,j,k] * weight[k]',
   'Sum q along axis 2'],
  ['三维点', '长度为 3 的向量', 'd = x3 - y3', 'q[i,j,k] = sq[i,j,k] * weight[k]', '对 q 沿轴 2 求和']],
 [['[H-U+1,W-V+1]',
   'patch = x[i:i+U,j:j+V]',
   'Slice stops are exclusive',
   'new array products',
   'Sum all elements of products',
   'Keep x unchanged'],
  ['[H-U+1,W-V+1]', 'patch = x[i:i+U,j:j+V]', '切片终点不包含在内', '新数组 products', '对 products 的全部元素求和', '保持 x 不变']],
 [['non-overlapping 2-by-2 blocks',
   '0 <= i < H/2 and 0 <= j < W/2',
   'one pixel per row and one channel per column',
   'minimum along axis 0 of p',
   'b = p - m',
   'sum along axis 0',
   'output[i,j,c] = r[c]'],
  ['互不重叠的 2×2 像素区域',
   '0 <= i < H/2、0 <= j < W/2',
   '每行对应一个像素，每列对应一个通道',
   '对 p 沿轴 0 求最小值',
   'b = p - m',
   '沿轴 0 求和',
   'output[i,j,c] = r[c]']],
 [['three rounds',
   'independent copy',
   '1 <= i < H-1 and 1 <= j < W-1',
   'mask[i,j]=1',
   'Divide the entire total by 8, round down',
   'read values from current and write updates to next_grid',
   'Otherwise, leave next_grid[i,j] at its copied value'],
  ['共进行三轮',
   '独立复制',
   '1 <= i < H-1 且 1 <= j < W-1',
   'mask[i,j]=1',
   '将整个 total 除以 8 后向下取整',
   '三轮全部结束后',
   '从 current 读取数值，将更新结果写入 next_grid',
   '否则，next_grid[i,j] 保留复制时的值']]]))
TOPICS=dict(zip(ORDER,[['broadcast','clamp','minmax','sum'],['sum','transpose','matmul'],['expand','broadcast','sum'],['slicing','sum'],['slicing','reshape','minmax','sum'],['updates','slicing']]))

def names(task):return ARGS[ORDER.index(task)]
def starter(task,condition):
 if condition=='zinnia':return 'def kernel('+', '.join(names(task))+'):\n    # Write your implementation here.\n    raise NotImplementedError("Implement kernel")\n'
 params=', '.join(f'{n}: &Array{r}<i64>' for n,r in zip(names(task),RANKS[task]))
 return f'fn kernel({params}) -> Array{OUT_RANK[task]}<i64> {{\n    // Write your implementation here.\n    todo!("Implement kernel")\n}}\n'

def cases(task,hidden=False):
 rng=np.random.default_rng(510+ORDER.index(task)+(100 if hidden else 0))
 profiles=[(3,2),(1,1),(2,4)] if not hidden else [(1,3),(4,1),(2,3),(4,4)]
 out=[]
 for index,(n,m) in enumerate(profiles):
  arr=lambda shape:rng.integers(-8,9,size=shape).tolist()
  if task==ORDER[0]:v={'x':arr((n,m)),'gain':rng.integers(0,5,m).tolist(),'bias':arr((m,))}
  elif task==ORDER[1]:v={'x':arr((n,m)),'y':arr((n,m+1))}
  elif task==ORDER[2]:v={'x':arr((n,3)),'y':arr((m,3)),'weight':[2,0,1] if index==0 else rng.integers(0,5,3).tolist()}
  elif task==ORDER[3]:v={'x':arr((n+1,m+2)),'filter':arr((min(n,2),min(m+1,3)))}
  elif task==ORDER[4]:v={'x':arr((2*n,2*m,1+index%3))}
  else:
   h,w=(n+2,m+2) if index==0 else (n,m)
   v={'x':rng.integers(0,256,(h,w)).tolist(),'mask':rng.integers(0,2,(h,w)).tolist()}
   if h>2 and w>2:v['mask'][1][1]=1
  out.append(v)
 if hidden:
  if task==ORDER[0]:out += [{'x':[[-64,64],[64,-64]],'gain':[4,4],'bias':[-16,16]}]
  if task==ORDER[3]:out += [{'x':[[3]],'filter':[[2]]}]
  if task==ORDER[4]:out += [{'x':[[[3,7]]*2]*2}]
  if task==ORDER[5]:out += [{'x':[[1,2,3],[4,100,6],[7,8,9]],'mask':[[0,0,0],[0,0,0],[0,0,0]]}]
 return out

def validate_input(task,data):
 if not isinstance(data,dict) or set(data)!=set(names(task)):raise ValueError('Input must contain exactly: '+', '.join(names(task)))
 arrays=[]
 for name,rank in zip(names(task),RANKS[task]):
  a=np.asarray(data[name]);lo,hi=(0,255) if task==ORDER[5] else (-64,64)
  if name=='mask':lo,hi=0,1
  if name in ('gain','weight'):lo,hi=0,4
  if name=='bias':lo,hi=-16,16
  if a.dtype.kind not in 'iu' or a.ndim!=rank or not all(a.shape) or a.size>1024 or max(a.shape)>16:raise ValueError(f'{name}: expected nonempty rank-{rank} integer array, dimensions <=16 and at most 1024 values')
  if np.any(a<lo) or np.any(a>hi):raise ValueError(f'{name} values must be in [{lo},{hi}]')
  arrays.append(a.astype(np.int64))
 x=arrays[0]
 if task==ORDER[0] and any(a.shape!=(x.shape[1],) for a in arrays[1:]):raise ValueError('gain and bias must have one entry per channel')
 if task==ORDER[1] and x.shape[0]!=arrays[1].shape[0]:raise ValueError('x and y must have the same number of rows')
 if task==ORDER[2] and (x.shape[1]!=3 or arrays[1].shape[1]!=3 or arrays[2].shape!=(3,)):raise ValueError('Points and weights must have exactly three coordinates')
 if task==ORDER[3] and any(f>s for f,s in zip(arrays[1].shape,x.shape)):raise ValueError('filter must fit inside x')
 if task==ORDER[4] and (x.shape[0]%2 or x.shape[1]%2):raise ValueError('H and W must be even')
 if task==ORDER[5] and x.shape!=arrays[1].shape:raise ValueError('mask and x must have the same shape')
 return arrays

def answer(task,data):
 a=validate_input(task,data);x=a[0]
 if task==ORDER[0]:
  out=[]
  for c in range(x.shape[1]):
   vals=[min(255,max(0,int(x[t,c])*int(a[1][c])+int(a[2][c]))) for t in range(x.shape[0])];low=min(vals);out.append(sum((v-low)**2 for v in vals))
  return out
 if task==ORDER[1]:
  y=a[1];t=x.shape[0]
  return [[sum((t*int(x[i,c])-sum(int(v) for v in x[:,c]))*(t*int(y[i,d])-sum(int(v) for v in y[:,d])) for i in range(t)) for d in range(y.shape[1])] for c in range(x.shape[1])]
 if task==ORDER[2]:return [[sum(int(a[2][d])*(int(p[d])-int(q[d]))**2 for d in range(3)) for q in a[1]] for p in x]
 if task==ORDER[3]:
  f=a[1];u,v=f.shape
  return [[sum(int(x[i+r,j+c])*int(f[r,c]) for r in range(u) for c in range(v)) for j in range(x.shape[1]-v+1)] for i in range(x.shape[0]-u+1)]
 if task==ORDER[4]:
  h,w,c=x.shape;out=np.zeros((h//2,w//2,c),dtype=np.int64)
  for i,j,k in np.ndindex(out.shape):
   vals=[int(x[2*i+r,2*j+s,k]) for r in range(2) for s in range(2)];low=min(vals);out[i,j,k]=sum((v-low)**2 for v in vals)
  return out.tolist()
 cur=x.tolist();h,w=x.shape
 for _ in range(3):
  nxt=[row[:] for row in cur]
  for i in range(1,h-1):
   for j in range(1,w-1):
    if a[1][i,j]:nxt[i][j]=(4*cur[i][j]+cur[i-1][j]+cur[i+1][j]+cur[i][j-1]+cur[i][j+1])//8
  cur=nxt
 return cur

def task_payload(task,condition):
 desc=[d+'\n\n'+e for d,e in zip(DESCRIPTIONS[task],['Equivalent pure-loop implementations are also accepted.','也接受计算结果相同的纯循环实现。'])]
 return {'id':task,'titles':TITLES[task],'descriptions':desc,'manual_topics':TOPICS[task],'requirements':REQUIREMENTS[task], 'input_details':['Each dimension is positive and at most 16; each input has at most 1,024 integers. Values are in [-64,64], except gain/weight in [0,4], bias in [-16,16], diffusion x in [0,255], and mask in {0,1}. Inputs satisfy the stated shape constraints. Return the result from kernel.','各维度为 1 到 16，每个输入最多 1,024 个整数。一般数值范围为 [-64,64]；gain/weight 为 [0,4]，bias 为 [-16,16]，扩散题的 x 为 [0,255]，mask 为 0 或 1。输入满足题目中的形状约束。请从 kernel 返回结果。'],'starter':starter(task,condition),'ranks':RANKS[task],'output_rank':OUT_RANK[task],'arguments':names(task),'examples':[{'name':f'Case {i+1}','input':v,'expected':answer(task,v)} for i,v in enumerate(cases(task))]}
