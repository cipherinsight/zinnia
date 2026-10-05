"""Bilingual revision-3 contracts shared by both participant conditions."""
import numpy as np
from .tasks import TASKS, fixtures, signatures, oracle
ORDER=[t.name for t in TASKS]
BUDGETS=dict(zip(ORDER,[600,720,900,1080,1080,1200]))
TITLES=dict(zip(ORDER,[['Batch-specific calibration and energy','按批次校准并计算能量'],['Centered products between two sensor arrays','两组传感器的中心化交叉乘积'],['Weighted distances after projection','投影后的加权距离'],['Strided image filters with dilation','带步长和膨胀的图像滤波'],['Channel covariance in rectangular blocks','矩形区块内的通道协方差'],['Diffusion with batch-specific masks and mixing','按批次掩码和混合矩阵进行扩散']]))
DESCRIPTIONS = {'channel_energy_v3': ['Inputs: x[b,t,c] is channel c’s reading at time t in batch b. There are B '
                       'batches, T readings per batch, and C channels. gain and bias each have shape '
                       '[B,C]: gain[b,c] and bias[b,c] apply to every time point of that batch and '
                       'channel.\n'
                       '\n'
                       '1. Read B,T,C from x.shape. Insert a length-1 time axis into gain and bias: '
                       'their shapes change from [B,C] to [B,1,C]. Multiply x by the expanded gain and '
                       'add the expanded bias. Broadcasting repeats each batch’s parameters over its T '
                       'readings.\n'
                       '2. Clamp this [B,T,C] array to [0,255], replacing values below 0 or above 255 '
                       'with the respective limit. Call the result a.\n'
                       '3. Find a’s minimum along the time axis (axis 1), keeping that axis at length '
                       '1. The minimum array has shape [B,1,C]. Subtract it from a; broadcasting '
                       'subtracts the correct minimum from every time point.\n'
                       '4. Square each difference, then sum along axis 1 without keeping that axis. '
                       'Return the resulting [B,C] array. When T=1, it contains zeros.\n'
                       '\n'
                       'You can follow these steps with array operations. A pure loop implementation of '
                       'the same calculation is also accepted.\n'
                       '\n'
                       'Example for one batch and one channel: readings [1,3,5], gain 2, and bias -3 '
                       'give calibrated values [0,3,7]. Their minimum is 0, so the result is 0² + 3² + '
                       '7² = 58.',
                       '输入：x[b,t,c] 表示第 b 批数据在时刻 t、通道 c 的读数。共有 B 个批次，每批 T 次读数，每次包含 C 个通道。gain 和 bias '
                       '的形状都是 [B,C]；gain[b,c] 和 bias[b,c] 用于该批次、该通道的所有时刻。\n'
                       '\n'
                       '1. 从 x.shape 读取 B、T、C。给 gain 和 bias 插入一个长度为 1 的时间轴，将形状从 [B,C] 改为 [B,1,C]。用 x '
                       '乘以扩展后的 gain，再加上扩展后的 bias。广播会将每个批次的参数重复用于该批的 T 次读数。\n'
                       '2. 将得到的 [B,T,C] 数组限制到 [0,255]：小于 0 的改为 0，大于 255 的改为 255。将结果称为 a。\n'
                       '3. 沿 a 的时间轴（轴 1）求最小值，并保留这个长度为 1 的轴，得到 [B,1,C]。用 a '
                       '减去这个最小值数组；广播会为每个时刻减去对应批次、对应通道的最小值。\n'
                       '4. 将差值逐元素平方，再沿轴 1 求和，不保留该轴。返回得到的 [B,C] 数组。T=1 时结果全部为零。\n'
                       '\n'
                       '可以按以上步骤使用数组操作，也接受计算结果相同的纯循环实现。\n'
                       '\n'
                       '例如，一个批次的一个通道有读数 [1,3,5]，gain 为 2，bias 为 -3。校准后为 [0,3,7]，最小值是 0，因此结果为 0² + 3² + '
                       '7² = 58。'],
 'cross_products_v3': ['x has shape [B,T,C] and y has shape [B,T,D]: B batches, T time points, and C or '
                       'D channels. C and D may differ.\n'
                       '\n'
                       '1. Read T from x.shape[1]. Sum x over time (axis 1), keeping that axis at '
                       'length 1, to obtain sx with shape [B,1,C]. Calculate z = T*x - sx, using '
                       'broadcasting. z has shape [B,T,C].\n'
                       '2. Do the same for y using its own sum: sy has shape [B,1,D], and w = T*y - sy '
                       'has shape [B,T,D]. No division is needed.\n'
                       '3. For each batch b, transpose z[b] from [T,C] to [C,T]. Matrix-multiply it by '
                       'w[b], which has shape [T,D]. This produces [C,D]: each entry is the sum of '
                       'products at matching times.\n'
                       '4. Store the matrix for batch b in output[b]. Return shape [B,C,D]. Include all '
                       'C-by-D channel pairs; C and D may differ. Do not divide by T or T-1.\n'
                       '\n'
                       'You can follow these steps with array operations. A pure loop implementation of '
                       'the same calculation is also accepted.\n'
                       '\n'
                       'Example: in one batch, x has rows [1,2], [3,6] and y has rows [2], [5]. Then z '
                       'has rows [-2,-4], [2,4], and w has rows [-3], [3]. The output for this batch is '
                       '[[12],[24]].',
                       'x 的形状为 [B,T,C]，y 的形状为 [B,T,D]：共有 B 个批次、T 个时刻，两组数据分别有 C 和 D 个通道。C 和 D 可以不同。\n'
                       '\n'
                       '1. 从 x.shape[1] 读取 T。沿 x 的时间轴（轴 1）求和，并保留长度为 1 的时间轴，得到形状为 [B,1,C] 的 sx。利用广播计算 z '
                       '= T*x - sx，z 的形状为 [B,T,C]。\n'
                       '2. 对 y 做同样的操作，使用 y 自己的总和：sy 的形状为 [B,1,D]，w = T*y - sy 的形状为 [B,T,D]。这里不需要除法。\n'
                       '3. 对每个批次 b，将 z[b] 从 [T,C] 转置为 [C,T]，再与形状为 [T,D] 的 w[b] 做矩阵乘法，得到 '
                       '[C,D]。每个结果就是两个通道在相同时刻的乘积总和。\n'
                       '4. 将第 b 批的矩阵存入 output[b]，返回形状 [B,C,D]。包括全部 C×D 个通道对，C 与 D 可以不同。不要除以 T 或 T-1。\n'
                       '\n'
                       '可以按以上步骤使用数组操作，也接受计算结果相同的纯循环实现。\n'
                       '\n'
                       '例如，一个批次中，x 的两行为 [1,2]、[3,6]，y 的两行为 [2]、[5]。计算后 z 的两行为 [-2,-4]、[2,4]，w 的两行为 '
                       '[-3]、[3]，该批次的输出为 [[12],[24]]。'],
 'projected_distances_v3': ['Inputs: x has shape [B,N,D] and y has shape [B,M,D]. Each point has D '
                            'coordinates; the two sets contain N and M points per batch. transform has '
                            'shape [B,D,F], and weight has shape [B,F]. Each batch uses its own '
                            'transform and weights. D and F may differ, as may N and M.\n'
                            '\n'
                            '1. For each batch b, matrix-multiply x[b] ([N,D]) by transform[b] ([D,F]) '
                            'to obtain a ([N,F]). Multiply y[b] ([M,D]) by the same transform[b] to '
                            'obtain q ([M,F]). Complete these matrix products before squaring '
                            'anything.\n'
                            '2. Insert a length-1 axis into a to make [N,1,F], and into q to make '
                            '[1,M,F]. Subtract q from a using broadcasting. The result has shape '
                            '[N,M,F] and contains the coordinate differences for every pair of points.\n'
                            '3. Square these differences elementwise. Multiply by weight[b], shape [F]; '
                            'broadcasting applies the weight for coordinate f to every point pair.\n'
                            '4. Sum along the final axis F to obtain [N,M]. Store it in output[b]. '
                            'Repeat for all batches and return [B,N,M]. Do not take a square root. N '
                            'and M, and D and F, may differ.\n'
                            '\n'
                            'You can follow these steps with array operations. A pure loop '
                            'implementation of the same calculation is also accepted.\n'
                            '\n'
                            'Example: a transform [[1],[2]] maps [1,2] to [5] and [3,1] to [5]. Their '
                            'distance is zero. It maps [0,1] to [2]; with weight [3], the distance from '
                            '[5] to [2] is 3*(5-2)² = 27.',
                            '输入：x 的形状为 [B,N,D]，y 的形状为 [B,M,D]。每个点有 D 个坐标，两组点在每批中分别有 N 个和 M 个。transform '
                            '的形状为 [B,D,F]，weight 的形状为 [B,F]。每个批次使用自己的变换矩阵和权重。D 与 F 可以不同，N 与 M 也可以不同。\n'
                            '\n'
                            '1. 对每个批次 b，将 x[b]（[N,D]）与 transform[b]（[D,F]）做矩阵乘法，得到 a（[N,F]）。将 '
                            'y[b]（[M,D]）与同一个 transform[b] 相乘，得到 q（[M,F]）。先完成这两次矩阵乘法，再进行后面的平方操作。\n'
                            '2. 为 a 插入一个长度为 1 的轴，得到 [N,1,F]；为 q 插入一个长度为 1 的轴，得到 [1,M,F]。利用广播计算 a 减 '
                            'q，得到形状为 [N,M,F] 的数组，其中保存每对点在各坐标上的差。\n'
                            '3. 将这些差逐元素平方，再乘以形状为 [F] 的 weight[b]。广播会把坐标 f 的权重用于所有点对。\n'
                            '4. 沿最后的 F 轴求和，得到 [N,M]，存入 output[b]。处理全部批次后返回 [B,N,M]。不要开平方。N 与 M、D 与 F '
                            '均可以不同。\n'
                            '\n'
                            '可以按以上步骤使用数组操作，也接受计算结果相同的纯循环实现。\n'
                            '\n'
                            '例如，变换矩阵 [[1],[2]] 将 [1,2] 变为 [5]，将 [3,1] 也变为 [5]，两者距离为零。它将 [0,1] 变为 '
                            '[2]；若权重为 [3]，则 [5] 与 [2] 的加权平方距离为 3*(5-2)² = 27。'],
 'strided_filters_v3': ['x has shape [B,H,W,C], and filters has shape [K,U,V,C]. Each of K filters '
                        'combines U by V spatial positions and all C input channels. Use stride 2 and '
                        'dilation 2 along both spatial axes: adjacent output positions start two input '
                        'pixels apart, and adjacent filter coefficients sample input pixels two apart.\n'
                        '\n'
                        'No padding and no filter flipping. The filter’s effective height is 2*(U-1)+1 '
                        'and width is 2*(V-1)+1; both fit within the image. Let OH = (H-(2*(U-1)+1)) // '
                        '2 + 1 and OW = (W-(2*(V-1)+1)) // 2 + 1.\n'
                        '\n'
                        'Return [B,OH,OW,K], where result[b,i,j,k] = sum over u,v,c of '
                        'x[b,2*i+2*u,2*j+2*v,c]*filters[k,u,v,c]. Sum all input channels into each '
                        'output filter. A leftover row or column that cannot begin a complete window is '
                        'ignored.',
                        'x 的形状为 [B,H,W,C]，filters 的形状为 [K,U,V,C]。K 个滤波器各自组合 U×V 个空间位置和全部 C '
                        '个输入通道。两个空间轴的步长和膨胀率都为 2：相邻输出位置的起点相隔两个输入像素，滤波器中相邻系数采样的输入位置也相隔两个像素。\n'
                        '\n'
                        '不填充，不翻转滤波器。滤波器的实际覆盖高度为 2*(U-1)+1，宽度为 2*(V-1)+1，都不超过图像大小。令 OH = (H-(2*(U-1)+1)) '
                        '// 2 + 1，OW = (W-(2*(V-1)+1)) // 2 + 1。\n'
                        '\n'
                        '返回 [B,OH,OW,K]，其中 result[b,i,j,k] 是 x[b,2*i+2*u,2*j+2*v,c]*filters[k,u,v,c] 对 '
                        'u、v、c 的总和。每个输出滤波器都需要累加全部输入通道。末尾无法开始完整窗口的行或列不参与新的窗口。'],
 'rectangular_covariance_v3': ['x has shape [B,H,W,C]: B images, height H, width W, and C channels per '
                               'pixel. H is divisible by 2 and W by 3.\n'
                               '\n'
                               '1. Allocate output with shape [B,H/2,W/3,C,C]. For each image b and '
                               'block position (i,j), take rows 2*i through 2*i+1 and columns 3*j '
                               'through 3*j+2, including all channels. This gives a [2,3,C] block. '
                               'Blocks do not overlap.\n'
                               '2. Arrange that block as a [6,C] matrix p: each row is one pixel, and '
                               'each column is one channel. You can fill a new matrix with loops, or '
                               'copy the block into contiguous storage before reshaping it.\n'
                               '3. Sum p over its six rows (axis 0), giving one sum per channel. '
                               'Calculate z = 6*p - sums, broadcasting the [C] sums across the rows. z '
                               'has shape [6,C]. Calculate these sums independently for each block.\n'
                               '4. Transpose z to [C,6] and matrix-multiply by z ([6,C]). Store the '
                               'resulting [C,C] matrix in output[b,i,j]. Include diagonal entries and '
                               'both orders of each channel pair. Do not divide the result. Return '
                               'output after processing all blocks.\n'
                               '\n'
                               'You can follow these steps with array operations. A pure loop '
                               'implementation of the same calculation is also accepted.\n'
                               '\n'
                               'Example: if a block has one channel with values [1,2,3,4,5,6], its sum '
                               'is 21. The adjusted values are [-15,-9,-3,3,9,15]. Its single '
                               'channel-pair total is 630.',
                               'x 的形状为 [B,H,W,C]：B 张图像，高 H、宽 W，每个像素有 C 个通道。H 能被 2 整除，W 能被 3 整除。\n'
                               '\n'
                               '1. 创建形状为 [B,H/2,W/3,C,C] 的输出。对每张图像 b 的每个区块位置 (i,j)，取第 2*i 到 2*i+1 行、第 '
                               '3*j 到 3*j+2 列，包含所有通道，得到 [2,3,C] 区块。上述范围含两端，区块互不重叠。\n'
                               '2. 将该区块排列成 [6,C] 矩阵 '
                               'p：每行是一个像素，每列是一个通道。可以用循环填入新矩阵，也可以先把区块复制到连续存储空间，再改变形状。\n'
                               '3. 沿 p 的六像素轴（轴 0）求和，得到每个通道的总和。计算 z = 6*p - sums，将 [C] 的 sums 广播到六行。z '
                               '的形状为 [6,C]。每个区块都要独立计算自己的 sums。\n'
                               '4. 将 z 转置为 [C,6]，再与 z（[6,C]）做矩阵乘法，得到 [C,C]。将它存入 '
                               'output[b,i,j]。包括对角线和每对通道的两个顺序，不做除法。处理全部区块后返回 output。\n'
                               '\n'
                               '可以按以上步骤使用数组操作，也接受计算结果相同的纯循环实现。\n'
                               '\n'
                               '例如，某区块只有一个通道，六个值为 [1,2,3,4,5,6]，总和是 21。调整后为 '
                               '[-15,-9,-3,3,9,15]，唯一的通道对结果为 630。'],
 'batch_diffusion_v3': ['Inputs: x has shape [B,H,W,C] and contains integer values from 0 to 255. mask '
                        'has shape [B,H,W]: 1 means update this pixel, and 0 means keep it unchanged. '
                        'mix has shape [B,C,C]; mix[b,c,d] is the weight from source channel c to '
                        'destination channel d. Masks and mixing matrices are separate for each image '
                        'b.\n'
                        '\n'
                        '1. Copy x into current. Perform steps 2–4 exactly three times.\n'
                        '2. At the start of each round, make an independent copy of current called '
                        'next. For each image b, visit only interior pixels: rows 1 through H-2 and '
                        'columns 1 through W-2, inclusive. Skip pixels with mask[b,i,j]=0. If H<3 or '
                        'W<3, there are no pixels to update.\n'
                        '3. For an updated pixel, take the length-C channel vectors at the center, '
                        'above, below, left, and right, all from current. Calculate average = '
                        'floor((4*center + above + below + left + right)/8), elementwise. Finish '
                        'rounding down before mixing channels.\n'
                        '4. Treat average as a row vector [C] and matrix-multiply it by mix[b] ([C,C]). '
                        'Thus destination d receives the sum of average[c]*mix[b,c,d] over c. Clamp the '
                        'resulting vector to [0,255] and store it in next[b,i,j,:]. After every pixel '
                        'has been processed, replace current with next.\n'
                        '5. After three rounds, return current with shape [B,H,W,C]. Never read '
                        'neighbors from next. Border pixels and mask=0 pixels remain unchanged, but '
                        'still supply their current values when used as neighbors.\n'
                        '\n'
                        'You can follow these steps with array operations. A pure loop implementation '
                        'of the same calculation is also accepted.\n'
                        '\n'
                        'Example of step 2: a center value of 10 and neighbor values 1,2,3,4 give '
                        '(40+1+2+3+4)//8 = 6. If two channels give [6,2] and the mixing matrix is '
                        '[[1,2],[3,1]], their new values are [12,14].',
                        '输入：x 的形状为 [B,H,W,C]，包含 0 到 255 的整数。mask 的形状为 [B,H,W]：1 表示更新该像素，0 表示保持不变。mix '
                        '的形状为 [B,C,C]；mix[b,c,d] 是源通道 c 到目标通道 d 的权重。每张图像 b 都有自己的 mask 和 mix。\n'
                        '\n'
                        '1. 将 x 复制为 current。将下面的第 2–4 步恰好执行三轮。\n'
                        '2. 每轮开始时，将 current 独立复制为 next。对每张图像 b，只遍历内部像素：第 1 到 H-2 行、第 1 到 W-2 列，范围含两端。跳过 '
                        'mask[b,i,j]=0 的像素。H<3 或 W<3 时没有需要更新的像素。\n'
                        '3. 对需要更新的像素，从 current 取出自身、上、下、左、右五个长度为 C 的通道向量。逐元素计算 average = (4*自身 + 上 + 下 '
                        '+ 左 + 右) 除以 8 并向下取整。先完成取整，再混合通道。\n'
                        '4. 将 average 作为长度为 C 的行向量，与 mix[b]（[C,C]）做矩阵乘法，即目标通道 d 的结果为所有 '
                        'average[c]*mix[b,c,d] 的总和。将结果逐元素限制到 [0,255]，存入 next[b,i,j,:]。全部像素处理完后，再用 next '
                        '替换 current。\n'
                        '5. 三轮结束后，返回形状为 [B,H,W,C] 的 current。不要从 next 读取邻居值。边界像素和 mask=0 '
                        '的像素保持不变，但作为邻居时仍提供 current 中的值。\n'
                        '\n'
                        '可以按以上步骤使用数组操作，也接受计算结果相同的纯循环实现。\n'
                        '\n'
                        '第 2 步举例：自身值为 10，四个邻居为 1、2、3、4，则结果为 (40+1+2+3+4)//8 = 6。若两个通道的结果为 [6,2]，混合矩阵为 '
                        '[[1,2],[3,1]]，则它们的新值为 [12,14]。']}
HINT_TEXT=[
 ['Insert a time axis in the [B,C] parameters before broadcasting. Then reduce the time axis for the minimum and energy.','广播 [B,C] 参数前插入时间轴，再沿时间轴求最小值和能量。'],
 ['Center each input over time first. Then contract their common time axis, preserving batch, left-channel, and right-channel axes.','先分别沿时间轴中心化，再收缩共同的时间轴，保留批次、左侧通道和右侧通道轴。'],
 ['Build projected arrays [B,N,F] and [B,M,F], then combine point axes and reduce over F. Alternatively project each point difference before squaring.','可先生成 [B,N,F] 和 [B,M,F] 的投影数组，再组合点的轴并沿 F 求和；也可先投影每对点的坐标差，再平方。'],
 ['Compute the output dimensions first. Extract each window with step-2 slices, or accumulate contributions from each filter offset. Keep output stride and filter dilation separate.','先计算输出大小。可用步长为 2 的切片提取窗口，也可按滤波器偏移累加。分别处理输出步长和滤波器膨胀率。'],
 ['Group pixels into [B,H/2,W/3,6,C] using slices or reshape and transpose. Center over the six-pixel axis before contracting channel products.','可通过切片，或 reshape 与 transpose，将像素组成 [B,H/2,W/3,6,C]。沿六像素轴中心化，再收缩通道乘积。'],
 ['Use separate current and next buffers. Preserve the batch axis in the mask and matrix product; do spatial averaging before channel mixing.','使用独立的当前和下一轮缓冲区。掩码和矩阵乘法都保留批次轴；先计算空间平均值，再混合通道。']]
TOPICS=[['expand','broadcast','clamp','minmax','sum'],['sum','transpose','matmul','batches'],['matmul','expand','broadcast','sum'],['slicing','sum','matmul'],['reshape','transpose','matmul'],['updates','clamp','batches']]
HINTS={t:([h[0]+' A straightforward implementation using only loops (nested where needed) is also accepted.',h[1]+' 也接受最直接的纯循环实现，需要时使用嵌套循环。'],topics) for t,h,topics in zip(ORDER,HINT_TEXT,TOPICS)}
REQUIREMENTS = {'channel_energy_v3': [['1. Read B,T,C from x.shape. Insert a length-1 time axis into gain and bias: '
                        'their shapes change from [B,C] to [B,1,C]. Multiply x by the expanded gain and '
                        'add the expanded bias. Broadcasting repeats each batch’s parameters over its T '
                        'readings.'],
                       ['1. 从 x.shape 读取 B、T、C。给 gain 和 bias 插入一个长度为 1 的时间轴，将形状从 [B,C] 改为 [B,1,C]。用 x '
                        '乘以扩展后的 gain，再加上扩展后的 bias。广播会将每个批次的参数重复用于该批的 T 次读数。']],
 'cross_products_v3': [['1. Read T from x.shape[1]. Sum x over time (axis 1), keeping that axis at '
                        'length 1, to obtain sx with shape [B,1,C]. Calculate z = T*x - sx, using '
                        'broadcasting. z has shape [B,T,C].'],
                       ['1. 从 x.shape[1] 读取 T。沿 x 的时间轴（轴 1）求和，并保留长度为 1 的时间轴，得到形状为 [B,1,C] 的 sx。利用广播计算 z '
                        '= T*x - sx，z 的形状为 [B,T,C]。']],
 'projected_distances_v3': [['1. For each batch b, matrix-multiply x[b] ([N,D]) by transform[b] ([D,F]) '
                             'to obtain a ([N,F]). Multiply y[b] ([M,D]) by the same transform[b] to '
                             'obtain q ([M,F]). Complete these matrix products before squaring '
                             'anything.'],
                            ['1. 对每个批次 b，将 x[b]（[N,D]）与 transform[b]（[D,F]）做矩阵乘法，得到 a（[N,F]）。将 '
                             'y[b]（[M,D]）与同一个 transform[b] 相乘，得到 q（[M,F]）。先完成这两次矩阵乘法，再进行后面的平方操作。']],
 'rectangular_covariance_v3': [['1. Allocate output with shape [B,H/2,W/3,C,C]. For each image b and '
                                'block position (i,j), take rows 2*i through 2*i+1 and columns 3*j '
                                'through 3*j+2, including all channels. This gives a [2,3,C] block. '
                                'Blocks do not overlap.'],
                               ['1. 创建形状为 [B,H/2,W/3,C,C] 的输出。对每张图像 b 的每个区块位置 (i,j)，取第 2*i 到 2*i+1 行、第 '
                                '3*j 到 3*j+2 列，包含所有通道，得到 [2,3,C] 区块。上述范围含两端，区块互不重叠。']],
 'batch_diffusion_v3': [['1. Copy x into current. Perform steps 2–4 exactly three times.'],
                        ['1. 将 x 复制为 current。将下面的第 2–4 步恰好执行三轮。']],
 'strided_filters_v3': [['stride 2 and dilation 2', 'No padding and no filter flipping.', '[B,OH,OW,K]'],
                        ['步长和膨胀率都为 2', '不填充，不翻转滤波器。', '[B,OH,OW,K]']]}
def contract(task):return next(t for t in TASKS if t.name==task)
def names(task):return list(contract(task).arguments)
RANKS={t.name:[len(s) for s in signatures(t,t.profiles[0])[0].values()] for t in TASKS}
OUT_RANK={t.name:t.output_rank for t in TASKS}
def starter(task,condition):
 if condition=='zinnia':return 'def kernel('+', '.join(names(task))+'):\n    # Write your implementation here.\n    raise NotImplementedError("Implement kernel")\n'
 args=', '.join(f'{n}: &Array{r}<i64>' for n,r in zip(names(task),RANKS[task]))
 return f'fn kernel({args}) -> Array{OUT_RANK[task]}<i64> {{\n    // Write your implementation here.\n    todo!("Implement kernel")\n}}\n'
def cases(task,hidden=False):
 t=contract(task)
 selections=[(0,i) for i in ([1,3,4,5,7] if hidden else [0,2,6])]
 if hidden:selections += [(p,i) for p in range(1,len(t.profiles)) for i in (2,6)]
 return [dict(zip(t.arguments,[v.tolist() for v in fixtures(t,t.profiles[p])[i]])) for p,i in selections]
def bounds(task):
 out={n:(-16,16) for n in names(task)}
 for n,r in [('gain',(0,4)),('weight',(0,4)),('mask',(0,1)),('mix',(-2,2)),('transform',(-4,4))]:
  if n in out:out[n]=r
 if task=='channel_energy_v3':out['x']=(-64,64)
 if task=='batch_diffusion_v3':out['x']=(0,255)
 return out
def profile(task,a):
 x=a[0]
 if task=='channel_energy_v3':return dict(zip(('B','T','C'),x.shape))
 if task=='cross_products_v3':return dict(zip(('B','T','C'),x.shape),D=a[1].shape[2])
 if task=='projected_distances_v3':return dict(zip(('B','N','D'),x.shape),M=a[1].shape[1],F=a[2].shape[2])
 p=dict(zip(('B','H','W','C'),x.shape))
 if task=='strided_filters_v3':p.update(zip(('K','U','V'),a[1].shape[:3]))
 return p
def validate_input(task,data):
 t=contract(task)
 if not isinstance(data,dict) or set(data)!=set(t.arguments):raise ValueError('Input must contain exactly: '+', '.join(t.arguments))
 arrays=[]
 for n,r in zip(t.arguments,RANKS[task]):
  a=np.asarray(data[n]);lo,hi=bounds(task)[n]
  if a.dtype.kind not in 'iu' or a.ndim!=r or not all(a.shape) or a.size>1024 or max(a.shape)>16:raise ValueError(f'{n}: expected a nonempty rank-{r} integer array, dimensions <=16, at most 1024 values')
  if np.any(a<lo) or np.any(a>hi):raise ValueError(f'{n} values must be in [{lo},{hi}]')
  arrays.append(a.astype(np.int64))
 p=profile(task,arrays)
 if task=='strided_filters_v3' and (2*(p['U']-1)+1>p['H'] or 2*(p['V']-1)+1>p['W']):raise ValueError('The dilated filter must fit inside the image')
 if task=='rectangular_covariance_v3' and (p['H']%2 or p['W']%3):raise ValueError('H must be divisible by 2 and W by 3')
 shapes,out=signatures(t,p)
 if any(a.shape!=shape for a,shape in zip(arrays,shapes.values())):raise ValueError('Related batch, observation, channel, projection or parameter dimensions do not match')
 if np.prod(out)>4096:raise ValueError('Output must contain at most 4096 values')
 return arrays
def answer(task,data):
 a=validate_input(task,data);return oracle(contract(task),profile(task,a),a).tolist()
def task_payload(task,condition):
 ranges=', '.join(f'{n}: [{v[0]},{v[1]}]' for n,v in bounds(task).items())
 details=['Integer input ranges: '+ranges+'. Each dimension is positive and at most 16; each input has at most 1,024 values. All supplied inputs satisfy the stated shape constraints. Return the result from kernel.',
 '整数输入范围：'+ranges+'。各维度为 1 到 16，每个输入最多 1,024 个元素。输入满足题目规定的形状关系。请从 kernel 返回结果。']
 return {'id':task,'titles':TITLES[task],'descriptions':DESCRIPTIONS[task],'hints':HINTS[task][0],'manual_topics':HINTS[task][1],'requirements':REQUIREMENTS[task],'input_details':details,'starter':starter(task,condition),'examples':[{'name':f'Case {i+1}','input':v,'expected':answer(task,v)} for i,v in enumerate(cases(task))],'ranks':RANKS[task],'output_rank':OUT_RANK[task],'arguments':names(task)}
