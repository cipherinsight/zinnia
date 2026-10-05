# Revision 2 participant task contracts

Both conditions receive these same contracts.


## 1. Calibrate batches and measure channel energy


x has shape [B, T, C]: B batches, T readings per batch, and C channels. gain and bias each contain C values. First compute a[b,t,c] = clip(x[b,t,c] * gain[c] + bias[c], 0, 255).

For each batch and channel separately, find the minimum of these calibrated readings over T. Subtract that minimum from each calibrated reading, square the differences, and sum over T. Return one energy per batch and channel, with shape [B, C].

Clip before finding the minimum. Do not combine different batches or channels. A batch with one reading has zero energy in every channel.


Implementation hint: Calibrate using channel broadcasting, then reduce the time axis for the minimum and the squared sum. Keep the time axis when subtracting the minimum. A straightforward implementation using only loops (nested where needed) is also accepted.


批量校准并计算通道能量


x 的形状为 [B, T, C]：共有 B 个批次，每批有 T 次读数，每次包含 C 个通道。gain 和 bias 各有 C 个值。先计算 a[b,t,c] = clip(x[b,t,c] * gain[c] + bias[c], 0, 255)，即把结果限制在 0 到 255。

对每个批次、每个通道，分别找出这 T 个校准后读数的最小值。每个校准后读数减去该最小值，将差平方，再沿 T 求和。返回形状为 [B, C] 的数组，每个位置是对应批次、对应通道的能量。

先限制数值范围，再求最小值。不同批次、不同通道不要混合。只有一次读数的批次，各通道能量都为零。


实现提示：先沿通道广播校准，再沿时间轴求最小值和平方和。相减时保留时间轴以便广播。 也可以采用最直接的纯循环实现，需要时使用嵌套循环。


## 2. Centered products between channels


x has shape [B, T, C]: B batches of T observations, each with C channels. For each batch and channel compute its sum over T, then define z[b,t,c] = T * x[b,t,c] - sum_t(x[b,t,c]). This is integer-scaled centering; no division is needed.

Return a [B, C, C] array where result[b,c,d] = sum_t(z[b,t,c] * z[b,t,d]). Compute every channel pair, including the diagonal, within the same batch. Do not divide the result by T or T-1. With T=1 the entire output is zero.

For one batch with rows [1,2] and [3,6], the centered rows are [-2,-4] and [2,4], giving [[8,16],[16,32]].


Implementation hint: Compute channel sums along the observation axis first. Then use an outer product reduction, a matrix product for each batch, or direct accumulation. A straightforward implementation using only loops (nested where needed) is also accepted.


计算通道间的中心化乘积


x 的形状为 [B, T, C]：B 个批次，每批 T 次观测，每次有 C 个通道。先对每个批次、每个通道沿 T 求和，再计算 z[b,t,c] = T * x[b,t,c] - sum_t(x[b,t,c])。这是用整数缩放的中心化，不需要除法。

返回形状为 [B, C, C] 的数组，其中 result[b,c,d] = sum_t(z[b,t,c] * z[b,t,d])。在同一批次内计算所有通道对，包括对角线。结果不要除以 T 或 T-1。T=1 时，输出全部为零。

例如，一个批次的两行为 [1,2] 和 [3,6]，中心化后为 [-2,-4] 和 [2,4]，结果是 [[8,16],[16,32]]。


实现提示：先沿观测轴求通道总和，再对外积归约、逐批矩阵相乘，或直接累加。 也可以采用最直接的纯循环实现，需要时使用嵌套循环。


## 3. Weighted distances within batches


x has shape [B, N, D] and y has shape [B, M, D]. Each batch contains N and M points respectively; each point has D coordinates. weight contains D nonnegative coordinate weights. D is not fixed to 3.

Return shape [B, N, M], with result[b,i,j] = sum_d(weight[d] * (x[b,i,d] - y[b,j,d])²). Compare points only within the same batch, and sum all weighted squared coordinate differences into one value per point pair. Do not take a square root. N and M may differ; zero weights and repeated points are allowed.

For points [1,2] and [3,5] with weights [2,0], the result is 2*(1-3)² + 0*(2-5)² = 8.


Implementation hint: Insert a point axis into each input to form point pairs, then reduce over coordinates. Processing one pair at a time avoids a large intermediate tensor. A straightforward implementation using only loops (nested where needed) is also accepted.


计算批次内的加权距离


x 的形状为 [B, N, D]，y 的形状为 [B, M, D]。每个批次分别包含 N 个和 M 个点，每个点有 D 个坐标。weight 包含 D 个非负坐标权重。D 不固定为 3。

返回形状 [B, N, M]，其中 result[b,i,j] = sum_d(weight[d] * (x[b,i,d] - y[b,j,d])²)。只比较同一批次内的点，把所有坐标的加权平方差相加，每对点得到一个值。不要开平方。N 和 M 可以不同；权重可以为零，点可以重复。

例如，[1,2] 与 [3,5] 的权重为 [2,0] 时，结果为 2*(1-3)² + 0*(2-5)² = 8。


实现提示：分别插入点的轴来组合点对，再沿坐标轴归约。也可以逐对处理，避免较大的中间张量。 也可以采用最直接的纯循环实现，需要时使用嵌套循环。


## 4. Apply multichannel image filters


x has shape [B, H, W, C]: B images with H rows, W columns, and C channels. filters has shape [K, U, V, C]: K filters, each covering U rows, V columns, and all C channels. U<=H and V<=W.

Slide each filter one pixel at a time over each image. At each position multiply matching filter and image values and sum over both spatial axes and all channels. Use only positions where the entire filter fits; no padding. Use filter coefficients in their given order; do not flip the filter.

Return [B, H-U+1, W-V+1, K], where result[b,i,j,k] = sum over u,v,c of x[b,i+u,j+v,c] * filters[k,u,v,c]. Keep images and output filters separate. This operation is also called valid cross-correlation.


Implementation hint: Extract each valid image patch and reduce its product with the filters. Alternatively, accumulate one filter offset at a time over the output grid. A straightforward implementation using only loops (nested where needed) is also accepted.


应用多通道图像滤波器


x 的形状为 [B, H, W, C]，表示 B 张 H 行、W 列、C 通道的图像。filters 的形状为 [K, U, V, C]：K 个滤波器，每个覆盖 U 行、V 列和全部 C 个通道。U<=H，V<=W。

在每张图像上，以一个像素为步长滑动每个滤波器。每个位置上，将滤波器与图像对应的值相乘，再对两个空间轴和全部通道求和。只计算滤波器完全落在图像内的位置，不填充边界。滤波器系数按给定顺序使用，不要翻转。

返回形状 [B, H-U+1, W-V+1, K]，其中 result[b,i,j,k] 是 x[b,i+u,j+v,c] * filters[k,u,v,c] 对 u、v、c 的总和。不同图像、不同输出滤波器分别计算。这也称为有效互相关。


实现提示：可提取每个有效图像区块，与滤波器相乘后归约；也可每次固定一个滤波器偏移，在输出网格上累加。 也可以采用最直接的纯循环实现，需要时使用嵌套循环。


## 5. Channel covariance within image blocks


x has shape [B, H, W, C]. H and W are positive even numbers. Split each image into non-overlapping 2 by 2 blocks starting at the top-left. Treat the four pixels in each block as four observations of C channels.

Within each block, let s[c] be the sum of its four values for channel c, and z[p,c] = 4*x[p,c] - s[c], where p identifies a pixel in that block. For every channel pair (c,d), sum z[p,c]*z[p,d] over the four pixels.

Return shape [B, H/2, W/2, C, C], preserving image and block order. Include diagonal and off-diagonal entries. Do not mix blocks or divide the result; these are scaled covariance values. A constant channel within a block has a zero row and column in that block's result.


Implementation hint: Group the four pixels of each block using slices or reshape and transpose. Center within each block, then reduce channel products over its four pixels. A straightforward implementation using only loops (nested where needed) is also accepted.


计算图像区块内的通道协方差


x 的形状为 [B, H, W, C]，H 和 W 都是正偶数。从左上角开始，把每张图像划分为互不重叠的 2×2 区块。把每个区块的四个像素看作 C 个通道的四次观测。

在每个区块内，s[c] 是通道 c 的四个值之和，z[p,c] = 4*x[p,c] - s[c]，p 表示区块中的一个像素。对每一对通道 (c,d)，将四个像素的 z[p,c]*z[p,d] 相加。

返回形状 [B, H/2, W/2, C, C]，保持图像和区块的顺序。包括对角线和非对角线元素。不同区块不要混合，结果不做除法；这是缩放后的协方差。如果某个通道在区块内为常数，其对应的行和列都应为零。


实现提示：可通过切片，或 reshape 与 transpose，把每个区块的四个像素分组。在区块内中心化，再对四个像素的通道乘积求和。 也可以采用最直接的纯循环实现，需要时使用嵌套循环。


## 6. Update masked grids and mix channels


x has shape [B, H, W, C] and values from 0 to 255. mask is an [H, W] array of zeros and ones, shared by all batches. mix is a [C, C] matrix: mix[c,d] weights source channel c into destination channel d. Perform exactly three rounds.

Keep the outermost rows and columns unchanged. Also keep every cell with mask[i,j]=0 unchanged. For each remaining cell, first compute one spatial average per source channel: (4*center + above + below + left + right) // 8. Round down before mixing channels.

For destination channel d, sum average[c]*mix[c,d] over c and clip that sum to [0,255]. All spatial reads in a round must use the values at the start of that round, including frozen neighbors. Apply the new values together after the round. Return the full [B,H,W,C] array after three rounds. If H<3 or W<3, return the input values unchanged.


Implementation hint: Keep separate current and next arrays for each round. Compute spatial averages before channel mixing, and copy frozen cells into the next array. A straightforward implementation using only loops (nested where needed) is also accepted.


按掩码更新网格并混合通道


x 的形状为 [B, H, W, C]，值在 0 到 255 之间。mask 是所有批次共享的 [H, W] 数组，只含 0 和 1。mix 为 [C, C] 矩阵，mix[c,d] 是从源通道 c 到目标通道 d 的权重。恰好执行三轮。

最外侧的行和列始终保持不变。mask[i,j]=0 的单元也保持不变。对其余单元，先分别计算每个源通道的空间平均值：(4*自身 + 上 + 下 + 左 + 右) // 8。先向下取整，再混合通道。

目标通道 d 的值是所有 average[c]*mix[c,d] 的和，再限制到 [0,255]。同一轮的所有空间读取都必须使用本轮开始时的值，包括被冻结的邻居。整轮结束后一起应用新值。返回三轮后的完整 [B,H,W,C] 数组。如果 H<3 或 W<3，输入值全部保持不变。


实现提示：每轮使用独立的当前数组和下一轮数组。先计算空间平均值，再混合通道；被冻结的单元复制到下一轮数组。 也可以采用最直接的纯循环实现，需要时使用嵌套循环。

