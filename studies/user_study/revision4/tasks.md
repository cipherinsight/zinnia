# Current participant task contracts (revision 4)

The same specifications are used for both language conditions.


## 1. Batch-specific calibration and energy


x has shape [B,T,C]: B batches, T readings per batch, C channels. gain and bias each have shape [B,C]. Each batch has its own calibration parameters, shared across its T readings. Compute a[b,t,c] = clip(x[b,t,c]*gain[b,c]+bias[b,c],0,255).

For each batch and channel separately, find the minimum of a over T. Subtract that minimum from each calibrated reading, square the differences, and sum over T. Return shape [B,C]. Clip before finding the minimum. Do not reuse one batch’s parameters for another batch. T=1 gives zero energy.


Implementation hint: Insert a time axis in the [B,C] parameters before broadcasting. Then reduce the time axis for the minimum and energy. A straightforward implementation using only loops (nested where needed) is also accepted.


按批次校准并计算能量


x 的形状为 [B,T,C]：B 个批次，每批 T 次读数，每次有 C 个通道。gain 和 bias 的形状均为 [B,C]。每个批次有自己的校准参数，在该批次的 T 次读数之间共享。先计算 a[b,t,c] = clip(x[b,t,c]*gain[b,c]+bias[b,c],0,255)，即把结果限制到 0 到 255。

对每个批次、每个通道，沿 T 求 a 的最小值。各校准后读数减去该最小值，将差平方，再沿 T 求和。返回形状 [B,C]。先限制范围，再求最小值。不要把一个批次的参数用于其他批次。T=1 时能量为零。


实现提示：广播 [B,C] 参数前插入时间轴，再沿时间轴求最小值和能量。 也接受最直接的纯循环实现，需要时使用嵌套循环。


## 2. Centered products between two sensor arrays


x has shape [B,T,C] and y has shape [B,T,D]. They record two sensor arrays at the same T times in each batch. Their channel counts C and D may differ. Center the two arrays separately using integer scaling: z[b,t,c] = T*x[b,t,c] - sum_t(x[b,t,c]); w[b,t,d] = T*y[b,t,d] - sum_t(y[b,t,d]).

Return shape [B,C,D], with result[b,c,d] = sum_t(z[b,t,c]*w[b,t,d]). Pair observations at the same time and in the same batch. Do not assume a square or symmetric result. Do not divide by T or T-1. T=1 gives all zeros.

For x rows [1,2], [3,6] and y rows [2], [5] in one batch, the result is [[12],[24]].


Implementation hint: Center each input over time first. Then contract their common time axis, preserving batch, left-channel, and right-channel axes. A straightforward implementation using only loops (nested where needed) is also accepted.


两组传感器的中心化交叉乘积


x 的形状为 [B,T,C]，y 的形状为 [B,T,D]。它们记录同一批次中、同样 T 个时刻的两组传感器数据。通道数 C 和 D 可以不同。分别进行整数缩放的中心化：z[b,t,c] = T*x[b,t,c] - sum_t(x[b,t,c])；w[b,t,d] = T*y[b,t,d] - sum_t(y[b,t,d])。

返回形状 [B,C,D]，其中 result[b,c,d] = sum_t(z[b,t,c]*w[b,t,d])。只配对同一批次、同一时刻的观测。不要假定结果是方阵或对称矩阵。不要除以 T 或 T-1。T=1 时结果全部为零。

例如，一个批次中 x 的两行为 [1,2]、[3,6]，y 的两行为 [2]、[5]，结果为 [[12],[24]]。


实现提示：先分别沿时间轴中心化，再收缩共同的时间轴，保留批次、左侧通道和右侧通道轴。 也接受最直接的纯循环实现，需要时使用嵌套循环。


## 3. Weighted distances after projection


x has shape [B,N,D] and y has shape [B,M,D]: two sets of D-coordinate points in each batch. transform has shape [B,D,F] and weight has shape [B,F]. Every batch has its own projection and nonnegative weights. D and F may differ.

First project each point: a[b,i,f] = sum_d(x[b,i,d]*transform[b,d,f]) and q[b,j,f] = sum_d(y[b,j,d]*transform[b,d,f]). Both point sets use the same projection within their batch.

Return shape [B,N,M], with result[b,i,j] = sum_f(weight[b,f]*(a[b,i,f]-q[b,j,f])²). Sum the projection contributions before squaring the projected differences. Do not mix batches or take a square root. N and M may differ; zero weights and repeated points are allowed.


Implementation hint: Build projected arrays [B,N,F] and [B,M,F], then combine point axes and reduce over F. Alternatively project each point difference before squaring. A straightforward implementation using only loops (nested where needed) is also accepted.


投影后的加权距离


x 的形状为 [B,N,D]，y 的形状为 [B,M,D]，表示每个批次中的两组点，每个点有 D 个坐标。transform 的形状为 [B,D,F]，weight 的形状为 [B,F]。每个批次有自己的投影矩阵和非负权重。D 和 F 可以不同。

先投影每个点：a[b,i,f] = sum_d(x[b,i,d]*transform[b,d,f])，q[b,j,f] = sum_d(y[b,j,d]*transform[b,d,f])。同一批次的两组点使用相同的投影矩阵。

返回形状 [B,N,M]，其中 result[b,i,j] = sum_f(weight[b,f]*(a[b,i,f]-q[b,j,f])²)。先累加投影贡献，再对投影后的坐标差求平方。不要混合批次，也不要开平方。N 和 M 可以不同；权重可以为零，点可以重复。


实现提示：可先生成 [B,N,F] 和 [B,M,F] 的投影数组，再组合点的轴并沿 F 求和；也可先投影每对点的坐标差，再平方。 也接受最直接的纯循环实现，需要时使用嵌套循环。


## 4. Image filters with stride 2 and dilation 3


x has shape [B,H,W,C], and filters has shape [K,U,V,C]. Each of K filters combines U by V spatial positions and all C input channels. Use stride 2 and dilation 3 along both spatial axes: adjacent output positions start two input pixels apart, while adjacent filter coefficients sample input pixels three apart.

No padding and no filter flipping. The effective filter height is EH = 3*(U-1)+1 and width is EW = 3*(V-1)+1; both fit within the image. Let OH = (H-EH) // 2 + 1 and OW = (W-EW) // 2 + 1.

Return [B,OH,OW,K], where result[b,i,j,k] = sum over u,v,c of x[b,2*i+3*u,2*j+3*v,c]*filters[k,u,v,c]. Sum all input channels into each output filter. Ignore trailing positions that cannot start a complete window.

For one image with one row and one channel, values [1,2,3,4,5,6] and filter [2,3] produce [14,24]: the first window samples 1 and 4; the second samples 3 and 6.


Implementation hint: Calculate the effective window extent and output dimensions first. You can extract each window using step-3 slices, accumulate filter offsets while stepping output starts by 2, or use a sliding-window view. Keep the output step and the spacing within a filter separate. A straightforward implementation using only loops (nested where needed) is also accepted.


步长 2、膨胀率 3 的图像滤波


x 的形状为 [B,H,W,C]，filters 的形状为 [K,U,V,C]。K 个滤波器各自组合 U×V 个空间位置和全部 C 个输入通道。两个空间轴的步长为 2、膨胀率为 3：相邻输出位置的起点相隔两个输入像素，滤波器中相邻系数采样的输入位置相隔三个像素。

不填充，不翻转滤波器。实际覆盖高度为 EH = 3*(U-1)+1，宽度为 EW = 3*(V-1)+1，都不超过图像大小。令 OH = (H-EH) // 2 + 1，OW = (W-EW) // 2 + 1。

返回 [B,OH,OW,K]，其中 result[b,i,j,k] 是 x[b,2*i+3*u,2*j+3*v,c]*filters[k,u,v,c] 对 u、v、c 的总和。每个输出滤波器都要累加全部输入通道。末尾无法开始完整窗口的位置忽略。

例如，单张、单行、单通道图像的值为 [1,2,3,4,5,6]，滤波器为 [2,3]，输出为 [14,24]：第一个窗口采样 1 和 4，第二个采样 3 和 6。


实现提示：先计算实际窗口范围和输出大小。可以用步长为 3 的切片提取窗口，按滤波器偏移累加并将输出起点每次移动 2，或使用滑动窗口视图。分别维护输出步长与滤波器内部的采样间距。也接受最直接的纯循环实现，需要时使用嵌套循环。


## 5. Channel covariance in rectangular blocks


x has shape [B,H,W,C]. H is divisible by 2 and W by 3. Starting at the top-left, split each image into non-overlapping blocks of 2 rows and 3 columns. Treat the six pixels of each block as six observations of C channels.

Within each block, s[c] is the sum of its six values in channel c; z[p,c] = 6*x[p,c]-s[c], where p is a pixel within that block. For each channel pair (c,d), sum z[p,c]*z[p,d] over those six pixels.

Return shape [B,H/2,W/3,C,C], preserving image and block order. Include every channel pair. Center independently within each block, not across an image or batch. Do not divide the result; it represents scaled covariance.


Implementation hint: Group pixels into [B,H/2,W/3,6,C] using slices or reshape and transpose. Center over the six-pixel axis before contracting channel products. A straightforward implementation using only loops (nested where needed) is also accepted.


矩形区块内的通道协方差


x 的形状为 [B,H,W,C]。H 能被 2 整除，W 能被 3 整除。从左上角开始，把每张图像划分为互不重叠的 2 行、3 列区块。把每个区块的六个像素视为 C 个通道的六次观测。

在每个区块内，s[c] 是通道 c 的六个值之和；z[p,c] = 6*x[p,c]-s[c]，p 是区块内的像素。对每对通道 (c,d)，将这六个像素的 z[p,c]*z[p,d] 相加。

返回形状 [B,H/2,W/3,C,C]，保持图像和区块的顺序，包括所有通道对。每个区块独立中心化，不要沿整张图像或批次中心化。结果不做除法，它表示缩放后的协方差。


实现提示：可通过切片，或 reshape 与 transpose，将像素组成 [B,H/2,W/3,6,C]。沿六像素轴中心化，再收缩通道乘积。 也接受最直接的纯循环实现，需要时使用嵌套循环。


## 6. Diffusion with batch-specific masks and mixing


x has shape [B,H,W,C], with values in [0,255]. mask has shape [B,H,W], and mix has shape [B,C,C]. Each batch has its own mask and mixing matrix; mix[b,c,d] weights source channel c into destination channel d. Perform exactly three rounds.

Keep outermost rows and columns unchanged, and also keep cells where mask[b,i,j]=0 unchanged. For each remaining cell, first calculate each source channel’s spatial average: (4*center + above + below + left + right) // 8. Round down before channel mixing.

Destination channel d receives sum_c(average[c]*mix[b,c,d]), clipped to [0,255]. All spatial reads in a round use values from the start of that round, including frozen neighbors. Apply updates together. Return the full [B,H,W,C] array. If H<3 or W<3, all input values remain unchanged.


Implementation hint: Use separate current and next buffers. Preserve the batch axis in the mask and matrix product; do spatial averaging before channel mixing. A straightforward implementation using only loops (nested where needed) is also accepted.


按批次掩码和混合矩阵进行扩散


x 的形状为 [B,H,W,C]，值在 [0,255]。mask 的形状为 [B,H,W]，mix 的形状为 [B,C,C]。每个批次有自己的掩码和混合矩阵；mix[b,c,d] 是源通道 c 到目标通道 d 的权重。恰好执行三轮。

最外侧的行和列保持不变，mask[b,i,j]=0 的单元也保持不变。对其余单元，先计算每个源通道的空间平均值：(4*自身 + 上 + 下 + 左 + 右) // 8。先向下取整，再混合通道。

目标通道 d 的值为 sum_c(average[c]*mix[b,c,d])，再限制到 [0,255]。同一轮的所有空间读取都使用本轮开始时的值，包括被冻结的邻居。整轮一起应用更新。返回完整的 [B,H,W,C] 数组。如果 H<3 或 W<3，所有输入值保持不变。


实现提示：使用独立的当前和下一轮缓冲区。掩码和矩阵乘法都保留批次轴；先计算空间平均值，再混合通道。 也接受最直接的纯循环实现，需要时使用嵌套循环。

