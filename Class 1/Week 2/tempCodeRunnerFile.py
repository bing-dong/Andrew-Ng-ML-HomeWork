def initialize_with_zeros(dim):
#     """
#         此函数为w创建一个维度为（dim，1）的0向量，并将b初始化为0。
#         参数：
#             dim  - 我们想要的w矢量的大小（或者这种情况下的参数数量）
#         返回：
#             w  - 维度为（dim，1）的初始化向量。
#             b  - 初始化的标量（对应于偏差）
#     """
#     w = np.zeros(shape = (dim,1))
#     b = 0
#     #使用断言来确保我要的数据是正确的
#     assert(w.shape == (dim, 1)) #w的维度是(dim,1)
#     assert(isinstance(b, float) or isinstance(b, int)) #b的类型是float或者是int

#     return (w , b)