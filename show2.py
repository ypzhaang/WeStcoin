import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
x = np.array([1,2,3,4,5,6,7])
# #nyt-fine
# #micro
# y1 = [0.90,0.90,0.91,0.91,0.92,0.92,0.92]
# #macro
# y2 = [0.64,0.65,0.67,0.69,0.72,0.80,0.80]

# # #nyt-coarse
# # #micro
# y1 = [0.97,0.97,0.97,0.97,0.98,0.98,0.98]
# #macro
# y2 = [0.83,0.85,0.89,0.87,0.89,0.89,0.89]

#20news-coarse[0.87/0.85]
#micro
y1 = np.array([0.85,0.85,0.86,0.86,0.87,0.87,0.87])
#macro
y2 = np.array([0.79,0.80,0.82,0.84,0.85,0.84,0.85])


plt.title('20News-coarse')  # 折线图标题
plt.xlabel('Iteration')  # x轴标题
plt.plot(x, y1, marker='*', markersize=7)
plt.plot(x, y2, marker='v', markersize=7)


plt.legend([ 'Micro-F1', 'Macro-F1'])  # 设置折线名称

plt.grid()
plt.show()  # 显示折线图

