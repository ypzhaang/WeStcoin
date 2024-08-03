import matplotlib.pyplot as plt

x = ["10%", "20%", "30%","40%","50%"]
#nyt-fine-micro
y1 = [0.93,0.92,0.91,0.90,0.86]
y2 = [0.92,0.91,0.90,0.89,0.87]
y3 = [0.93,0.92,0.90,0.89,0.86]
y4 = [0.92,0.91,0.90,0.88,0.86]
#nyt-fine-macro
# y1 = [0.83,0.80,0.72,0.78,0.61]
# y2 = [0.82,0.70,0.75,0.77,0.59]
# y3 = [0.79,0.74,0.68,0.56,0.53]
# y4 = [0.75,0.69,0.61,0.50,0.47]
#20news-coarse-macro
# y1 = [0.86,0.85,0.82,0.81,0.80]
# y2 = [0.84,0.80,0.81,0.80,0.77]
# y3 = [0.85,0.83,0.82,0.80,0.78]
# y4 = [0.84,0.79,0.78,0.78,0.73]
#20news-coarse-micro
# y1 = [0.89,0.87,0.87,0.86,0.85]
# y2 = [0.87,0.85,0.85,0.84,0.82]
# y3 = [0.88,0.87,0.86,0.84,0.83]
# y4 = [0.87,0.84,0.84,0.83,0.79]
#nyt-coarse-micro
# y1 = [0.98,0.98,0.93,0.93,0.94]
# y2 = [0.96,0.95,0.93,0.91,0.92]
# y3 = [0.97,0.97,0.94,0.93,0.93]
# y4 = [0.97,0.95,0.92,0.90,0.89]
#nyt-coarse-macro
# y1 = [0.90,0.89,0.77,0.79,0.69]
# y2 = [0.84,0.72,0.66,0.74,0.71]
# y3 = [0.87,0.85,0.68,0.67,0.75]
# y4 = [0.74,0.71,0.63,0.59,0.51]
plt.title('NYT-fine')  # 折线图标题
#plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.xlabel('The ratio of random label')  # x轴标题
plt.ylabel('Micro-F1')  # y轴标题
  # 绘制折线图，添加数据点，设置点的大小-------#WeStcoin
plt.plot(x, y2, linewidth=2, marker='v', markersize=8)#for NL
plt.plot(x, y3, linewidth=2, marker='s', markersize=8)#for Im3
plt.plot(x, y4, linewidth=2, marker='o', markersize=8)#HAN
plt.plot(x, y1, linewidth=2, marker='*', markersize=8)


plt.legend([ 'WeStcoin-forNL', 'WeStcoin-forIm3', 'HAN-supervised-Noise', 'WeStcoin'])  # 设置折线名称

plt.grid()
plt.show()  # 显示折线图

