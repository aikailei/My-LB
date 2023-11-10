
# 示例数据
all_means = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# 求均值的列表表达式
j = len(all_means)
k = len(all_means[0])
result = [sum(all_means[i][l] for i in range(j)) / j for l in range(k)]

# 打印结果
print(result)
