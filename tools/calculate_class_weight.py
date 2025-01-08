data = {
    'Low': 12680802,
    'Middle-Low': 13598315,
    'Middle': 9744128,
    'Middle-High': 7919316,
    'High': 31489375
}

# 计算总数
total = sum(data.values())

# 计算每个类别的频率
frequencies = {key: value / total for key, value in data.items()}
# frequencies

# 计算每个类别的权重，权重与频率成反比
weights = {key: 1 / freq for key, freq in frequencies.items()}
# weights
# 归一化权重
total_weight = sum(weights.values())
normalized_weights = {key: weight / total_weight for key, weight in weights.items()}
# normalized_weights
print(list(normalized_weights.values()))
