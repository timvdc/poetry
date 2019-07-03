def hmean(num):
    return len(num)/sum([1 / (n + 1e-20) for n in num])
