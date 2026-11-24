g = (x * 2 for x in range(3))
print(list(g))    # [0, 2, 4]
print(list(g))    # [], бо генератор уже вичерпано
