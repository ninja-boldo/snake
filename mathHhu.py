'''def func(n):
    return abs(3*(n**2) - 25)'''

'''def func(n):
    return ( (n**3) + (-5*(n**2)) + 4)'''

'''def func(n):
    return abs(n - 4)'''

def func(n):
    return (2*(n**2) - 8)

res = []
iterations = 1000000
for i in range(iterations):
    i = i - iterations * 0.5
    i = i / 1000
    if i < 11:
        print(f"func({i}) = {func(i)}")
    res.append(func(i))
    
print(f"min: {min(res)}")
print(f"max: {max(res)}")