
def test(n):
    for i in range(n):
        yield i


print(test(4))

for i in test(5):
    print(i)
