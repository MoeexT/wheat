
def fun(arg, *args):
    print(args)
    print(str(args).replace('(', '').replace(')', ''))
    # print(list(args)[1])


def func(arg, **kwargs):
    print(list(kwargs.keys()))
    print(list(kwargs.values()))
    print(list(kwargs.items()))
    print(str(kwargs).replace('{', '').replace('}', '').replace(':', '='))


if __name__ == '__main__':
    fun(1, 'a')
    fun(1, 'a', True)

    func(1, url="google.com", uuid="ab1e33d4-0b84-11e9-99c4-54ee7586d9c5")
