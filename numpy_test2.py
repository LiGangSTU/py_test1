import numpy as np

# 创建数组


def createArray():
    # empty
    x = np.empty([3, 2], dtype=int)
    print(x)
    y = np.zeros(5)
    print(y)
    z = np.zeros((5,),dtype=np.int)
    print(z)
    # 自定义类型
    c = np.zeros((2,2),dtype=[('x','i4'),('y','i4')])
    print(c)
    # 通过已有数组来创建新的数组
    # x = [1,2,3]
    # x = (1,2,3) 元组
    x = [(1,2,3),(4,5)]
    a = np.asarray(x)
    print(a)
    return


def cutAndIndex():
    a = np.arange(10)
    s = slice(2,7,2) # 2-7 间隔为2
    print(a[s])
    # 或者直接切割也行
    b = a[2:7:2]
    print(b)
    c = np.array([[1,2,3],[3,4,5],[4,5,6]])
    print(c[...,1]) # 表示第2列元素
    print(c[1,...])
    print(c[...,1:])

    # 高级索引
    x = np.array([[1,2],[3,4],[5,6]])
    y = x[[0,1,2],[0,1,0]]
    print(y)

    # 花式索引
    x = np.arange(32).reshape((8,4))
    print(x)
    # 获取指定行
    print(x[[4,2,1,7]])
    # 传入多个索引数组
    print(x[np.ix_([1,5,7,2],[0,3])])
    return


# numpy的广播机制 当维数不一致时，自动广播
def Broadcast():
    a = np.array([[0, 0, 0],
                  [10, 10, 10],
                  [20, 20, 20],
                  [30, 30, 30]])
    b = np.array([1, 2, 3])
    print(a + b)
    return

def DieDai():
    a = np.arange(6).reshape(2, 3)
    print('原始数组是：')
    print(a)
    print('\n')
    print('迭代输出元素：')
    for x in np.nditer(a):
        print(x,end=",")
    print('\n')

    # 修改元素中的值 op_flags
    a = np.arange(0,60,5)
    a = a.reshape(3,4)
    print('原始数组是：')
    print(a)
    print('\n')
    for x in np.nditer(a,op_flags=['readwrite']):
        x[...]=2*x
    print('修改后的数组是：')
    print(a)
    return


def numpyChar():
    print('连接两个字符串：')
    print(np.char.add(['hello'],['xxx']))
    print('\n')
    print(np.char.add(['hello','hi'], [' xxx',' yy']))
    print('\n')
    # 多重连接 multiply
    print(np.char.multiply('Runoob ',3))
    # 字符串居中，并给两侧指定字符填充
    print(np.char.center('aaa',20,fillchar='-'))

    # 根据某个条件从数组中抽取元素
    x = np.arange(9.).reshape(3,3)
    print(x)
    condition = np.mod(x,2) == 0
    print(condition)
    print(np.extract(condition,x))
    return


def fileOp():
    a = np.array([1,2,3])
    np.savetxt('out.txt',a)
    b = np.loadtxt('out.txt')
    print(b)
    return


if __name__ == '__main__':
   fileOp()