import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 通过传递一个list对象来创建一个Series，pandas会默认创建整型索引
s = pd.Series([1,3,5,np.nan,6,8])
print(s)

# 通过创建时间索引和列标签来创建一个DataFrame
dates = pd.date_range('20201111',periods=6)
print(dates)
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))
print(df)

# 通过传递一个能够被转换为类似序列结构的字典对象来创建一个DataFrame
df2 = pd.DataFrame({'A' : 1.,
                   'B' : pd.Timestamp('20201111'),
                    'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D' : np.array([3]*4,dtype='int32'),
                    'E' : pd.Categorical(["test","train","haha","hehe"]),
                    'F' : 'foo'})
print(df2)

# 查看不同列类型
print(df2.dtypes)

# 分段查看数据
print(df.head(2))
print(df.tail(3))

# 显示索引、列和底层的numpy数据
print(df.index)
print(df.columns,df.values)
# 对于数据的快速统计
print(df.describe())

# 对于数据的转置
print(df.T)

# 对于数据的轴排序
print(df.sort_index(axis=1,ascending=False))

# 对于数据的值排序
print(df.sort_values(by='B'))

# 数据的选择
# 单独选择
print(df['A'])
# 利用[]来分片选择
print(df[0:3])
print(df['20201115':'20201116'])

# 通过标签选择 loc
print(df.loc[dates[0]])
print(df.loc[:,['A','B']])
print(df.loc['20201111':'20201113',['A','B']])
print(df.loc['20201111',['A','B']])
# 获取一个标量
print(df.loc[dates[2],['A']])
# 快速访问一个标量，与上面类似
print(df.at[dates[2],'D'])

# 通过位置选择
print(df.iloc[3])
print(df.iloc[3:5,0:2])

# 布尔索引
print(df[df > 0])
print(df[df.A > 0])
# 使用isin方法过滤
df2 = df.copy()
df2['E'] = ['one','one','two','three','four','three']
print(df2)
print(df2[df2['E'].isin(['two','four'])])

# 设置
s1 = pd.Series([1,2,3,4,5,6],index=pd.date_range('20201111', periods=6))
print(s1)
df['F'] = s1
'''
# 通过标签来设置新的值
df.at[dates[0]] = 0
# 通过位来设置新的值
df.iat[2,2] = 0
'''

# 通过numpy数组来设置一组新值
df.loc[:,'D'] = np.array([5] * len(df))
print(df)

df2 = df.copy()
df2[df2 > 0] = -df2
print(df2)

# 处理缺失值
# reindex 对指定轴上的索引进行改变，并返回原始数据的一个拷贝
df1 = df.reindex(index=dates[0:4],columns=list(df.columns)+['E'])
df1.loc[dates[0]:dates[1],'E'] = 1
print(df1)
# 去掉有异常值的行
print(df1.dropna(how='any'))
# 对有异常值的行填充
print(df1.fillna(value=5))
# 对数据进行布尔填充，换句话说就是判断是否有异常值的存在
print(pd.isnull(df1))

# 统计相关（一般不包括缺失值）
# 执行描述性统计
print(df.mean())
print(df.mean(1))

# 对不同维度进行辐射广播  ??? 此处在考虑考虑
s = pd.Series([1,3,5,np.nan,6,8],index=dates).shift(2)
print(s)
df.sub(s,axis='index')
print(df)

# 数据的应用 Apply
df.apply(np.cumsum)
print(df)
df.apply(lambda x: x.max() - x.min())
print(df.mean())

# 直方图和离散化
s = pd.Series(np.random.randint(0,7,size=10))
print(s)
# 统计数字出现次数
print(s.value_counts())

# 字符串方法 统计字符串的时候不区分大小写
s = pd.Series(['A','B','C','Abab','Dada',np.nan,'CABA','dat','der'])
print(s.str.lower())

# 数据合并 Concat  np.random.randn返回一个或一组样本，具有标准正态分布，其中dn表示表格维度
# 10*4的表格
df = pd.DataFrame(np.random.randn(10,4))
pieces = [df[:2],df[4:6],df[8:]]
print(pd.concat(pieces))
print(df)
# Join 方法，类似 SQL语句的合并
left = pd.DataFrame({'key':['foo','bar'],
                     'lval':['1','2']})
right = pd.DataFrame({'key':['foo','bar'],
                      'rval':['3','4']})
print(pd.merge(left,right,on='key'))

#Append 一行连接到另一个DataFrame上
df = pd.DataFrame(np.random.randn(8,4),columns=['A','B','C','D'])
s = df.iloc[3]
df.append(s,ignore_index=True) #忽视索引值的判断
print(df)

# group分组  ！！！为什么会出现排序错乱问题
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar','foo', 'bar', 'foo', 'foo'],
                   'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)})
print(df.groupby('A').sum())
# 通过多个列进行分组形成一个层次索引，然后执行函数
print(df.groupby(['A','B']).sum())

# 改变形状 stack

# 数据透视表 pivot_table ？

# 时间序列 对频率转换进行重新采样时具有简单，强大，高校的功能
# 时区表示
rng = pd.date_range('1/1/2020 00:00',periods=5,freq='D')
ts = pd.Series(np.random.randn(len(rng)),rng)
print(ts)
ts_utc = ts.tz_localize('UTC')
print(ts_utc)
# 时区转换
print(ts_utc.tz_convert('US/Eastern'))
# 时间跨度转换
ps = ts.to_period()
print(ps)
print(ps.to_timestamp())
# 时期和时间戳之间的转换使得可以使用一些方便的算术函数

#Categorical 绝对的 一种数据类型 离散特征，分类特征   ！！！还需要补充
df = pd.DataFrame({"id":[1,2,3,4,5,6],"raw_grade":['a' ,'b','b','a','a','e']})
# 提取出数据的特征值，或者是不重复的数据
df["grade"] = df["raw_grade"].astype("category")
print(df["grade"])

# 画图
ts = pd.Series(np.random.randn(1000),index=pd.date_range('1/1/2020',periods=1000))
ts = ts.cumsum()
plt.figure()
ts.plot()
plt.legend(loc = 'best')
print(ts.plot())

# 数据的导入与保存
# 写入csv文件
df.to_csv('foo.csv')
# 从文件中读取
pd.read_csv('foo.csv')

# # HDF5 存储
# # 写入HFD5 存储
# df.to_hdf('foo.h5','df')
# # 从HDF5存储中读取
# pd.read_hdf('foo.h5','df')

np.random.seed(1234)
df = pd.DataFrame(np.random.randint(10,50,20).reshape(5,4),
                  index=[['A','A','A','B','B'],
                         [1,2,3,1,2]],
                  columns=[['X','X','X','Y'],
                           ['x1','x2','x3','y1']])
print(df)