#定义一个函数，累加截止自然数为n，作为参数传给这个函数
def sum_1n(n):
    """该函数的参数为自然数n，其功能为累加从1到n的n个连续自然数"""
    #定义一个存放累加数的变量
    j=0
    #用range(1,n+1)生成1到n连续n个自然数
    for i in range(1,n+1):
       j+=i
    #把累加结果作为返回值
    return j


#定义一个函数，接受任意数量的参数
def calc_sum(*lst):
    """累加所有参数"""
    sum=0
    for i in lst:
        sum+=i
    return sum     
