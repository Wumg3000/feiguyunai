import func_op

#定义一个主函数
def main():
#输入一个自然数n
    n=input("输入一个自然数: ")
    #把字符型转换为整数型
    n=int(n)
    #调用模块func_op中的函数sum_1n
    result=func_op.sum_1n(n)
    print("1到{}的连续自然数的和为{}".format(n,result))

##判断是否以主程序形式运行      
if __name__=='__main__':
    main()
