import func_op
import sys


def main():
#输入一个自然数n
    #n=input("输入一个自然数: ")
	#从命令行获取参数
    n=sys.argv[1] 
	#进行数据类型转换
    n=int(n)	
	#如果命令行运行:python train_sum.py  100  
	#则sys.argv[0]是train_sum.py,sys.argv[1]是100
    #调用模块func_op中的函数sum_1n
    result=func_op.sum_1n(n)
    print("1到{}的连续自然数的和为{}".format(n,result))

##判断是否以主程序形式运行      
if __name__=='__main__':
    main()
