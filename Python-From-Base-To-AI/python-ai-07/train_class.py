#导入模块class_person中的Student类
from class_person import Student as st


def main():
#输入一所大学名称
    str=input("输入一所大学名称: ")
    #实例化st类
    s1=st("张华",21,str)
    #调用display方法
    s1.display()

##判断是否以主程序形式运行      
if __name__=='__main__':
    main()
