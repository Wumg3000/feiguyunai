#导入类
from createclasses.classes import Student,Teacher


def main():
#输入一所大学名称
    #实例化Student类
    st01=Student("张三丰",30,"男","人工智能学院","图像识别")
    st02=Student("吴用",24,"男","人工智能学院","图像识别")
    #调用displayinfo方法
    st01.displayinfo()
    st02.displayinfo()
    #实例化Teacher类
    tch01=Teacher("李教授",40,"男","人工智能学院","自然语言处理")
    tch01.displayinfo()

##判断是否以主程序形式运行      
if __name__=='__main__':
    main()

