
import random
class Login:
    """登录ATM机系统"""
    #插卡进入系统
    #import random
	
    def insertcard(self,users_info):
        print("#"*53)
        print("\n")
        print( " "*16  +"工商银行欢迎您"+ " "*16 )
        print("\n")
        print("请插卡：")
        #从用户存款信息中，随机采样一个卡号
        acc_list=[]
        for key in users_info.keys():
            acc_list.append(key)
        self.card = random.sample(acc_list, 1)[0]
        print(self.card)
        print("#"*53)
        return self.card
    #选择功能
    def selectfunctions(self):
        print("#"*53)
        print("#         查询（1）            取款（2）            #")
        print("#         存款（3）            退卡（q）            #")
        print("#                                                   #")
        print("#"*53)