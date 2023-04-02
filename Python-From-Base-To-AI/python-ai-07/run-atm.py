#导入模块
import  getpass
import  time
import json
import codecs
import random
#导入这次创建的模块
from atm_system.dataset.dataset import dict_json,json_file_to_dict
from atm_system.atm.atm import ATM
from atm_system.login.login import Login

# 主函数
def main():
    # 登录ATM机系统
    login = Login()

    # 进入提示插卡界面
    enterpwd=login.insertcard(users_info)
    #提示输入密集界面
    
    atm = ATM(enterpwd,users_info)

    while True:
        login.selectfunctions()
        # 等待用户操作
        option = input("请输入您的操作：")
        
        if option == "1":
            # print("查询")
            atm.searchUserInfo()
        elif option == "2":
            # print("取款")
            atm.getMoney()
        elif option == "3":
            # print("存储")
            atm.saveMoney()
        elif option == "5":
            # print("转4")
            atm.transferMoney()        
        elif option == "q":
            # print("退出")
            break
        
        time.sleep(2)

if __name__ == "__main__":
    #提取用户信息
    users_info={}
    users_info.update(json_file_to_dict())
    main()
    
