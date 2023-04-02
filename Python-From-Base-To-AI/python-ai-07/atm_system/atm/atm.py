
import  getpass
import  time
from atm_system.dataset.dataset import dict_json,json_file_to_dict

class ATM:
    #初始化卡号，后台根据卡号获取密码及余额
    def __init__(self, account,users_info):
        self.account = account
        #根据卡号获取用户密码及余额
        self.users_info=users_info
        user_info=self.users_info[self.account]
        self.name=user_info[0]
        self.userpwd=user_info[1]
        self.usermoney=user_info[2]
    
    # 查询
    def searchUserInfo(self):
        # 验证密码
        if not self.checkPasswd(self.userpwd):
            print("密码输入有误！")
            return -1
        print("账号：%s   余额：%d" % (self.account, self.usermoney))

    # 取款
    def getMoney(self):
        # 验证密码
        if not self.checkPasswd(self.userpwd):
            print("密码输入有误！")
            return -1

        # 开始取款
        amount = int(input("验证成功！请输入取款金额："))
        if amount > self.usermoney:
            print("取款金额有误，取款失败！")
            return -1
        if amount < 0:
            print("取款金额有误，取款失败！")
            return -1
        self.usermoney -= amount 
        #更新用户存款信息
        self.users_info[self.account][2]=self.usermoney
        dict_json(self.users_info)
        print("您取款%d元，余额为%d元！" % (amount, self.usermoney))
        

    # 存款
    def saveMoney(self):
        # 验证密码
        if not self.checkPasswd(self.userpwd):
            print("密码输入有误！")
            return -1

        # 开始存款
        amount = int(input("验证成功！请输入存款金额："))
        if amount < 0:
            print("存款金额有误，存款失败！")
            return -1
        self.usermoney += amount
        #更新用户存款信息
        self.users_info[self.account][2]=self.usermoney
        dict_json(self.users_info)
        print("您存款%d元，最新余额为%d元！" % (amount,self.usermoney))

    # 转账
    def transferMoney(self):
        # 验证密码
        if not self.checkPasswd(self.userpwd):
            print("密码输入有误！")
            return -1

        # 开始转账
        amount = int(input("验证成功！请输入转账金额："))
        if amount > self.usermoney or amount < 0:
            print("金额有误，转账失败！")
            return -1

        newcard = input("请输入转入账户：")
        newuser = self.allUsers.get(newcard)
        if not newuser:
            print("该卡号不存在，转账失败！")
            return -1
        
        user.card.cardMony -= amount
        newuser.card.cardMony += amount
        time.sleep(1)
        print("转账成功，请稍后···")
        time.sleep(1)
        print("转账金额%d元，余额为%d元！" % (amount, user.card.cardMony))

    # 验证密码
    def checkPasswd(self, realPasswd):
        for i in range(3):
            enterpwd = getpass.getpass("请输入密码:")
            if enterpwd == realPasswd:
                return True
            else:
                print("密码错误，请重新输入！")
        return False    
