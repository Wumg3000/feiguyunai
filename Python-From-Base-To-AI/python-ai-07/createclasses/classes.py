'''创建类'''
class Person:
    '''定义父类Person'''
    def __init__(self,name,age,sex):
        self.name=name
        self.age=age
        self.sex=sex
    #定义方法，显示基本信息    
    def displayinfo(self):
        print("{},{},{})".format(self.name,self.age,self.sex))
        
#定义Student子类，继承Person类,新增两个参数std_college，std_profession
class Student(Person):
    '''定义子类Student，集成Person类'''
    def __init__(self,name,age,sex,std_college,std_profession):
        super(Student,self).__init__(name,age,sex)
        self.std_college=std_college
        self.std_profession=std_profession
        
    #重写方法，显示学生基本信息    
    def displayinfo(self):
        #重写父类中displayinfo方法
        print("Student({},{},{},{},{}))".format(self.name,self.age,self.sex,self.std_college,self.std_profession))

#定义子类Teacher，继承Person类,新增两个参数tch_college，tch_profession
class Teacher(Person):
    '''定义子类Teacher，集成Person类'''
    def __init__(self,name,age,sex,tch_college,tch_profession):
        super(Teacher,self).__init__(name,age,sex)
        self.tch_college=tch_college
        self.tch_profession=tch_profession
        
    #重写方法，显示教师基本信息 
    def displayinfo(self):
        #重写父类中displayinfo方法
        print("Teacher({},{},{},{},{}))".format(self.name,self.age,self.sex,self.tch_college,self.tch_profession))
