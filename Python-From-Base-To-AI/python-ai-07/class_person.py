class  Person:
    '''表示人的基本信息'''
    pernum=0   #类属性
    __percount=1000  #定义类的私有属性
    
    #定义类的构造函数，初始化类的基本信息
    def __init__(self,name,age):
        self.name= name
        self.age=age
        Person.pernum+=1
        self.__pwd=123456   ##实例私有属性
    def display(self):
        print("person(姓名:{},年龄:{})".format(self.name,self.age))
    #通过添加装饰器，把方法变为属性    
    @property    
    def display_pernum(self):
        print(Person.pernum) 
    #通过添加装饰器，把私有属性变为只读属性
    @property
    def display_percount(self):
        return Person.__percount
		

class  Student(Person):
    '''表示学生的基本信息，继承Person类'''
      
    #定义类的构造函数，初始化类的基本信息
    def __init__(self,name,age,university):
        super(Student,self).__init__(name,age)
        self.university=university        
    def display(self):
        print("Student(姓名:{},年龄:{},所在大学:{})".format(self.name,self.age,self.university))
