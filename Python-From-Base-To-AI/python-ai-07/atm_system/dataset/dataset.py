"""把字典数据存储到文件，然后再把文件转换为字典"""
#把字典数据保存到磁盘
import json
import codecs
	
def dict_json(users_info):
    with codecs.open('users_info.json','w+', 'utf-8') as outfile:
        json.dump(users_info, outfile, ensure_ascii=False)
        outfile.write('\n')	
			

def json_file_to_dict():
    with open('users_info.json', 'r',encoding='utf-8') as f:
        dict = json.load(fp=f,encoding='utf-8')
        return dict