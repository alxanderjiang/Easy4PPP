# 批处理脚本 运行该脚本 
# 将依次对设定路径内的全部配置yaml文件进行执行
# Batch processing script, batch_prc.py
# all the configuration files in the set path will be processed by running this file

import os
xmls_path="xmls/"

for xml in os.listdir(xmls_path):
    os.system("python src/ppp_yaml.py {}{}".format(xmls_path,xml))