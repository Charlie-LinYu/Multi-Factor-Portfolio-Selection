import os
import csv
import pandas as pd

#path_os="C:\\Academic\\Georgia Tech\\Financial System-C++\\final project\\szss\\"
path_os="C:\\Academic\\Georgia Tech\\Financial System-C++\\final project\\project\\price_data\\"
os.chdir(path_os)

extension = '.csv'
res={}
i=1
#read all csv files in the selected folder and combine them together
for root, dirs_list, files_list in os.walk(path_os):
    for file in files_list:
        if os.path.splitext(file)[-1] == extension:
            dic_key = file[0:9].upper()
            file_name_path = os.path.join(root, file)
            #print(dic_key)
            with open(file_name_path,'r') as fin:
                csvin = csv.DictReader(fin)
                info = [line for line in csvin]
                res[dic_key]=pd.DataFrame(info)
                if info != []:
                    res[dic_key]=res[dic_key].set_index('Date')
            print(i)
            i=i+1

res=pd.Panel(res)
#res.to_hdf('factor_data.h5', 'key')
res.to_hdf('price_data.h5', 'key')

