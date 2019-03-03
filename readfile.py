import pandas as pd
import numpy as np

from const_data import *

def read_train_file(file):
	def get_cate_list(df):
		cate_list = {}
		for index,row in df.iterrows():
			# print(row['cate'])
			cate_list.setdefault(row['cate'],0)
			cate_list[row['cate']] = 1
		cate_list = [k for k in cate_list.keys()]
		return cate_list

	df=pd.read_csv(file,header=None,sep='\t',names=['col','cate','content','score'],encoding='gbk')

	res_dict = {}
	for cate in cate_list:
		res_dict.setdefault(cate,{})
		for score in score_list:
			# res_dict[cate].setdefault(score,0)
			data = df.loc[(df['score']==score)&(df['cate']==cate)]
			res_dict[cate][score] = data
	# print(cate_score_dict)
	return res_dict





