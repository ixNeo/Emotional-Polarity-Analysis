import pandas as pd
import numpy as np
import gensim
from gensim.models import word2vec
from sklearn import cross_validation, svm
from sklearn.datasets import load_svmlight_file  # 载入libsvm格式文件
from sklearn.externals import joblib


from const_data import *
from readfile import *
from word_vec import *

def test(data_input,data_cols, cate):
	vec_input = get_wordvec(data_input,'test')
	col_value_list = get_test_value(cate,vec_input,data_cols)
	value_input = [i[1] for i in col_value_list]
	col_input = [i[0] for i in col_value_list]
	print(cate,': vec-value: ',len(value_input))
	def predict(clf,value_input):
		score_input =  clf.predict(value_input)
		return score_input
	clf1 = joblib.load(svm_model_dict[cate][:-2]+"1.m")
	clf2 = joblib.load(svm_model_dict[cate][:-2]+"2.m")
	res = []
	for col, sen, value, score1 in zip(col_input, data_input, value_input,predict(clf1,value_input)):
		if score1!=0:
			score1 = predict(clf2,value)[0]
		res.append((col,score1))
	# res = [res_single[1] for res_single in res]
	# print(res)
	return res



def read_test_file(file):
	df=pd.read_csv(file,header=None,sep='\t',names=['col','cate','content','score'],encoding='gbk')
	# res_dict[cate].setdefault(score,0)
	res_list = []
	for cate in cate_list:
		# print('*'*50)
		data = df.loc[df['cate']==cate]
		# print(cate,': raw-data: ',len(data))
		test_score = test(data['content'],data['col'],cate)
		# print(cate,': test-score: ',len(test_score))
		res_list = res_list + test_score
	# res_list = list(zip(range(1,len(res_list)+1),res_list))
	# res = pd.DataFrame(res_list)
	# res.to_csv('result/jktian-v3.csv',index=None,header=None,encoding='gbk')



def test_custome():
	data_input = ['真难吃','好吃呀','从肯德基宅急送官网上订餐轻松，不用排队，优惠也很多，还可以自由的搭配，挺实惠的。从肯德基宅急送官网上订餐服务也很好，配送及时，网上什么都有，还有什么套餐推荐之类的，比实体店便宜一点，就是在饭点儿的时候，肯德基太慢了，这个是要饿死的节奏好嘛，可以快一点啦'
		,'蛋黄_南瓜非常好吃，狮子头也不错，特价小肘不好吃，全是骨头。'
		,'食品餐饮	人太多，上菜有些慢，味道还可以，有点小贵吧',]
	test(data_input)
