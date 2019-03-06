import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import codecs,sys,string,re
import gensim
from gensim.models import word2vec
from sklearn import cross_validation, svm
from sklearn.datasets import load_svmlight_file  # 载入libsvm格式文件
from sklearn.externals import joblib

from const_data import *
from readfile import *
from word_vec import *


def store_svm(punish_index):
	cate_score_dict = read_train_file(data_train)
	# lenth = len(cate_score_dict['食品餐饮'][0])
	# print(lenth)
	# print(cate_score_dict['食品餐饮'][0][0]['content'])
	# sens = get_wordvec(cate_score_dict['食品餐饮'][0])
	for cate, v in cate_score_dict.items():
		if cate == '食品餐饮':	continue
		# print(cate,'*'*30)
		vec_pos_neg = getvecs(cate, v)
		X = []
		Y = []
		X1 = []
		Y1 = []
		for std in score_list:
			for (vecsArray,col,content,score) in vec_pos_neg[std]:
				X.append(vecsArray)
				if std==1 or std==2:
					X1.append(vecsArray)
					Y1.append(score)
					score = 1
				Y.append(score)
		# print(X)
		# print(Y)
		# print('x',len(X))
		# print('y',len(Y))
		punish = punish_list[punish_index]
		if cate!='旅游住宿':
			clf1 = svm.SVC(kernel='linear', C=punish)
			clf1.fit(X,Y)
			joblib.dump(clf1, svm_model_dict[cate][:-2]+'c'+str(punish)+'1.m')  # 永久保存

		clf2 = svm.SVC(kernel='linear', C=punish)
		clf2.fit(X1,Y1)
		joblib.dump(clf2,  svm_model_dict[cate][:-2]+'c'+str(punish)+'2.m')  # 永久保存


if __name__=="__main__":
	store_svm(4)
