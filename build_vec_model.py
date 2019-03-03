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


def buildmodel(df,modelpath):
	model_data = []
	for index in score_list:
		model_data += get_wordvec(df[index],'train')
	mymodel = word2vec.Word2Vec(model_data, min_count=1)
	mymodel.save(modelpath)


def build_all_model():
	cate_score_dict = read_train_file(data_train)
	for k, v in cate_score_dict.items():
		buildmodel(v,model_path_dict[k])


if __name__=="__main__":
	build_all_model()