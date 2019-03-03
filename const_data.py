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


data_train = 'data/data_train.csv'
data_test = 'data/data_test.csv'
score_list = [0,1,2]
cate_list = ['食品餐饮', '旅游住宿', '金融服务', '医疗服务', '物流快递']
model_path = 'data/mymodel2.model'
model_path_dict = {'食品餐饮': 'data/model-food.model', '旅游住宿': 'data/model-travel.model'
			, '金融服务': 'data/model-finance.model', '医疗服务': 'data/model-hospital.model'
			,'物流快递': 'data/model-trans.model'}
svm_model_dict = {'食品餐饮': 'svm-model/model-food.m', '旅游住宿': 'svm-model/model-travel.m'
			, '金融服务': 'svm-model/model-finance.m', '医疗服务': 'svm-model/model-hospital.m'
			, '物流快递': 'svm-model/model-trans.m'}
test_error_row = [11506, 14844, 27187]