import pandas as pd
import numpy as np
from word_vec import *
from sklearn.datasets import load_svmlight_file  # 载入libsvm格式文件
from sklearn.externals import joblib

data_train = 'data/data_train.csv'
data_test = 'data/data_test.csv'
score_list = [0,1,2]
cate_list = ['食品餐饮', '旅游住宿', '金融服务', '医疗服务', '物流快递']
model_path_dict = {'食品餐饮': 'data/model-food.model', '旅游住宿': 'data/model-travel.model'
			, '金融服务': 'data/model-finance.model', '医疗服务': 'data/model-hospital.model'
			,'物流快递': 'data/model-trans.model'}
svm_model_dict = {'食品餐饮': 'svm-model/model-food.m', '旅游住宿': 'svm-model/model-travel.m'
			, '金融服务': 'svm-model/model-finance.m', '医疗服务': 'svm-model/model-hospital.m'
			, '物流快递': 'svm-model/model-trans.m'}

test_error_row = [11506, 14844, 27187]

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



def store_svm():
	cate_score_dict = read_train_file(data_train)
	# lenth = len(cate_score_dict['食品餐饮'][0])
	# print(lenth)
	# print(cate_score_dict['食品餐饮'][0][0]['content'])
	# sens = get_wordvec(cate_score_dict['食品餐饮'][0])
	for cate, v in cate_score_dict.items():
		print(cate,'*'*30)
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
		print(X)
		print(Y)
		print('x',len(X))
		print('y',len(Y))
		clf1 = svm.SVC(kernel='linear', C=1)

		# scores = cross_validation.cross_val_score(clf1, X, Y, cv=3)#5-fold cv
		# print(scores)


		clf1.fit(X,Y)
		joblib.dump(clf1, svm_model_dict[cate][:-2]+'1.m')  # 永久保存

		clf2 = svm.SVC(kernel='linear', C=1)
		clf2.fit(X1,Y1)
		joblib.dump(clf2,  svm_model_dict[cate][:-2]+'2.m')  # 永久保存



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
	res = pd.DataFrame(res_list)
	res.to_csv('result/jktian-v3.csv',index=None,header=None,encoding='gbk')

def test_custome():
	data_input = ['真难吃','好吃呀','从肯德基宅急送官网上订餐轻松，不用排队，优惠也很多，还可以自由的搭配，挺实惠的。从肯德基宅急送官网上订餐服务也很好，配送及时，网上什么都有，还有什么套餐推荐之类的，比实体店便宜一点，就是在饭点儿的时候，肯德基太慢了，这个是要饿死的节奏好嘛，可以快一点啦'
		,'蛋黄_南瓜非常好吃，狮子头也不错，特价小肘不好吃，全是骨头。'
		,'食品餐饮	人太多，上菜有些慢，味道还可以，有点小贵吧',]
	test(data_input)


def build_all_model():
	cate_score_dict = read_train_file(data_train)
	for k, v in cate_score_dict.items():
		buildmodel(v,model_path_dict[k])


if __name__=="__main__":


	# cross_score()
	# cate_score_dict = read_train_file(data_train)
	read_test_file(data_test)
	# cross_score()

	# lenth = len(cate_score_dict['食品餐饮'][0])
	# print(lenth)
	# print(cate_score_dict['食品餐饮'][0][0]['content'])
	# sens = get_wordvec(cate_score_dict['食品餐饮'][0])



