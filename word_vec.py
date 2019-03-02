import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import codecs,sys,string,re
import gensim
from gensim.models import word2vec
from sklearn import cross_validation
from sklearn import svm

data_train = 'data/data_train.csv'
data_test = 'data/data_test.csv'
score_list = [0,1,2]
cate_list = ['食品餐饮', '旅游住宿', '金融服务', '医疗服务', '物流快递']
model_path = 'data/mymodel2.model'
model_path_dict = {'食品餐饮': 'data/model-food.model', '旅游住宿': 'data/model-travel.model'
			, '金融服务': 'data/model-finance.model', '医疗服务': 'data/model-hospital.model'
			,'物流快递': 'data/model-trans.model'}

def get_wordvec(df,mode):
	stopkey = [w.strip() for w in codecs.open('data/stopWord.txt', 'r', encoding='utf-8').readlines()]
	# print('df',type(df))
	# print('df-content',type(df['content']))
	if mode=='train':
		sens = list(df['content'])
	else:
		sens = df
	sens_split = [] 
	# 清洗文本
	def clearTxt(line):
		line = str(line)
		if line != '': 
			line = line.strip()
			# intab = ""
			# outtab = ""
			# trantab = str.maketrans(intab, outtab)
			# pun_num = string.punctuation + string.digits
			# line = line.encode('utf-8')
			# line = line.translate(trantab,pun_num)
			# line = line.decode("utf8")
			#去除文本中的英文和数字
			line = re.sub("[a-zA-Z0-9]","",line)
			#去除文本中的中文符号和英文符号
			line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+", "",line) 
		return line

	#文本切割
	def sent2word(line):
		segList = jieba.cut(line,cut_all=False)    
		segSentence = ''
		for word in segList:
			if word != '\t':
				segSentence += word + " "
		return segSentence.strip()

	#删除停用词
	def delstopword(line,stopkey):
		wordList = line.split(' ')          
		sentence = ''
		for word in wordList:
		    word = word.strip()
		    if word not in stopkey:
		        if word != '\t':
		            sentence += word + " "
		return sentence.strip()

	for sen in sens:
		line = clearTxt(sen)
		seg_line = sent2word(line)
		sentence = delstopword(seg_line,stopkey)
		sens_split.append(sentence)
	return sens_split


# def make_single_vec_file():
# 	file_list = ['data/食品餐饮_0.csv','data/食品餐饮_12.csv']
# 	for file,i in zip(file_list,range(len(file_list))):
# 		df=pd.read_csv(file,header=None,sep=',') #filename可以直接从盘符开始，标明每一级的文件夹直到csv文件，header=None表示头部为空，sep=' '表示数据间使用空格作为分隔符，如果分隔符是逗号，只需换成 ‘，’即可。
# 		df[4] = get_wordvec(df)
# 		print(str(i))
# 		df.to_csv('data/食品餐饮_vec'+str(i)+'.csv',header=None,index=False)


# def buildmodel():
# 	file = 'data/食品餐饮.csv'
# 	df = pd.read_csv(file,header=None,sep=',')
# 	sens = get_wordvec(df)
# 	# df[4] = sens

def buildmodel(df,modelpath):
	model_data = []
	for index in score_list:
		model_data += get_wordvec(df[index],'train')
	mymodel = word2vec.Word2Vec(model_data, min_count=1)
	mymodel.save(modelpath)






def getvecs(cate, df):
		# 构建文档词向量 
	def buildVecs(df,model):
		# print('df-content',len(df['content']))
		fileVecs = []
		def getWordVecs(wordList,model):
			vecs = []
			for word in wordList:
				word = word.replace('\n','')
				#print word2vec
				try:
				    vecs.append(model[word])
				except KeyError:
					continue
			return np.array(vecs, dtype='float')
		# print('df[score]-len',len(df['score']))
		# i = 0
		for col,content,score in zip(df['col'],df['content'],df['score']):
			line = str(content)
			# wordList = line.split(' ')
			wordList = line
			vecs = getWordVecs(wordList,model)
			if len(vecs) >0:
				# i+=1
				vecsArray = sum(np.array(vecs))/len(vecs) # mean
			#print vecsArray
			#sys.exit()
				fileVecs.append((vecsArray,col,content,score))
		# print(i)
		# print(len(fileVecs))
		return fileVecs


	mymodel = gensim.models.Word2Vec.load(model_path_dict[cate])
	vec_pos_neg = {}
	# vec_list = []
	for index in score_list:
		vec_pos_neg.setdefault(index,[])
		vec_pos_neg[index] = buildVecs(df[index],mymodel)
		# print(vec_pos_neg[index][0])
	return vec_pos_neg


	# 	vec_pos_neg[str(i)] = vecs
	# return vec_pos_neg




		# df[5] = vecs
	# print(vec_pos_neg)
	# model = gensim.models.Word2Vec(sens, min_count=1)
	# model.save('data/mymodel')
	# df=pd.read_csv('data/食品餐饮_vec1.csv',header=None,sep=',')
	# print(df.head())

def get_test_value(cate, df,df_cols):
	mymodel = gensim.models.Word2Vec.load(model_path_dict[cate])
		# print('df-content',len(df['content']))
	fileVecs = []
	def getWordVecs(wordList,model):
		vecs = []
		for word in wordList:
			word = word.replace('\n','')
			#print word2vec
			try:
			    vecs.append(model[word])
			except KeyError:
				continue
		return np.array(vecs, dtype='float')
	# i = 0
	for content,col in zip(df,df_cols):
		line = str(content)
		# wordList = line.split(' ')
		wordList = line
		vecs = getWordVecs(wordList,mymodel)
		if len(vecs) >0:
			vecsArray = sum(np.array(vecs))/len(vecs) # mean
		#print vecsArray
		#sys.exit()
			fileVecs.append((col,vecsArray))
	# print(i)
	# print(len(fileVecs))
	return fileVecs
