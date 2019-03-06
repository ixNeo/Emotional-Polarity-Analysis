import pandas as pd


def store_predict_res():
	file_res = "data/predict_result.csv"
	file_test = "data/data_test.csv"
	data_res = pd.read_csv(file_res,names=['value'])
	data_test = pd.read_csv(file_test,header=None,sep='\t',encoding='gbk')
	# print("data-res",data_res.shape)
	# print('data-test',data_test.shape)
	# data_res.to_csv('data/predict_res.csv',index=None,header=None,encoding='gbk')
	# print(data_test.head())
	print(data_res.head())
	print('*'*30)

	v1 = []
	v2 = []
	for index,row in data_res.iterrows():
	# print(row['cate'])
		sen = row['value'].split(',')
		v1.append(int(sen[0]))
		v2.append(int(sen[1]))

	res_list = [sen for sen in zip(v1,v2)]
	# print(res_list[:5])
	res = pd.DataFrame(res_list)
	res.to_csv('data/predict_res.csv',index=None,header=None,encoding='gbk')


def get_error_list(file):
	df = pd.read_csv(file,names=['col','score'])
	df_cols = df['col']
	# std_range = range(1,35158)
	err_list = []
	for i in range(len(df_cols)-1):
		if df_cols[i]!=df_cols[i+1]-1:
			err_list.append(df_cols[i]+1)
	return df,err_list


def complete_res(df,file,error_list,error_dict):
	for col in error_list:
		if col in error_dict.keys():
			score = error_dict[col]
			insertRow = pd.DataFrame([[col,score]],columns = ['col','score'])
			gap = df.loc[(df['col']==col-1)].index[0]
			above = df.loc[:gap]
			below = df.loc[gap+1:]
			df = pd.concat([above,insertRow,below],ignore_index = True)
	df.to_csv(file[:-4]+'-modify.csv',index=None,header=None,encoding='gbk')



if __name__=="__main__":
	# file = "result/jktian-v11-s300-c1000.csv"
	file = 'result/jktian-v4.csv'
	error_dict = {11506:0,14844:1, 27187:1,35141:2}
	df, error_list = get_error_list(file)
	complete_res(df,file,error_list,error_dict)
	# data = pd.read_csv('result/jktian-v4.csv',header=None,encoding='gbk')
	# print(data)
# for i in df_cols:
# 	print(i)
# for index,row in df.iterrows():
# 	row['col']

