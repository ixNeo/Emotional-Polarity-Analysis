import csv

def getdata():
	file = 'data/data_train.csv'
	type_dict = {}
	data = []
	with open(file,'r+',encoding='gbk') as f:
		f_csv = csv.reader(f,delimiter='\t')
		for row in f_csv:
			# type_list.setdefault(row[1],0)
			type_dict[row[1]] = 1
			if row[1]=='食品餐饮':
				data.append(row)

	type_list = []
	for k in type_dict.keys():
		type_list.append(k)
	with open(type_list[0]+'.csv','w') as f:
		f_csv = csv.writer(f)
		f_csv.writerows(data)


def splitdata():
	score0 = []
	score1 = []
	score2 = []
	file = '食品餐饮.csv'
	with open(file,'r+') as f:
		f_csv = csv.reader(f)
		for row in f_csv:
			if row[3]=='0':
				score0.append(row)
			elif row[3]=='1':
				score1.append(row)
			elif row[3]=='2':
				score2.append(row)
			else:
				print('score not in range')
	for i in range(3):
		with open('食品餐饮_'+str(i)+'.csv','w') as f:
			f_csv = csv.writer(f)
			f_csv.writerows(eval('score'+str(i)))

