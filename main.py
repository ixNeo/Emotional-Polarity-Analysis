from test import *
from build_vec_model import *
from build_svm_model import *

if __name__=="__main__":

	# cross_score()
	# cate_score_dict = read_train_file(data_train)
	# read_test_file(data_test)
	# cross_score()

	# lenth = len(cate_score_dict['食品餐饮'][0])
	# print(lenth)
	# print(cate_score_dict['食品餐饮'][0][0]['content'])
	# sens = get_wordvec(cate_score_dict['食品餐饮'][0])

	# build_all_model(300)
	# for i in range(7,12):
	# store_svm(3)
	read_test_file(data_test,4)
