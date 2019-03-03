# 文本情感极性分析

1. 句子＝＝》词向量
   1. jieba分词
   2. 去除停用词
2. 词向量的数值化
   1. word2vec框架
3. 模型搭建
   1. svm

# 运行环境

python3.6

库：genism, pandas, jieba, word2vec

详情见requirements.txt

# 最终效果

官方测试：87.03%

# 文件目录结构

1. 源文件：
- main.py # 通过选取test.py中相应的函数，运行文件
- const_data.py # 存储常量
- readfile.py # 读取训练数据
- build_vec_model.py # 构建数值化词向量模型
- build_svm_model.py # 构建svm模型
- test.py # 读取测试数据，使用模型，输出结果
- mess.py # 垃圾存储（忽略）
2. 文件夹
- data # 初始数据：训练数据以及测试数据
- result # 预测结果的csv文件
- vec-value-model # 经过word2vec处理得到的词向量模型
- svm-model # 训练得到的svm模型。三分类，所以使用两个模型。每个类别（食品/旅游）使用两个svm模型