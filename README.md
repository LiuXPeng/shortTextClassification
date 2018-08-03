# shortTextClassification
&emsp;&emsp;对英文短文本进行分类，文本为英文新闻标题，长度集中在3-10个词之间，
类别标签为科技类（+1）、非科技类（-1），共计42万余条数据。
## 程序
### Preprocess.py
&emsp;&emsp;所使用语料摘自kaggle，对数据无用属性进行删除；按照目标要求，修改标签。在这里遇到了数据异常的情况：大致有分布不规律的200-300条词预处理达不到预期结果，debug发现，原因是数据中个别样本分隔符异常，最终删除异常样本。对数据进行分词，安照英文本身空格分隔分词，并且筛除停词库中的词语。<br>
&emsp;&emsp;特征提取方法主要考虑词袋模型和word embeddings模型：对于词袋模型，因为文本太短，绝大多数样本往往只有一个或者没有热词，造成向量矩阵的稀疏，而又因为此原因，不得不加长向量长度，以避免全0向量出现，最后实则实现的是一个复杂化的关键词索引模型，与关键词索引模型相比，加大了计算量，精度反而下降；word embeddings模型，通过向量化训练过程，可以引入外部语料，增加了分类模型的泛化能力，可以综合考量整个文本所有分词的语义，因此选择word embeddings模型。<br>
&emsp;&emsp;使用42万条数据，训练word2vec模型，之后对每个样本分词的向量取平均，生成每个样本的向量；后期改进模型时候发现，42万条数据规模太小，最终引进谷歌新闻的word2vec模型。<br>

#### 输入
uci-news-aggregator.csv <br>
GoogleNews-vectors-negative300.bin <br>
#### 输出
notCleandata.csv <br>
data.txt <br>
segmentation.txt <br>
sentences.txt <br>
labels.txt <br>
42w.model <br>
vecs.txt <br>
vecdata.csv <br>
### SvmForSTC.py
&emsp;&emsp;使用SVM算法，在整个样本中，科技类比例大致为四分之一，考虑到样本不均衡，训练时对样本增加了权重。经过测试，训练样本在8000以上时，分类模型分类稳定，泛化能力达到要求，因此采取部分冗余，训练样本为42万条中的10000条。
#### 输入
vecdata.csv <br>
#### 输出
train_model.m <br>
#### 打印输出
十次实验结果，包括TP、NP、准确率
## 输出文件说明
### notCleandata.csv
去除无关属性，数据中包含异常数据，类别为1、-1，数据为英文标题类别在前
### data.txt
赶紧数据，类别为1、-1，数据为英文标题，类别在前
### segmentation.txt
分词数据，类别在前，分词在后，逗号隔开
### sentences.txt
分词数据
### labels.txt
类别数据，与分词数据一一对应
### 42w.model
42万条标题，word embedings训练出的模型
### vecs.txt
42万条标题向量化结果，用的是谷歌新闻训练出的word2vec模型
### vecdata.csv
类别，向量
### train_model.m
训练出的svm模型
## 结果
TP: 0.7948164146868251&emsp;&emsp;
TN: 0.902931508687194&emsp;&emsp;
正确率： 0.8754<br>
TP: 0.7974512454141727&emsp;&emsp;
TN: 0.903717697861143&emsp;&emsp;
正确率： 0.8762<br>
TP: 0.7994574694826584&emsp;&emsp;
TN: 0.8990498011995417&emsp;&emsp;
正确率： 0.87335<br>
TP: 0.7975341937969562&emsp;&emsp;
TN: 0.9001958268620434&emsp;&emsp;
正确率： 0.87355<br>
TP: 0.794180407371484&emsp;&emsp;
TN: 0.9017851128326035&emsp;&emsp;
正确率： 0.87405<br>
TP: 0.7942564909520063&emsp;&emsp;
TN: 0.8999731831590239&emsp;&emsp;
正确率： 0.8731<br>
TP: 0.79375&emsp;&emsp;
TN: 0.9003360215053764&emsp;&emsp;
正确率： 0.87305<br>
TP: 0.7934272300469484&emsp;&emsp;
TN: 0.8971655024180548&emsp;&emsp;
正确率： 0.87065<br>
TP: 0.7925407925407926&emsp;&emsp;
TN: 0.9005521141933747&emsp;&emsp;
正确率： 0.87275<br>
TP: 0.797420634920635&emsp;&emsp;
TN: 0.8975267379679145&emsp;&emsp;
正确率： 0.8723<br>
## 语料
### 英文标题语料
[介绍及下载](https://www.kaggle.com/uciml/news-aggregator-dataset)<br>
[百度云下载](https://pan.baidu.com/s/1-kIwG1uCUE2ekCSsh9cdsA) &emsp;&emsp;密码：fr89<br>
整个数据集类别统计<br>
e&emsp;&emsp;152469<br>
b&emsp;&emsp;115967<br>
t&emsp;&emsp;108344<br>
m&emsp;&emsp;45639<br>
Name: CATEGORY,&emsp;&emsp;dtype: int64<br>
整个数据集二分类别统计 <br>
-1&emsp;&emsp;314075<br>
 1&emsp;&emsp;108344<br>
Name: CATEGORY,&emsp;&emsp;dtype: int64<br>
### 谷歌新闻训练好的word2vec语料
[下载](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)<br>
[百度云下载](https://pan.baidu.com/s/1_ciXDCoV_kVrAKnCJck6Rg) &emsp;&emsp;密码：37dz
## 运行
下载GoogleNews-vectors-negative300.bin放到当前目录 <br>
执行Preprocess.py <br>
执行SvmForSTC.py