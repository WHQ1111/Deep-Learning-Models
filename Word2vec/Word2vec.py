import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib.request
import tensorflow as tf

#下载数据
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename,expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename,filename)
    statinfo = os.stat(filename)#在给定的路径filename上执行一个系统stat的调用,返回相关文件系统状态信息
    if statinfo.st_size == expected_bytes:#核对文件大小
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename +'. Can you get to it with a browser?')#文件不符异常跳出
    return filename

filename = maybe_download('text8.zip',31344016)


#解压数据,将数据转为单词列表(17005207个单词)
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()#将数据转为单词列表
    return data

#获得单词列表words
words = read_data(filename)
print('Data size',len(words))

#创建词汇表
vocabulary_size = 50000

def build_dataset(words):
    count = [['UNK',-1]]#UNK代表未知(频数不在前49999)单词的总体
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))#统计单词频数并把Top 50000加入词汇表
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0#不在词汇表中的单词个数
    for word in words:
        if word in dictionary:
            index = dictionary[word]#编号
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count#'UNK'的频数
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary
'''
频数前49999的单词与全体未知(频数不在前49999)单词的代表UNK构成词汇表,共50000个单词;dictionary是词汇表(按词汇频数从大到小赋予编号),由单词索引到序号;
count是词汇表中每个单词及其频数统计;data是把单词列表words中每个单词转换为其词汇表中的编号;reverse_dictionary与dictionary恰好相反,由序号索引单词
'''
data, count, dictionary, reverse_dictionary = build_dataset(words)

del words #删除原始单词列表,节约内存
print('Most common words (+UNK)',count[:5])#打印包括未知单词在内的频数排名前五的单词
print('Sample data',data[:10],[reverse_dictionary[i] for i in data[:10]])#查看data中前10个单词的序号

#生成Word2vec的训练样本
data_index = 0

def generate_batch(batch_size, num_skips, skip_window):#skip_window指单词最远可以联系的距离;num_skips指为每个单词生成的样本数
    global data_index #确保data_index可以在函数generate_batch中被修改
    assert batch_size % num_skips == 0 #确保每个batch包含了一个词汇对应的所有样本
    assert num_skips <= 2 * skip_window#assert函数用于如果后面的条件不满足则终止程序执行
    batch = np.ndarray(shape=(batch_size),dtype=np.int32)  #将batch设置为int32型数组
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)  #将labels设置为int32数组
    span = 2*skip_window+1#span为某个单词创建相关样本时会使用的单词数量
    buffer = collections.deque(maxlen=span)#双向队列,只保留最后插入的span个变量
    for _ in range(span):#将span个单词顺序填入buffer作为初始值,如果填满,后续数据将替换掉前面的数据
        buffer.append(data[data_index])
        data_index = (data_index+1)%len(data)
    for i in range(batch_size//num_skips):#每个batch涉及batch_size//num_skips个词汇的样本
        target = skip_window #buffer中第skip_window个变量为目标单词
        targets_to_avoid = [skip_window] #生成样本时应避免的单词列表
        for j in range(num_skips):#为每个单词生成num_skips个样本
            while target in targets_to_avoid:
                target = random.randint(0,span-1)
            targets_to_avoid.append(target)#把已经用过的语境单词加入targets_to_avoid
            batch[i*num_skips+j] = buffer[skip_window]
            labels[i*num_skips+j,0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index+1)%len(data)
    return batch,labels #batch和labels对应元素(整型数字)组成样本对

#调用generate_batch简单测试bacth和labels的效果
batch,labels = generate_batch(batch_size=8,num_skips=2,skip_window=1)
for i in range(8):
    print(batch[i],reverse_dictionary[batch[i]],'->',labels[i,0],reverse_dictionary[labels[i,0]])

#定义训练参数
batch_size = 128 #批量大小
embedding_size = 128 #词向量长度
skip_window = 1 #滑动窗口宽度
num_skips = 2
valid_size = 16 #验证单词数目
valid_window = 100 #验证单词从频数前valid_window的单词中抽取
valid_examples = np.random.choice(valid_window, valid_size, replace=False)#从0至valid_window中均匀不重复的抽取valid_size个整数(索引)
num_sampled = 64 #噪声单词数目



#定义Skip-Gram Word2Vec模型的网络结构
graph = tf.Graph()#建立计算图;与Session类似,但可以多次调用建立多个计算图
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size,1])
    valid_dataset = tf.constant(valid_examples,dtype=tf.int32)#将valid_examples转为常量张量
    with tf.device('/cpu:0'):#限定在CPU上运行
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size],-1.0,1.0))#随机生成所有单词的词向量,构成矩阵Look-up table
        embed = tf.nn.embedding_lookup(embeddings,train_inputs)#查找train_inputs(索引)对应的词向量
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0/math.sqrt(embedding_size)))#初始化NCE权重
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))#初始化NCE偏差bias
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         labels=train_labels,#训练batch的标签
                                         inputs=embed,#训练batch的输入
                                         num_sampled=num_sampled,#噪声样本的数量
                                         num_classes=vocabulary_size))#类别数目
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss) #定义优化器
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keepdims=True)) #计算嵌入向量的L2范数(1表示横向相加),keepdims=True表示保持维度
    normalized_embeddings = embeddings / norm #将嵌入向量标准化
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,valid_dataset) #查看验证单词的嵌入向量(valid_size ×embedding_size)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    '''
    计算验证单词的嵌入向量与词汇表中的其他单词的相似度(valid_size×vocabulary_size);transpose_b=True表示将ormalized_embeddings转置;
    similarity[i][j]表示第i个验证单词与第j个单词的余弦相似度
    '''
    init = tf.global_variables_initializer()
    

#开始训练
num_steps=100000

with tf.Session(graph=graph) as session:
    init.run()
    print('Initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips,skip_window)#滑动窗口宽度为1,每批量取64个词汇
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _,loss_val = session.run([optimizer,loss],feed_dict=feed_dict)
        average_loss += loss_val
        if (step+1)%2000 == 0:#每2000步算一次平均Loss,并打印输出
            average_loss/=2000
            print('Average loss at step ', step+1, ': ', average_loss)
            average_loss = 0
        if (step+1)%10000 == 0:#每10000步展示一次其他词汇中与验证单词最相关的单词
            sim = similarity.eval()#相似度矩阵
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]#验证单词
                top_k = 8 #对每个验证单词找最相关的前8个
                nearest = (-sim[i,:]).argsort()[1:top_k+1]#argsort用于从小到大排序;返回下标向量
                log_str = 'Nearest to %s:'%valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)#打印与每个验证单词最近的前8个单词
    final_embeddings = normalized_embeddings.eval()


#可视化Word2Vec效果的函数
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_with_labels(low_dim_embs,labels,filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels)
    plt.figure(figsize=(18,18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]#第i个单词的坐标
        plt.scatter(x, y)#画出第i个单词的坐标点
        plt.annotate(label,xy=(x, y),xytext=(5,2),textcoords='offset points',ha='center',va='bottom')#label表示注释内容;xytext表示注释的坐标;ha和va表示点在注释的位置
    plt.savefig(filename)

tsne = TSNE(n_components=2, init='pca', n_iter=5000)#n_components表示嵌入空间的维度;init表示嵌入的初始化,可选'pca'或'随机';n_iter表示优化的最大迭代次数
plot_only = 300 #表示只画前plot_only个点
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])#plot_only×2的矩阵,表示前plot_only个单词的坐标
labels = [reverse_dictionary[i] for i in range(plot_only)]#词汇表中前plot_only个单词作为点的标签
plot_with_labels(low_dim_embs, labels)
