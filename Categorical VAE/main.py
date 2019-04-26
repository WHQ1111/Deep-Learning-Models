import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from tqdm import tqdm
import csv

# 定义超参数
BATCH_SIZE=100
NUM_ITERS=50000
tau0=1.0  # 初始temperature参数
np_temp=tau0
np_lr=0.001
ANNEAL_RATE=0.00003
MIN_TEMP=0.5
K=10 # 类别数
N=30 # 分布数

x = tf.placeholder(tf.float32,[None,784])# 输入图片x(shape=(batch_size,784))
slim = tf.contrib.slim

def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape,minval=0,maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y/temperature)

def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)#[batch_size,n_class]
    if hard:
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)#将y化为one-hot向量(最大值变成1其他位置变为0)
        y = tf.stop_gradient(y_hard - y) + y
    return y

# 编码器q(y|x), 784->512->256->300
def encoder(X):
    net = slim.stack(X, slim.fully_connected, [512, 256])  # (batch_size,256)
    net = slim.fully_connected(net, K*N, activation_fn=None)
    return net

# 解码器p(x|y), 300->256->512->784
def decoder(Y):
    net = slim.stack(Y,slim.fully_connected,[256,512]) #(batch_size,512)
    logits_x = slim.fully_connected(net,784,activation_fn=None)#(batch_size,784)
    p_x = tf.nn.sigmoid(logits_x)#(batch_size,784)
    return p_x

# 损失函数及训练操作
net = encoder(x) #(batch_size,N*K)
logits_y = tf.reshape(net,[-1,K])# N个独立的K-元分布的非标准化logits, shape=(batch_size*N,K)
tau = tf.Variable(5.0,name="temperature")# temperature参数
y = tf.reshape(gumbel_softmax(logits_y,tau,hard=False),[-1,N,K]) #shape=(batch_size,N,K)
p_x = decoder(slim.flatten(y))
q_y = tf.nn.softmax(logits_y)

kl_tmp = tf.reshape(q_y*(tf.log(q_y+1e-20)-tf.log(1.0/K)),[-1,N,K])
KL = tf.reduce_sum(kl_tmp,[1,2])#输出分布与均匀分布(先验)的KL散度
ML = tf.reduce_mean(tf.reduce_sum(x*tf.log(p_x)+(1-x)*tf.log(1-p_x),1))#计算交叉熵损失
Elbo= ML - KL
loss=tf.reduce_mean(-Elbo)

global_step = tf.Variable(0)
learn_rate = tf.train.exponential_decay(np_lr,global_step,100,0.96,staircase=True)
train_op=tf.train.AdamOptimizer(learn_rate).minimize(loss,global_step=global_step,var_list=slim.get_model_variables())

# 开始训练
data = input_data.read_data_sets('data', one_hot=True).train #读取数据
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in tqdm(range(NUM_ITERS)):
    np_x,np_y=data.next_batch(BATCH_SIZE)
    _, np_loss, _ = sess.run([train_op, loss, global_step], {x: np_x, tau: np_temp})
    if (i+1) % 10 == 1:
        with open('Loss.csv','a',newline='') as outp:
            writer = csv.writer(outp)
            writer.writerow([np_loss])
    if (i+1) % 1000 == 0:
        np_temp=np.maximum(tau0*np.exp(-ANNEAL_RATE*i),MIN_TEMP)
        np_lr*=0.9

#保存动态图
np_x1, _ = data.next_batch(100)#真实样本
np_x2 = sess.run(p_x,{x:np_x1})#y表示隐变量
def save_anim(data,figsize,filename):
    fig=plt.figure(figsize=(figsize[1]/10.0,figsize[0]/10.0))
    im = plt.imshow(data[0].reshape(figsize),cmap=plt.cm.gray,interpolation='none')
    plt.gca().set_axis_off()
    #fig.tight_layout()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    def updatefig(t):
        im.set_array(data[t].reshape(figsize))
        return im,
    anim = animation.FuncAnimation(fig, updatefig, frames=100, interval=30, blit=True, repeat=True)
    anim.save(filename, writer='imagemagick', fps=1, metadata=dict(artist='Me'), bitrate=1800)
    return
save_anim(np_x1,(28,28),'real.gif')
save_anim(np_x2,(28,28),'generator.gif')

# 数据可视化
M=100*N
np_y = np.zeros((M,K))
np_y[range(M),np.random.choice(K,M)] = 1
np_y = np.reshape(np_y,[100,N,K]) #从先验中采样的隐变量
np_x= sess.run(p_x,{y:np_y}) #生成样本
## 可视化生成的图像 (100,784)->(10,10,28,28)->(10,1,10,28,28)->(1,10,28,280)->(10,1,1,28,280)->(1,1,280,280)->(280,280)
np_x = np_x.reshape((10,10,28,28))
np_x = np.concatenate(np.split(np_x,10,axis=0),axis=3)#(1,10,28,280)
np_x = np.concatenate(np.split(np_x,10,axis=1),axis=2)#(1,1,280,280)
x_img = np.squeeze(np_x)#(280,280)
fig = plt.figure(figsize=(10,10))
plt.imshow(x_img,cmap='Greys_r')
plt.savefig('figures.png')
plt.close(fig)