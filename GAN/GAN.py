import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data

W_init = tf.contrib.layers.variance_scaling_initializer()

if not os.path.exists('out'):
    os.makedirs('out')

#生成变量监控信息并定义生成监控信息日志的操作
def variable_summaries(var,name):
    #将生成监控信息的操作放在同一命名空间下
    with tf.name_scope('summaries'):
        #通过tf.histogram_summary函数记录张量中元素的取值分布
        tf.summary.histogram(name,var)
        #计算变量的平均值,并定义生成平均值信息日志的操作
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/'+name,mean)
        #计算变量的标准差,并定义生成其日志的操作
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev/'+name,stddev)


def plot(samples):
    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(10,10)
    gs.update(wspace=0.05,hspace=0.05)
    for i,sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        plt.imshow(sample.reshape(28,28),cmap='Greys_r')
    return fig

def sample_Z(m,n):
    return np.random.normal(0,1.0,size=[m,n])

with tf.name_scope('Input'):
    X = tf.placeholder(tf.float32,[None,784])
    Z = tf.placeholder(tf.float32,[None,100])
    
with tf.name_scope('input_shape'):
    image_input = tf.reshape(X,[-1,28,28,1])
    tf.summary.image('input',image_input,10)

with tf.name_scope('D_var'):
    D_W1 = tf.Variable(W_init([784,128]))
    variable_summaries(D_W1,'D_var/D_W1')
    
    D_b1 = tf.Variable(tf.zeros([128]))
    variable_summaries(D_b1,'D_var/D_b1')
    
    D_W2 = tf.Variable(W_init([128,1]))
    variable_summaries(D_W2,'D_var/D_W2')
    
    D_b2 = tf.Variable(tf.zeros([1]))
    variable_summaries(D_b2,'D_var/D_b2')
    
    theta_D = [D_W1,D_W2,D_b1,D_b2]

with tf.name_scope('G_var'):
    G_W1 = tf.Variable(W_init([100,128]))
    variable_summaries(G_W1,'G_var/G_W1')
    
    G_b1 = tf.Variable(tf.zeros([128]))
    variable_summaries(G_b1,'G_var/G_b1')
    
    G_W2 = tf.Variable(W_init([128,784]))
    variable_summaries(G_W2,'G_var/G_W2')
    
    G_b2 = tf.Variable(tf.zeros([784]))
    variable_summaries(G_b2,'G_var/G_b2')
    theta_G = [G_W1,G_W2,G_b1,G_b2]


def generator(z):
    with tf.name_scope('Generator'):
        G_h1 = tf.nn.selu(tf.matmul(z,G_W1)+G_b1)
        G_out = tf.nn.sigmoid(tf.matmul(G_h1,G_W2)+G_b2)
        tf.summary.histogram('Generator/G_out',G_out)
    return G_out

def discriminator(x):
    with tf.name_scope('Discriminator'):
        D_h1 = tf.nn.selu(tf.matmul(x,D_W1)+D_b1)
        D_logit = tf.matmul(D_h1,D_W2)+D_b2
        D_out = tf.nn.sigmoid(D_logit)
        tf.summary.histogram('Discriminator/D_out',D_out)
    return D_logit,D_out

G_sample = generator(Z)
D_logit_fake,D_fake = discriminator(G_sample)
D_logit_real,D_real = discriminator(X)

with tf.name_scope('Loss'):
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,labels=tf.zeros_like(D_logit_fake)))
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real,labels=tf.ones_like(D_logit_real)))
    D_loss = D_loss_fake + D_loss_real
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,labels=tf.ones_like(D_logit_fake)))
    tf.summary.scalar('D_loss',D_loss)
    tf.summary.scalar('G_loss',G_loss)
    
with tf.name_scope('Train'):
    G_solver = tf.train.AdamOptimizer().minimize(G_loss,var_list=theta_G)
    D_solver = tf.train.AdamOptimizer().minimize(D_loss,var_list=theta_D)

mb_size = 256
Z_dim = 100
SUMMARY_DIR = 'Log'
mnist = input_data.read_data_sets('data',one_hot=True)

merged = tf.summary.merge_all()
with tf.Session() as sess:
    writer = tf.summary.FileWriter(SUMMARY_DIR,sess.graph)
    sess.run(tf.global_variables_initializer())
    for it in range(50000):
        X_mb,_ = mnist.train.next_batch(mb_size)
        summary,_,D_loss_itr = sess.run([merged,D_solver,D_loss],feed_dict={X:X_mb,Z:sample_Z(mb_size,100)})
        _,G_loss_itr = sess.run([G_solver,G_loss],feed_dict={Z:sample_Z(mb_size,100)})
        writer.add_summary(summary,it)
        if it % 1000 == 0:
            print('Itr {}:'.format(it))
            print('G_loss = {:.4}'.format(G_loss_itr))
            print('D_loss = {:.4}'.format(D_loss_itr))
            samples = sess.run(G_sample,feed_dict={Z:sample_Z(100,Z_dim)})
            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(it).zfill(2)))
            plt.close(fig)
writer.close()
