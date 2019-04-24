import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data

if not os.path.exists('out'):
    os.makedirs('out')

W_init = tf.contrib.layers.variance_scaling_initializer()

#编码器E:X-->z
X = tf.placeholder(tf.float32,[None,784])
E_W1 = tf.Variable(W_init([784,400]))
E_b1 = tf.Variable(tf.zeros([400]))
E_W2 = tf.Variable(W_init([400,400]))
E_b2 = tf.Variable(tf.zeros([400]))
E_W3 = tf.Variable(W_init([400,200]))
E_b3 = tf.Variable(tf.zeros([200]))
#解码器D:z-->X
Z = tf.placeholder(tf.float32,[None,100])
D_W1 = tf.Variable(W_init([100,400]))
D_b1 = tf.Variable(tf.zeros([400]))
D_W2 = tf.Variable(W_init([400,400]))
D_b2 = tf.Variable(tf.zeros([400]))
D_W3 = tf.Variable(W_init([400,784]))
D_b3 = tf.Variable(tf.zeros([784]))
theta = [E_W1,E_W2,E_W3,E_b1,E_b2,E_b3,D_W1,D_W2,D_W3,D_b1,D_b2,D_b3]

kep = tf.placeholder(tf.float32)

def plot(samples):
    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(10,10)
    gs.update(wspace=0.05,hspace=0.05)
    for i,sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        plt.imshow(sample.reshape(28,28),cmap='Greys_r')
    return fig

# Gaussian MLP as encoder
def gaussian_MLP_encoder(x,keep_prob):
    # 1nd hidden layer
    h1 = tf.nn.selu(tf.matmul(x,E_W1) + E_b1)
    h1 = tf.nn.dropout(h1,keep_prob)
    # 2nd hidden layer
    h2 = tf.nn.tanh(tf.matmul(h1,E_W2) + E_b2)
    h2 = tf.nn.dropout(h2,keep_prob)
    # output layer
    gaussian_params = tf.matmul(h2, E_W3) + E_b3
    mean = gaussian_params[:,:100]
    stddev = 1e-6 + tf.nn.softplus(gaussian_params[:,100:])
    return mean, stddev

# Bernoulli MLP as decoder
def bernoulli_MLP_decoder(z,keep_prob):
    # 1nd hidden layer
    h1 = tf.nn.tanh(tf.matmul(z,D_W1) + D_b1)
    h1 = tf.nn.dropout(h1,keep_prob)
    # 2nd hidden layer
    h2 = tf.nn.selu(tf.matmul(h1,D_W2) + D_b2)
    h2 = tf.nn.dropout(h2,keep_prob)
    # output layer
    out = tf.sigmoid(tf.matmul(h2, D_W3) + D_b3)
    return out
#encoding
mu,sigma = gaussian_MLP_encoder(X,kep)
#sample z
z = mu + sigma * tf.random_normal(tf.shape(mu),0,1,dtype=tf.float32)
#decoding
y = bernoulli_MLP_decoder(z,kep)
y = tf.clip_by_value(y,1e-8,1-1e-8)
marginal_likelihood = tf.reduce_sum(X*tf.log(y)+(1-X)*tf.log(1-y),1)
KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma)-tf.log(1e-8+tf.square(sigma))-1,1)
marginal_likelihood = tf.reduce_mean(marginal_likelihood)
KL_divergence = tf.reduce_mean(KL_divergence)
ELBO = marginal_likelihood - KL_divergence
loss = -ELBO
solver = tf.train.AdamOptimizer().minimize(loss,var_list=theta)

mb_size = 256
Z_dim = 100
mnist = input_data.read_data_sets('data',one_hot=True)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

i = 0
for it in range(40000):
    X_mb,_ = mnist.train.next_batch(mb_size)
    loss_itr,KL_itr,_ = sess.run([loss,KL_divergence,solver],feed_dict={X:X_mb,kep:0.75})
    if it % 1000 == 0:
        print('Itr {}:'.format(it))
        print('loss :{:.4}'.format(loss_itr))
        print('KL :{:.4}'.format(KL_itr))
        print()
    if it % 1000 == 0:
        samples = sess.run(y,feed_dict={X:X_mb[:100,:],kep:1.0})
        fig = plot(samples)
        i += 1
        plt.savefig('out/{}.png'.format(str(i).zfill(2)))
        plt.close(fig)
