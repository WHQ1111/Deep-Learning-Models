import time
import numpy as np
import tensorflow as tf
import reader

#定义输入数据的类:PTBInput
class PTBInput(object):
    def __init__(self,config,data,name=None):
        self.batch_size = batch_size = config.batch_size #读取参数config中的batch_size到本地变量
        self.num_steps = num_steps = config.num_steps #num_steps为LSTM的展开步数
        self.epoch_size = ((len(data)//batch_size)-1)//num_steps #每个epoch内迭代的轮数(每num_steps个单词构成一个样本;每批量有batch_size个样本;所有训练集分成epoch_size个批量)
        self.input_data, self.targets = reader.ptb_producer(data,batch_size,num_steps,name=name)#获取特征数据input_data和标签数据targets;每次执行获取一个batch的数据

#定义语言模型的类:PTBModel
class PTBModel(object):
    def __init__(self,is_training,config,input_):#训练标记is_training;配置参数config;PTBInput类的实例input_
        self._input = input_
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size #LSTM的节点数(词向量维数,即隐层h的维数)
        vocab_size = config.vocab_size #词汇表大小
        
        #定义LSTM单元(多个小单元的堆叠)
        def lstm_cell():#定义默认的LSTM单元
            return tf.contrib.rnn.BasicLSTMCell(size,forget_bias=0.0,state_is_tuple=True)#size为LSTM隐藏结点数;forget_bias为遗忘门的bias;state_is_tuple=True表示返回(c_state,h_state)的二元组
        attn_cell = lstm_cell #不加dropout层
        if is_training and config.keep_prob < 1:#is_training表示在训练状态;若条件满足则在lstm_cell后面加dropout层
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(),output_keep_prob=config.keep_prob) #在lstm_cell后接一个Dropout层            
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell()]*config.num_layers,state_is_tuple=True) #将num_layers个attn_cell堆叠得到cell
        self._initial_state = cell.zero_state(batch_size,tf.float32) #设置LSTM单元的初始化状态为0
        
        #创建网络的词嵌入和输入
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding',[vocab_size,size],dtype=tf.float32) #初始化词嵌入
            inputs = tf.nn.embedding_lookup(embedding,input_.input_data) #从look up stable中获取对应输入(len(input_data)行size列)
        if is_training and config.keep_prob <1:#为训练状态再添加一层dropout
            inputs = tf.nn.dropout(inputs,config.keep_prob)
        #定义输出    
        outputs = []
        state = self._initial_state
        with tf.variable_scope('RNN'):
            for t in range(num_steps):#设置循环控制梯度传播步数
                if t > 0:#从第二步开始设置复用变量(允许同一命名域variable_scope内存在多个同一名字的变量)
                    tf.get_variable_scope().reuse_variables()
                (cell_output,state) = cell(inputs[:,t,:],state)#把整个批量中的每个样本的第t个单词和state(h_t+c_t)传入堆叠的LSTM单元中
                outputs.append(cell_output) #将cell_output(ct)添加到输出列表outputs中;outputs规格为(number_steps,batch_size,hidden_size)
        output = tf.reshape(tf.concat(outputs,1),[-1,size]) #将outputs的内容串接在一起,用reshape将其转为一个很长的一维"向量","向量"中每个分量是词向量
        softmax_w = tf.get_variable('softmax_w',[size,vocab_size],dtype=tf.float32) #softmax层的权重
        softmax_b = tf.get_variable('softmax_b',[vocab_size],dtype=tf.float32)#softmax层的偏置
        logits = tf.matmul(output,softmax_w)+softmax_b #网络最终输出(number_steps×batch_size,hidden_size)
        #定义损失函数
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],[tf.reshape(input_.targets,[-1])],[tf.ones([batch_size*num_steps],dtype=tf.float32)])#logits与targets(number_steps×.batch_size)的偏差
        self._cost = cost = tf.reduce_sum(loss)/batch_size #汇总batch的误差
        self._final_state = state #保留最终状态
        if not is_training:#如果不是训练状态,则直接返回
            return
        
        self._lr = tf.Variable(0.0,trainable=False)#定义学习率变量,其不可被训练
        tvars = tf.trainable_variables()#获取全部可训练的参数
        grads,_ = tf.clip_by_global_norm(tf.gradients(cost,tvars),config.max_grad_norm)#计算cost关于tvars的参数,并限制其最大范数;(_表示所有张量的全局范数)
        optimizer = tf.train.GradientDescentOptimizer(self._lr) #定义优化器
        self._train_op = optimizer.apply_gradients(zip(grads,tvars),global_step=tf.train.get_or_create_global_step())#创建训练操作

        self._new_lr = tf.placeholder(tf.float32,shape=[],name='new_learning_rate')#控制学习率
        self._lr_update = tf.assign(self._lr,self._new_lr)#将_new_lr的值赋给当前学习率_lr
    def assign_lr(self,session,lr_value):#在外部修改学习率(传入_new_lr这个placeholder,再执行_lr_update)
        session.run(self._lr_update,feed_dict={self._new_lr:lr_value})
    #Python中的@property装饰器将返回变量设为只读,防止修改变量引发问题
    @property
    def input(self):
        return self._input
    @property
    def initial_state(self):
        return self._initial_state
    @property
    def cost(self):
        return self._cost
    @property
    def final_state(self):
        return self._final_state
    @property
    def lr(self):
        return self._lr
    @property
    def train_op(self):
        return self._train_op


class SmallConfig(object):#小模型的设置
    init_scale = 0.1 #网络中初始scale的权重值
    learning_rate = 1.0 #学习速率的初始值
    max_grad_norm = 5 #梯度的最大范数
    num_layers = 2 #LSTM可以堆叠的层数
    num_steps = 20 #LSTM梯度反向传播的展开步数
    hidden_size = 200 #隐藏结点数
    max_epoch = 4 #初始学习速率可训练的epoch数,再次之后调整学习率
    max_max_epoch = 13 #总共可训练的epoch数
    keep_prob = 1.0 
    lr_decay = 0.5 #学习率的衰减速度
    batch_size = 20
    vocab_size = 10000 #词汇表长度
    
class ediumConfig(object):#中型模型设置
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000

class LargeConfig(object):#大型模型设置
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1/1.15
    batch_size = 20
    vocab_size = 10000

class TestConfig(object):#测试模型设置
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

def run_epoch(session,model,eval_op=None,verbose=False):#训练一个epoch
    start_time = time.time() #记录当前时间
    costs = 0.0 #初始化损失函数
    iters = 0 #初始化迭代数
    state = session.run(model.initial_state) #初始化状态
    fetches = {'cost':model.cost,'final_state':model.final_state} #创建输出结果的字典表fetches,包括cost和final_state
    if eval_op is not None:#若有评测操作,一并加入fetches
        fetches['eval_op'] = eval_op
    for step in range(model.input.epoch_size):#循环epoch_size次
        feed_dict = {}
        for i,(c,h) in enumerate(model.initial_state):#将全部LSTM单元的state加入feed_dict
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        vals = session.run(fetches,feed_dict)#对网络进行一次训练
        cost = vals['cost']
        state = vals['final_state'] #更新state
        costs += cost #将cost累加到costs
        iters += model.input.num_steps #将num_steps累加到iters
        if verbose and step%(model.input.epoch_size//10)==10:#每完成epoch的10%就进行一次结果的展示
            print('完成度:%.3f perplexity:%.3f speed:%.0f wps'%(step*1.0/model.input.epoch_size,np.exp(costs/iters),iters*model.input.batch_size/(time.time()-start_time)))#展示epoch的进度,perplexity和训练速度(单词数/秒)
    return np.exp(costs/iters)#返回平均cost的自然常数指数

raw_data = reader.ptb_raw_data('data')#直接读取解压后的数据
train_data,valid_data,test_data,_ = raw_data #获得训练数据,验证数据和测试数据
config = SmallConfig() #定义模型训练配置为SmallConfig
eval_config = SmallConfig() #测试配置与训练配置一致
eval_config.batch_size = 1 #修改测试的batch_size为1
eval_config.num_steps = 1 #修改测试的num_steps为1

with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale) #初始化器
    with tf.name_scope('Train'):#创建用来训练的模型m
        train_input = PTBInput(config=config,data=train_data,name='TrainInput')
        with tf.variable_scope('Model',reuse=None,initializer=initializer):
            m = PTBModel(is_training=True,config=config,input_=train_input)
    with tf.name_scope('Valid'):#创建用来验证的模型mvalid
        valid_input = PTBInput(config=config,data=valid_data,name='ValidInput')
        with tf.variable_scope('Model',reuse=True,initializer=initializer):
            mvalid = PTBModel(is_training=False,config=config,input_=valid_input)
    with tf.name_scope('Test'):#创建用来测试的模型mtest
        test_input = PTBInput(config=eval_config,data=test_data,name='TestInput')
        with tf.variable_scope('Model',reuse=True,initializer=initializer):
            mtest = PTBModel(is_training=False,config=eval_config,input_=test_input)
    sv = tf.train.Supervisor()#创建训练的管理器
    with sv.managed_session() as session:
        for i in range(config.max_max_epoch):#训练max_max_epoch次循环
            lr_decay = config.lr_decay ** max(i+1-config.max_epoch,0.0) #累计学习速率衰减值(前max_epoch个不进行衰减)
            m.assign_lr(session,config.learning_rate*lr_decay) #更新学习速率
            print('Epoch:%d Learning rate:%.3f'%(i+1,session.run(m.lr))) #输出学习率
            
            train_perplexity = run_epoch(session,m,eval_op=m.train_op,verbose=True)#执行一个epoch的训练;verbose控制是否展示中间进度结果
            print('Train Perplexity:%.3f'%(train_perplexity)) #输出训练集上的perplexity
            
            valid_perplexity = run_epoch(session,mvalid)#执行一个epoch的验证
            print('Valid Perplexity:%.3f'%(valid_perplexity)) #输出验证集上的perplexity
            print('-----------------------------------------------')
        test_perplexity = run_epoch(session,mtest)#求在测试集上的perplexity
        print('Test Perplexity: %.3f'%test_perplexity) #输出测试集上的perplexity
