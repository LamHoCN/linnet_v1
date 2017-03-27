# -*- coding: utf-8 -*-
from datetime import datetime
import math
import time
import tensorflow as tf


# 先写一个函数从哦conv_op 用来创建卷积层并把本层的参数存入参数列表，input_op是输入的tensor，name是这一层的名字，kh是kernel height即卷积核的高，
# kw是kernel weight即卷积核的宽，n_out是卷积核数量即输出通道数，dh是步长的高，dw是步长的宽
# 下面使用get_shape()[-1].value获取输入input_op的通道数，比如输入图片的尺寸是224×224×3中最后的那个3
# 然后使用tf.scope(name)设置scope;我们的kernel（即卷积核参数）使用tf.get_viriable创建，其中shape就是[kh,kw,n_in,n_out]即[卷积核的高。卷积核的宽，输入通道数，输出通道数]
# 同时使用tf.contrib.layers.xavier_initializer_conv2d()做参数初始化

def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):
    print "conv:",input_op.get_shape()
    n_in = input_op.get_shape()[-1].value


    with tf.name_scope(name) as scope:
        kernel =tf.get_variable(scope+'w',shape=[kh,kw,n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d(input_op,kernel,(1,dh,dw,1),padding='SAME')
        bias_init_val = tf.constant(0.0,shape=[n_out],dtype=tf.float32)
        biases = tf.Variable(bias_init_val,trainable = True,name = 'b')
        z = tf.nn.bias_add(conv,biases)
        activation = tf.nn.relu(z,name=scope)
        p+=[kernel,biases]
        return activation
#下面定义全连接层的创建函数fc_op,一样是先获取输入input_op的通道数，然后使用tf.get_varible创建全连接层的参数，只不过参数的维度只有两个，第一个维度为输入的通道数n_in，第二个为输出的通道数n_out
#参数初始化的方法也使用xavier

def fc_op(input_op,name,n_out,p):
    print "fc:",input_op.get_shape()
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',shape=[n_in,n_out],dtype = tf.float32,initializer = tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name = 'b')
        activation = tf.nn.relu_layer(input_op,kernel,biases,name = scope)
        p+=[kernel,biases]
        return activation
#下面定义最大池化层的创建函数mpool_op。这里直接使用tf.nn.max_pool，输入即为input_op，池化尺寸为kh×kw，步长dh×dw，padding模式设为SAME
def mpool_op(input_op,name,kh,kw,dh,dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1,kh,kw,1],
                          strides=[1,dh,dw,1],
                          padding='SAME',
                          name=name)

#下面我们开始定义vgg-16的网络结构函数inference_op输入有input_op和keep_prob,这里的keep_prob 是控制dropout比率的一个placeholder。第一步先初始化参数列表p.然后创建第一段卷积网络，
#第一段结构为2个卷积层与1个池化层：这两个卷积核的大小都是3×3,同时卷积核的数量为64（输出通道数),步长为1×1，全像素扫描。
#第一个卷积层的输入input_op的尺寸为224×224×3,输出尺寸为224×224×64;而第二个卷积层的输入输出尺寸均为224×224×64.卷积层后的最大池化层则是一个标准的2×2的最大池化，将最大输出结果有尺寸变成112×112×64

def inference_op(input_op,keep_prob):
    p =[]
    con1_1 = conv_op(input_op,name="conv1_1",kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
    con1_2 = conv_op(con1_1,name="conv_1_2",kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
    pool1 =mpool_op(con1_2,name="pool1",kh=2,kw=2,dw=2,dh=2)

#第二段卷积网络和第一段非常类似，同样是两个卷积层加一个最大池化层，两个卷积层的卷积核尺寸也是3×3,步长为1×1，但是输出通道数变为128,是以前的两倍。最大池化层则和前面保持一致，因此这一卷积网络的输出尺寸变为56×56×128
    conv2_1 = conv_op(pool1,name="conv2_1",kh=3,kw=3,n_out=128,dh=1,dw=1,p=p)
    conv2_2 = conv_op(conv2_1,name="conv2_2",kh=3,kw=3,n_out=128,dh=1,dw=1,p=p)
    pool2 = mpool_op(conv2_2,name="pool2",kh=2,kw=2,dh=2,dw=2)
#接下来是第三段网络，这里有3个卷积层和一个最大池化层。3个卷积层的卷积核大小依旧是3×3,但是输出通道数为256,而最大池化层保持不变，因此这一段卷积网络的输出尺寸为28×28×256
    conv3_1 = conv_op(pool2,name="conv3_1",kh=3,kw=3,n_out=256,dh=1,dw=1,p=p)
    conv3_2 = conv_op(conv3_1,name="conv3_2",kh=3,kw=3,n_out=256,dh=1,dw=1,p=p)
    conv3_3 = conv_op(conv3_2,name="conv3_3",kh=3,kw=3,n_out=256,dh=1,dw=1,p=p)
    pool3 =mpool_op(conv3_3,name="pool3",kh=2,kw=2,dh=2,dw=2)
#第四段卷积网络也是3个卷积层加一个最大池化层，3个卷积层的卷积核大小依旧是3×3,但是输出通道数为512,而最大池化层保持不变，因此这一段卷积网络的输出尺寸为14×14×512
    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)
#最后一段卷积网络有所变化，这里卷积输出的通道数不再增加，继续维持在512.最后一段卷积网络3个卷积层加一个最大池化层 通道数不变 7*7*512
    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool5 = mpool_op(conv5_3, name="pool5", kh=2, kw=2, dh=2, dw=2)
#我们将5段卷积网络的输出结果进行扁平化，使用tf.reshape函数将每个样本化为长度为7×7×512=25088的一维向量
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5,[-1,flattened_shape],name="resh1")
#然后连接一个隐层节点为4096的全链接层，之后连接一个的人dropout层

    fc6 = fc_op(resh1,name="fc6",n_out=4096,p=p)
    fc6_drop = tf.nn.dropout(fc6,keep_prob,name="fc6_drop")
#接下来是一个和前面一样的全链接层，之后同样连接一个dropout层
    fc7= fc_op(fc6_drop,name="fc7",n_out=4096,p=p)
    fc7_drop = tf.nn.dropout(fc7,keep_prob,name="fc7_drop")
#最后连接一个有1000个输出节点的全链接层，并使用softmax进行处理得到分类输出概率。这里使用tf.argmax求出输出概率最大类别。最后fc8,softmax,predictions和参数列表p一起返回，这样vgg16就搭建完成了。
    fc8 = fc_op(fc7_drop,name="fc8",n_out=1000,p=p)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax,1)
    return predictions,softmax,fc8,p
#我们的评测函数time_tensorflow_run(),session.run()方法中引入额feed_dict，方便后面传入keep_prob来控制Dropout层的保留比率
def time_tensorflow_run(session,target,feed,info_srting):
    num_step_burn_in =10 #预热轮数
    total_duration = 0.0 #总时间
    total_duration_squared = 0.0 #总时间方差
    for i in range(num_batches+num_step_burn_in):
        start_time = time.time()
        _=session.run(target,feed_dict=feed)
        duration = time.time()-start_time
        if i >= num_step_burn_in:
            #if not i%10:
                #print "%s:step %d,duration = %f"%(datetime.now(),i-num_step_burn_in)
            total_duration+=duration
            total_duration_squared = duration*duration

    mn = total_duration/num_batches
    vr = total_duration_squared/num_batches - mn*mn
    sd = math.sqrt(vr)
    #print "%s:%s across %d steps,%f +/- %f sec/batch"%(datetime.now(),info_srting,num_batches,mn,sd)

#下面定义评测的主函数run_benchmark,我们的目标依旧是仅评测forward和backward的运算性能，并不进行实质的训练和预测。
#首先是生成224 × 224的随机图片，通过tf.random_normal函数生成标准差为0.1的正态分布随机数

def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,image_size,image_size,3],dtype=tf.float32,stddev=1e-1))
#创建keep_prob的placeholder,并调用inference_op函数构建VGG_16的网络结构，获得predictions，softmax，fc8和参数列表p
        keep_prob = tf.placeholder(tf.float32)
        predictions,softmax,fc8,p = inference_op(images,keep_prob)
#然后创建Session并初始化全局参数
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

#我们通过将keep_prob设为1.0来执行预测，并使用time_tensorflow_run 来评测forward的计算时间。再计算vgg16最后的全连接层输出的fc8的l2 loss,并使用tf.gradients求相对于这个loss的所有模型参数的梯度
#最后使用time_tensorflow_run评测backward运算时间，这里target为求解梯度的操作grad,keep_prob为0.5

        time_tensorflow_run(sess,predictions,{keep_prob:1.0},"Forward")
        objective= tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective,p)
        time_tensorflow_run(sess,grad,{keep_prob:0.5},"Forward-backward")

if __name__ == '__main__':

    batch_size = 32
    num_batches = 100
    run_benchmark()


