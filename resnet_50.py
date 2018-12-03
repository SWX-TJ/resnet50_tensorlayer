import tensorflow as tf
import tensorlayer as tl
from commonset import load_ckpt
import os
import datetime
import time
import imgaug as ia
from imgaug import augmenters as iaa
def bottleneck(inputs,insert_channel,output_channel, stride=None,is_training = True,scope='bottleblock'):
    if stride is None:
        stride = 1 if inputs.outputs.get_shape()[-1]==output_channel else 2
    with tf.variable_scope(scope):
        network = tl.layers.Conv2dLayer(inputs,
                                        act=tf.identity,
                                        shape=(1,1,inputs.outputs.get_shape()[-1],insert_channel),
                                        strides=(1,stride,stride,1),
                                        padding='SAME',
                                        name='conv_1')
        network = tl.layers.BatchNormLayer(network,act=tf.nn.relu,is_train=is_training,name='bn_1')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.identity,
                                        shape=(3,3,insert_channel,insert_channel),
                                        strides=(1,1,1,1),
                                        padding="SAME",
                                        name='conv_2')
        network = tl.layers.BatchNormLayer(network,act=tf.nn.relu,is_train=is_training,name='bn_2')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.identity,
                                        shape=(3,3,insert_channel,output_channel),
                                        strides=(1,1,1,1),
                                        padding="SAME",
                                        name='conv_3')
        network = tl.layers.BatchNormLayer(network,act=tf.identity,is_train=is_training,name='bn_3')
        if inputs.outputs.get_shape()[-1]!=output_channel:
            shortcutnet = tl.layers.Conv2dLayer(inputs,
                                                act=tf.identity,
                                                shape=(1,1,inputs.outputs.get_shape()[-1],output_channel),
                                                strides=(1,stride,stride,1),
                                                padding="SAME",
                                                name='conv_4')
            shortcutnet = tl.layers.BatchNormLayer(shortcutnet,act=tf.identity,is_train=is_training,name='bn_4')
        else:
            shortcutnet = inputs
        OutputLayer = tl.layers.ElementwiseLayer([shortcutnet,network],combine_fn=tf.add,act=tf.nn.relu,name='OutputLayer')
        return OutputLayer
def block(InputLayer,output_channel,res_num,init_stride = 2,is_training=True,scope='block'):
    with tf.variable_scope(scope):
        insert_channel = output_channel//4
        OutputLayer = bottleneck(InputLayer,insert_channel,output_channel,stride=init_stride,is_training=is_training,scope='bottlencek1')
        for i in range(1,res_num):
            OutputLayer = bottleneck(OutputLayer,insert_channel,output_channel,is_training=is_training,scope=("bottlencek%s"%(i+1)))
        return OutputLayer

def ResNet50(Input,is_training=True,reuse=False):
    with tf.variable_scope("resnet_50", reuse=reuse):
        InputLayer = tl.layers.InputLayer(Input,name="Input")
        network = tl.layers.Conv2dLayer(InputLayer,
                                        act=tf.identity,
                                        shape=(7,7,InputLayer.outputs.get_shape()[-1],64),
                                        strides=(1,2,2,1),
                                        padding='SAME',
                                        name="conv_1")
        network = tl.layers.PoolLayer(network,
                                      ksize=(1,3,3,1),
                                      strides=(1,2,2,1),
                                      padding="SAME",
                                      pool=tf.nn.max_pool,
                                      name="pool_1")
        network = block(network, 256, 3,
                             init_stride=1,
                             is_training=is_training,
                             scope="block2")
        network = block(network, 512, 4,
                        is_training=is_training,
                        scope="block3")
        network = block(network, 1024, 6,
                        is_training=is_training,
                        scope="block4")
        network = block(network, 2048, 3,
                        is_training=is_training,
                        scope="block5")
        network = tl.layers.PoolLayer(network,
                                      ksize=(1,3,3,1),
                                      strides=(1,2,2,1),
                                      pool=tf.nn.avg_pool,
                                      name="pool_2")
        network = tl.layers.FlattenLayer(network,name="flatten")
        network = tl.layers.DenseLayer(network,n_units=100,act=tf.identity,name="OutputLayer")
        return network



def distort_fn(x, is_train=False):
    #x = tl.prepro.crop(x, 24, 24, is_random=is_train)
    # if is_train:
    #     x = tl.prepro.flip_axis(x, axis=1, is_random=True)
    #    # x = tl.prepro.brightness(x, gamma=0.1, gain=1, is_random=True)
    x = x/256.0
    return x


if __name__ == '__main__':
    start_learning_rate = 0.001
    decay_rate = 0.9
    decay_steps = 1
    global_steps = tf.Variable(0, trainable=False)
    batchsize = 64
    train_epoch = 150
    project_current_dir = os.getcwd()
    curdata = datetime.datetime.now()  # current year,month,day
    model_dir = project_current_dir + "/resnet50_model/"
    X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3))
    #X_train = distort_fn(X_train)
    X_test= distort_fn(X_test)
    seq = iaa.Sequential([
        # iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0
    ])
    sess = tf.InteractiveSession()
    train_x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='train_x')
    tf.summary.image('train_image', train_x)
    test_x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='test_x')
    real_y = tf.placeholder(dtype=tf.int64, shape=[None], name='real_y')
    test_y = tf.placeholder(dtype=tf.int64, shape=[None], name='test_y')
    net = ResNet50(train_x, True, False)
    net_test = ResNet50(test_x,False, True)
    with tf.name_scope("loss"):
        l2 = 0
        for w in tl.layers.get_variables_with_name('W', train_only=True, printable=False):
            print(w.name)
            l2 += tf.contrib.layers.l2_regularizer(1e-4)(w)
        cost = tl.cost.cross_entropy(net.outputs, real_y, name="train_loss") + l2
        tf.summary.scalar('loss', cost)
        tf.summary.scalar('l2_loss', l2)
        test_loss = tl.cost.cross_entropy(net_test.outputs, test_y, name="test_loss")
    with tf.name_scope("train"):
        train_params = net.all_params
        train_op = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=start_learning_rate).minimize(cost)
        # with tf.Session(graph=g) as sess:
    tl.files.exists_or_mkdir(model_dir)
    merged = tf.summary.merge_all()
    trainwrite = tf.summary.FileWriter("resnet_logs/", sess.graph)
    sess.run(tf.global_variables_initializer())
    load_ckpt(sess=sess, save_dir=model_dir, var_list=net.all_params, printable=True)
    tensorboard_idx = 0
    saver = tf.train.Saver(max_to_keep=1)
    for epoch in range(train_epoch):
        idx = 0
        start_time = time.time()
        for batch_x, batch_y in tl.iterate.minibatches(X_train, y_train, batchsize, shuffle=True):
            batch_x = seq.augment_images(batch_x)
            sess.run(train_op, feed_dict={train_x: batch_x, real_y: batch_y})
            if idx % 10 == 0:
                print_loss = sess.run(cost, feed_dict={train_x: batch_x, real_y: batch_y})
                merge = sess.run(merged, feed_dict={train_x: batch_x, real_y: batch_y})
                trainwrite.add_summary(merge, tensorboard_idx)
                test_lossa = 0
                test_iter = 0
                for batch_test_x, batch_test_y in tl.iterate.minibatches(X_test, y_test, batchsize,shuffle=True):
                    #test_batch_x = tl.prepro.threading_data(batch_test_x, fn=distort_fn, is_train=False)
                    print_test_loss = sess.run(test_loss,feed_dict={test_x: batch_test_x, test_y: batch_test_y})
                    test_lossa = test_lossa + print_test_loss
                    test_iter = test_iter + 1
                print("idx:%d,epoch:%d,loss:%.4f,test_loss:%.4f,time:%.4f" % (
                    idx, epoch, print_loss, test_lossa/test_iter, time.time() - start_time))
                start_time = time.time()
                saver.save(sess, os.path.join(model_dir,'resnet_50model.ckpt'), global_step=idx)
                #tl.files.save_ckpt(sess=sess, mode_name='model.ckpt', save_dir=model_dir, printable=False)
            idx = idx + 1
            sess.graph.finalize()
            tensorboard_idx = tensorboard_idx + 1




