import tensorflow as tf
import numpy as np
import random

from random import sample
# readme we can use two ways to test the project
#1 similarity
#2 NN this is a low feature algorithm + kfold


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


#loadFeature vector
def loadInFeatureVect():
    # combine two vector into one
    vectors=[]
    labels = []
    # we fisrt train in order :father-daughter, father son, mother daughter, mother-son
    # loadfeature vectors acquire all the features
    vectors1 = np.load('../128vectors/f-d.npy')
    vectors.extend(vectors1)
    vectors2 = np.load('../128vectors/f-s.npy')
    vectors.extend(vectors2)
    vectors3 = np.load('../128vectors/m-d.npy')
    vectors.extend(vectors3)
    vectors4 = np.load('../128vectors/m-s.npy')
    vectors.extend(vectors4)

    #labels collect all the labels in order f-d f-s m-d m-s
    #labels in different cluster is 0 1 2 3

    labels1 = np.zeros(len(vectors1))
    labels.extend(labels1)
    labels2 = np.ones(len(vectors2))
    labels.extend(labels2)
    labels3 = 2*np.ones(len(vectors3))
    labels.extend(labels3)
    labels4 = 3*np.ones(len(vectors4))
    labels.extend(labels4)

    return vectors,labels

def feature_inputdata(vectors,labels):
    #combine parent and child feature into one vector
    comVec=[]
    comLab=[]
    length = len(vectors)
    for i in range(0,length,2):
        v=[]
        v.extend(vectors[i])
        v.extend(vectors[i+1])

        comVec.append(v)
        if labels[i]==0:
            comLab.append([1,0,0,0])
        elif labels[i] == 1:
            comLab.append([0, 1, 0, 0])
        elif labels[i] == 2:
            comLab.append([0, 0, 1, 0])
        elif labels[i] == 3:
            comLab.append([0, 0, 0, 1])
        else:
            comLab.append([0, 0, 0, 0])

    return comVec,comLab

# to be changed
def getimageFeaturebynxtbatch(vectors,labels,batchsize):
    # acquire corrsponding batch vectors
    batch_ys = []
    batch_xs = []
   # batch_xs = np.zeros([batchsize,vectors.shape(0)])
   #  for i in range(batchsize):
   #      num = randint(1, len(vectors) - 1)
   #      batch_ys.append(labels[num])
   #      batch_xs.applend(vectors[num])

    indxs = random.sample(range(0,len(vectors)),batchsize)
    for i in range(len(indxs)):
        batch_ys.append(labels[i])
        batch_xs.append(vectors[i])

    return batch_xs, batch_ys

#general Variables
batchsize=20 #batch size must be even
iterations =1000
hiddenunit = 256



#ts_vectors,ts_labels =loadInFeatureVect()
#test input------- done
#tr_vectors = np.load('../128vectors/f-d.npy')


def kv_similarity():
    #use similarity between two features
    accuracy =0
    loss =0




    return accuracy,loss

def train_NN():
    x = tf.placeholder(tf.float32, [None,256]) # we input 2 vector as our input 128*2
    y_ = tf.placeholder(tf.float32, [batchsize, 4])# we have four kinds of relationship
    #
    w1 = tf.Variable(tf.zeros([256,hiddenunit])) #how many items in each x,unit of hidden layer]
    b1 = tf.Variable(tf.zeros([hiddenunit])) #  how many hidden unit in each layer
    #to be changed
    # w1 = weight_variable([5, 5, 1, 32])
    # b1 = bias_variable([32])

    #layer 1
    nnhlyer1=tf.nn.relu(tf.matmul(x, w1) + b1)
    #keep_prob = tf.placeholder("float")
    #h_fc1_drop = tf.nn.dropout(nnhlyer1, keep_prob) #avoid overfit
    #layer 2
    # w2 = weight_variable([5, 5, 1, 32])
    # b2 = bias_variable([32])
    w2 = tf.Variable(tf.zeros([hiddenunit, 4]))  # [how many items in each x,unit of hidden layer],初始值是0
    b2 = tf.Variable(tf.zeros([4]))
    y_nn= tf.nn.softmax(tf.matmul(nnhlyer1,w2) + b2)

    # loss

    cross_entropy = -tf.reduce_sum(y_*tf.log(y_nn))

    # update
    # = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    #init
    init = tf.global_variables_initializer()

    # Session
    sess = tf.Session()
    sess.run(init)

    #
    for i in range(iterations):
        #random
        batch_xs, batch_ys = getimageFeaturebynxtbatch(tr_vectors,tr_labels,batchsize)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 50 == 0:
            correct_prediction = tf.equal(tf.argmax(y_nn, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print ("Setp: ", i, "Accuracy: ",sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))

# 评估模型
#correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

#test
#input data
# each vector size is 128



tr_vectors,tr_labels =loadInFeatureVect()
tr_vectors,tr_labels = feature_inputdata(tr_vectors,tr_labels)


#print('len of verctor2 = %d'%len(tr_vectors))
train_NN()


