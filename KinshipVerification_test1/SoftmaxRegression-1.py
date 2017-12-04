import tensorflow as tf
import numpy as np
import random
from random import sample
from random import randint
import datetime
# readme we can use two ways to test the project
#1 similarity
#2 NN this is a low feature algorithm + kfold
#import modules:
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import train_test_split
# from sklearn import metrics
# from sklearn.cross_validation import cross_val_score
# from sklearn import preprocessing


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
    vectors1 = np.load('../128vectors-K2/k2-f-d.npy')
    vectors.extend(vectors1)
    vectors2 = np.load('../128vectors-K2/k2-f-s.npy')
    vectors.extend(vectors2)
    vectors3 = np.load('../128vectors-K2/k2-m-d.npy')
    vectors.extend(vectors3)
    vectors4 = np.load('../128vectors-K2/k2-m-s.npy')
    vectors.extend(vectors4)
    vectors5 = np.load('../negdata/neg-1.npy')
    vectors.extend(vectors5)
    vectors6 = np.load('../negdata/neg-2.npy')
    vectors.extend(vectors6)

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
    labels5 = 4*np.ones(len(vectors5))
    labels.extend(labels5)
    labels6 = 4*np.ones(len(vectors6))
    labels.extend(labels6)

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
            comLab.append([1,0,0,0,0])
        elif labels[i] == 1:
            comLab.append([0, 1, 0, 0,0])
        elif labels[i] == 2:
            comLab.append([0, 0, 1, 0, 0])
        elif labels[i] == 3:
            comLab.append([0, 0, 0, 1, 0])
        else:
            comLab.append([0, 0, 0, 0, 1])

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
    for i in range(batchsize):
        num = randint(1, len(vectors) - 1)
        batch_ys.append(labels[num])
        batch_xs.append(vectors[num])

    return batch_xs, batch_ys

#general Variables
batchsize=80 #batch size must be even
iterations =80000
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
    tr_vectors,tr_labels =loadInFeatureVect()
    tr_vectors,tr_labels = feature_inputdata(tr_vectors,tr_labels)
    x = tf.placeholder(tf.float32, [batchsize,256], name = "feed") # we input 2 vector as our input 128*2
    y = tf.placeholder(tf.float32, [batchsize, 5], name = "labels")# we have four kinds of relationship
    #
    w1 = tf.Variable(tf.random_normal(shape = [256 , 5], stddev = 0.01), name = "weights")
    #w1 = tf.Variable(tf.zeros([256,hiddenunit])) #how many items in each x,unit of hidden layer]
    b1 = tf.Variable(tf.zeros([1, 5]), name = "bias")

    #b1 = tf.Variable(tf.zeros([hiddenunit])) #  how many hidden unit in each layer


    #to be changed
    # w1 = weight_variable([5, 5, 1, 32])
    # b1 = bias_variable([32])
    logits = tf.matmul(x, w1) + b1
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y)
    loss = tf.reduce_mean(entropy) # computes the mean over examples in the batch

    #layer 1
    #nnhlyer1=tf.nn.relu(tf.matmul(x, w1) + b1)
    #keep_prob = tf.placeholder("float")
    #h_fc1_drop = tf.nn.dropout(nnhlyer1, keep_prob) #avoid overfit
    #layer 2
    # w2 = weight_variable([5, 5, 1, 32])
    # b2 = bias_variable([32])
    # w2 = tf.Variable(tf.zeros([hiddenunit, 5]))  # [how many items in each x,unit of hidden layer],初始值是0
    # b2 = tf.Variable(tf.zeros([5]))
    # y_nn= tf.nn.softmax(tf.matmul(nnhlyer1,w2) + b2)

    # loss

    #cross_entropy = -tf.reduce_sum(y_*tf.log(y_nn))

    # update
    # = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(loss)

    prediction = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float64))

    #init
    init = tf.global_variables_initializer()

    # Session
    sess = tf.Session()
    sess.run(init)
    saver = tf.train.Saver()

    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)

    #
    for i in range(iterations):

        #random
        batch_xs, batch_ys = getimageFeaturebynxtbatch(tr_vectors,tr_labels,batchsize)
        # print(batch_ys)
        sess.run([optimizer, loss], feed_dict={x: batch_xs, y: batch_ys})
        if i % 50 == 0:
            # correct_prediction = tf.equal(tf.argmax(y_nn, 1), tf.argmax(y_, 1))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys})
            writer.add_summary(summary, i)
            print ("Setp: ", i, "Accuracy: ",sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))

        if (i % 1000 == 0 and i != 0):
            save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
            print("saved to %s" % save_path)
    writer.close()

# 评估模型
#correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

#test
#input data
# each vector size is 128

#def train_NN2():



# model = LogisticRegression()
# model = model.fit(tr_vectors, tr_labels)
# # check the accuracy on the training set
# model.score(tr_vectors, tr_labels)


#print('len of verctor2 = %d'%len(tr_vectors))
train_NN()


