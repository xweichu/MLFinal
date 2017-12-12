import tensorflow as tf
import numpy as np
import random
import os
import datetime
import numpy
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# how to improve: add more layers/use more data to train/change the number of the fold, may change the result of validation accuracy
# change accuracy function

from random import sample
# readme we can use two ways to test the project
#1 similarity
#2 NN this is a low feature algorithm + kfold

#general Variables
batchsize=100 #batch size must be even
best_iterations =50000
hiddenunit = 80
ita = 0.7
keep_pro = 0.75

lable_set=['f-d','f-s','m-d','m-s']
#loadFeature vector
def loadInFeatureVect(dir):
    # combine two vector into one
    vectors=[]
    labels = []
    # we fisrt train in order :father-daughter, father son, mother daughter, mother-son
    # loadfeature vectors acquire all the features
    # acquire file list


    for s in os.listdir(dir):
        newdir =os.path.join(dir, s)
        vectors1 = np.load(newdir)
        vectors.extend(vectors1)
        if s.find("f-d")>=0:
            labels1 = np.zeros(len(vectors1))
            labels.extend(labels1)
        elif s.find("f-s")>=0:#positive data
            labels2 = 1*np.ones(len(vectors1))
            labels.extend(labels2)
        elif s.find("m-d")>=0:  # positive data
            labels2 = 2 * np.ones(len(vectors1))
            labels.extend(labels2)
        elif s.find("m-s")>=0:  # positive data
            labels2 = 3 * np.ones(len(vectors1))
            labels.extend(labels2)
        else:# negative data
            labels2 = -1 * np.ones(len(vectors1))
            labels.extend(labels2)


    return np.array(vectors),np.array(labels)

def loadInFeatureVect_test(dir):
    # combine two vector into one
    vectors=[]
    labels = []
    # we fisrt train in order :father-daughter, father son, mother daughter, mother-son
    # loadfeature vectors acquire all the features
    # acquire file list
    #dir='../128vectors-k2/'

    for s in os.listdir(dir):
        newdir =os.path.join(dir, s)
        vectors1 = np.load(newdir)
        vectors.extend(vectors1)
        if s.find("f-d")>=0:
            labels1 = np.zeros(len(vectors1))
            labels.extend(labels1)
        elif s.find("f-s")>=0:#positive data
            labels2 = 1*np.ones(len(vectors1))
            labels.extend(labels2)
        elif s.find("m-d")>=0:  # positive data
            labels2 = 2 * np.ones(len(vectors1))
            labels.extend(labels2)
        elif s.find("m-s")>=0:  # positive data
            labels2 = 3 * np.ones(len(vectors1))
            labels.extend(labels2)
        else:# negative data
            labels2 = -1 * np.ones(len(vectors1))
            labels.extend(labels2)

    return np.array(vectors),np.array(labels)

def feature_inputdata_2layer(vectors,labels):
    #combine parent and child feature into one vector
    comVec=[]
    comLab=[]
    length = vectors.shape[0]
    for i in range(0,length,2):
        v=[]
        v.extend(vectors[i])
        v.extend(vectors[i+1])

        comVec.append(v)
        if labels[i] == 0:
            comLab.append([1,0,0,0])
        elif labels[i] == 1:
            comLab.append([0, 1, 0, 0])
        elif labels[i] == 2:
            comLab.append([0, 0, 1, 0])
        elif labels[i] == 3:
            comLab.append([0, 0, 0, 1])
        else:
            comLab.append([0, 0, 0, 0])
    # print(comLab[0])
    # print(comLab[200])
    # print(comLab[600])
    # shuffle data

    indices = np.arange(len(comVec))
    np.random.shuffle(indices)

    comVec1=[]
    comLab1 =[]
    for i in indices:
        comVec1.append(comVec[i])
        comLab1.append( comLab[i])

    #print(comLab1)
    return np.array(comVec1),np.array(comLab1)
def feature_inputdata(vectors,labels):
    #combine parent and child feature into one vector
    comVec=[]
    comLab=[]
    length = vectors.shape[0]
    for i in range(0,length,2):
        v=[]
        v.extend(vectors[i])
        v.extend(vectors[i+1])

        comVec.append(v)
        if labels[i] == 0:
            comLab.append([1,0,0,0,0])
        elif labels[i] == 1:
            comLab.append([0, 1, 0, 0,0])
        elif labels[i] == 2:
            comLab.append([0, 0, 1, 0,0])
        elif labels[i] == 3:
            comLab.append([0, 0, 0, 1,0])
        else:
            comLab.append([0, 0, 0, 0,1])
    # print(comLab[0])
    # print(comLab[200])
    # print(comLab[600])
    # shuffle data

    indices = np.arange(len(comVec))
    np.random.shuffle(indices)

    comVec1=[]
    comLab1 =[]
    for i in indices:
        comVec1.append(comVec[i])
        comLab1.append( comLab[i])

    #print(comLab1)
    return np.array(comVec1),np.array(comLab1)
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
        batch_ys.append(labels[indxs[i]])
        batch_xs.append(vectors[indxs[i]])

    return np.array(batch_xs), np.array(batch_ys)


#ts_vectors,ts_labels =loadInFeatureVect()
#test input------- done
#tr_vectors = np.load('../128vectors/f-d.npy')


def kv_similarity():
    #use similarity between two features
    accuracy =0
    loss =0




    return accuracy,loss

# def getdataByindex(indices_tr,indices_tst, vectx,vecty):
#     tr_vectx_result =[]
#     tr_vecty_result = []
#     tst_vectx_result = []
#     tst_vecty_result = []
#     for i in indices_tr:
#         tr_vectx_result.append(vectx[i])
#         tr_vecty_result.append(vecty[i])
#     for i in indices_tst:
#         tst_vectx_result.append(vectx[i])
#         tst_vecty_result.append(vecty[i])
#
#     return tr_vectx_result,tr_vecty_result,tst_vectx_result,tst_vecty_result



def train_NN_2layer():
    x = tf.placeholder(tf.float32, [None,256]) # we input 2 vector as our input 128*2
    y_ = tf.placeholder(tf.float32, [None, 4])# we have four kinds of relationship
    #
    #w1 = tf.Variable(tf.zeros([256, hiddenunit]))  # how many items in each x,unit of hidden layer]
    w1 = tf.Variable(tf.truncated_normal([256, hiddenunit], stddev=0.1))
    b1 = tf.Variable(tf.zeros([hiddenunit]))  # how many hidden unit in each layer
    w2 = tf.Variable(tf.truncated_normal([hiddenunit, hiddenunit], stddev=0.1))
    b2 = tf.Variable(tf.zeros([hiddenunit]))  # how many hidden unit in each layer
    w3 = tf.Variable(tf.zeros([hiddenunit, 4]))  # [how many items in each x,unit of hidden layer],初始值是0
    b3 = tf.Variable(tf.zeros([4]))
    keep_prob = tf.placeholder(tf.float32)

    #layer 1
    #relu
    #logistic
    #softmax
    nnhlyer1=tf.nn.relu(tf.matmul(x, w1) + b1)
    h_fc1_drop1 = tf.nn.dropout(nnhlyer1, keep_prob) #avoid overfit
    #layer 2
    nnhlyer2= tf.nn.sigmoid(tf.matmul(h_fc1_drop1,w2) + b2)
    h_fc1_drop2 = tf.nn.dropout(nnhlyer2, keep_prob)  # avoid overfit
    # layer 3
    y_nn = tf.nn.softmax(tf.matmul(h_fc1_drop2, w3) + b3)
    # loss
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_* tf.log(y_nn),
                                                  reduction_indices=[1]))  # loss

    #cross_entropy = -tf.reduce_sum(y_*tf.log(y_nn))

    correctPred = tf.equal(tf.argmax(y_nn, 1), tf.argmax(y_, 1))
    # with tf.Session() as sess:
    #     len = sess.run(tf.shape(y_nn)[0])
    # for i in range(len):
    #     if max(y_nn)!=max(y_):# one is 0, another one is 1
    #         if correctPred[i]==1:
    #             correctPred[i]= 0

    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    # change
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=(tf.matmul(h_fc1_drop2,w3) + b3), labels=y_))

    # update
    train_step = tf.train.GradientDescentOptimizer(ita).minimize(cross_entropy)

    #init
    init = tf.global_variables_initializer()




    # 10 -fold
    # train iterations
    # save accuracy
    # trans list into array for cross validation
    # array_trvect=np.array(tr_vectors)
    # array_trlab = np.array(tr_labels)
    #
    # tst_accuracy_setallfolds =[]
    # tr_accuracy_setallfolds =[]
    # iterations_set = [40000]#np.arange(6000,18000,4000)
    # #np.random.shuffle(tr_vectors)
    # kf = KFold(n_splits=10)#, shuffle=True, # try 20 fold
    #
    # #k-fold cross validation
    # for j in iterations_set:
    #     print("ieration:%d"%j)
    #     #KFold(n_splits=10, random_state=None, shuffle=False)
    #     accuracy_set_tr = []
    #     accuracy_set_tst = []
    #     # Session
    #     # init
    #     init = tf.global_variables_initializer()
    #     sess = tf.InteractiveSession()
    #     sess.run(init)
    #
    #     for train_index, test_index in kf.split(tr_vectors):
    #         X_train, X_test = tr_vectors[train_index], tr_vectors[test_index]
    #         y_train, y_test = tr_labels[train_index], tr_labels[test_index]
    #         for i in range(j):
    #             #random
    #             batch_xs, batch_ys = getimageFeaturebynxtbatch(X_train,y_train,batchsize)
    #             sess.run([cross_entropy,train_step], feed_dict={x: batch_xs, y_: batch_ys,keep_prob:keep_pro})
    #             # if i % 20 == 0:
    #             #     # correct_prediction = tf.equal(tf.argmax(y_nn, 1), tf.argmax(y_, 1))
    #             #     # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #             #     # print ("Setp: ", i, "Accuracy: ",sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
    #             #     save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
    #             #     print("saved to %s" % save_path)
    #             #     summary = sess.run(merged, {x: batch_xs, y_: batch_ys,keep_prob: 1.0})
    #             #     writer.add_summary(summary, i)
    #         correct_prediction = tf.equal(tf.argmax(y_nn, 1), tf.argmax(y_, 1))
    #         accuracy_validate = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #         accuracy_fold_tst= sess.run(accuracy_validate, feed_dict={x: X_test, y_: y_test,keep_prob: 1.0})
    #         accuracy_fold_tr = sess.run(accuracy_validate, feed_dict={x: X_train, y_: y_train,keep_prob: 1.0})
    #         print("train accuracy:%f,test accuracy :%f"%(accuracy_fold_tr,accuracy_fold_tst))
    #         accuracy_set_tr.append(accuracy_fold_tr)
    #         accuracy_set_tst.append(accuracy_fold_tst)
    #
    #     tst_accuracy_setallfolds.append(np.mean(accuracy_set_tst))
    #     tr_accuracy_setallfolds.append(np.mean(accuracy_set_tr))
    #
    #         # save accuracy of each fold
    #     #acquire best mean accuracy select best iteration
    # best_iterations=iterations_set[tst_accuracy_setallfolds.index(max(tst_accuracy_setallfolds))]
    # print("best iterations:%d and max test accuracy:%f "%(best_iterations,max(tst_accuracy_setallfolds)))
    #
    #
    #
    #
    # #plot
    # plt.plot(iterations_set, tr_accuracy_setallfolds, '-', label=u'train accuracy')
    # plt.plot(iterations_set, tst_accuracy_setallfolds, '-', label=u'validate accuracy')
    # plt.xlabel('# of iterations')
    # plt.ylabel('prediction accuracy')
    # plt.legend()
    # plt.show()

    #run best iteration
    #best_iterations =20000
    # Session
    #best_iterations=150000
    sess = tf.InteractiveSession()
    sess.run(init)
    saver = tf.train.Saver()
    tf.summary.scalar('Loss', cross_entropy)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)

    for i in range(best_iterations):
        # random
        batch_xs, batch_ys = getimageFeaturebynxtbatch(tr_vectors, tr_labels, batchsize)
        sess.run([cross_entropy, train_step], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: keep_pro})
        if i % 20 == 0:
            save_path = saver.save(sess, "models_kin/pretrained_kinship.ckpt", global_step=i)
            print("saved to %s" % save_path)
            summary = sess.run(merged, {x: batch_xs, y_: batch_ys,keep_prob: 1.0})
            writer.add_summary(summary, i)
    correct_prediction = tf.equal(tf.argmax(y_nn, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_tr= sess.run(accuracy,feed_dict={x: tr_vectors, y_: tr_labels,keep_prob: 1.0})
    accuracy_tst= sess.run(accuracy,feed_dict={x: tst_vectors, y_: tst_labels,keep_prob: 1.0})
    accuracy_tstfs= sess.run(accuracy,feed_dict={x: tst_vectorsfs, y_: tst_labelsfs,keep_prob: 1.0})
    accuracy_tstfd = sess.run(accuracy, feed_dict={x: tst_vectorsfd, y_: tst_labelsfd, keep_prob: 1.0})
    accuracy_tstms = sess.run(accuracy, feed_dict={x: tst_vectorsms, y_: tst_labelsms, keep_prob: 1.0})
    accuracy_tstmd= sess.run(accuracy, feed_dict={x: tst_vectorsmd, y_: tst_labelsmd, keep_prob: 1.0})
    print("whole train data accuracy:%f" % accuracy_tr)
    print("whole test data accuracy:%f" % accuracy_tst)
    print("whole f-d test data accuracy:%f" % accuracy_tstfd)
    print("whole f-s test data accuracy:%f" % accuracy_tstfs)
    print("whole m-d test data accuracy:%f" % accuracy_tstmd)
    print("whole m-s test data accuracy:%f" % accuracy_tstms)
    #2 cluster filter method
    pred_y,accuracy_tst_kin= sess.run([y_nn,accuracy], feed_dict={x: kin_tst_vectors, y_: kin_tst_lables, keep_prob: 1.0})

    print("whole kin test data accuracy:%f" % accuracy_tst_kin)
    # calculated the whole accuracy
    pred_ynn=[]
    for i in range(tst_labels.shape[0]):
        indx = numpy.argwhere(labels_kin==i)#kinship original index
        if indx.shape[0]>=1:
            maxindx = numpy.argmax(pred_y[indx[0][0],])
            if maxindx==0:
                pred_ynn.append([1,0,0,0])
            elif maxindx==1:
                pred_ynn.append([0,1,0,0])
            elif maxindx ==2:
                pred_ynn.append([0,0,1,0])
            elif maxindx == 3:
                pred_ynn.append([0, 0, 0, 1])
        else:# not tst_labels[i] is not kinship
            pred_ynn.append([0,0,0,0]) # not kinship
    # calculate the accuracy
    print(pred_ynn)
    sum = 0
    pred_ynn = np.array(pred_ynn)
    for i in range(pred_ynn.shape[0]):
        if (pred_ynn[i] == tst_labels[i]).all():
            print(pred_ynn[i])
            print(tst_labels[i])
            sum += 1
    final_accuracy = sum*1.0 /tst_labels.shape[0]

    print("test accuracy with 2 layer kinV:%f"%final_accuracy)

    writer.close()

def train_NN():
    x = tf.placeholder(tf.float32, [None,256]) # we input 2 vector as our input 128*2
    y_ = tf.placeholder(tf.float32, [None, 5])# we have four kinds of relationship
    #
    #w1 = tf.Variable(tf.zeros([256, hiddenunit]))  # how many items in each x,unit of hidden layer]
    w1 = tf.Variable(tf.truncated_normal([256, hiddenunit], stddev=0.1))
    b1 = tf.Variable(tf.zeros([hiddenunit]))  # how many hidden unit in each layer
    w2 = tf.Variable(tf.truncated_normal([hiddenunit, hiddenunit], stddev=0.1))
    b2 = tf.Variable(tf.zeros([hiddenunit]))  # how many hidden unit in each layer
    w3 = tf.Variable(tf.zeros([hiddenunit, 5]))  # [how many items in each x,unit of hidden layer],初始值是0
    b3 = tf.Variable(tf.zeros([5]))
    keep_prob = tf.placeholder(tf.float32)

    #layer 1
    #relu
    #logistic
    #softmax
    nnhlyer1=tf.nn.relu(tf.matmul(x, w1) + b1)
    h_fc1_drop1 = tf.nn.dropout(nnhlyer1, keep_prob) #avoid overfit
    #layer 2
    nnhlyer2= tf.nn.sigmoid(tf.matmul(h_fc1_drop1,w2) + b2)
    h_fc1_drop2 = tf.nn.dropout(nnhlyer2, keep_prob)  # avoid overfit
    # layer 3
    y_nn = tf.nn.softmax(tf.matmul(h_fc1_drop2, w3) + b3)
    # loss
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_* tf.log(y_nn),
                                                  reduction_indices=[1]))  # loss

    #cross_entropy = -tf.reduce_sum(y_*tf.log(y_nn))

    correctPred = tf.equal(tf.argmax(y_nn, 1), tf.argmax(y_, 1))
    # with tf.Session() as sess:
    #     len = sess.run(tf.shape(y_nn)[0])
    # for i in range(len):
    #     if max(y_nn)!=max(y_):# one is 0, another one is 1
    #         if correctPred[i]==1:
    #             correctPred[i]= 0

    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    # change
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=(tf.matmul(h_fc1_drop2,w3) + b3), labels=y_))

    # update
    train_step = tf.train.GradientDescentOptimizer(ita).minimize(cross_entropy)

    #init
    init = tf.global_variables_initializer()




    # 10 -fold
    # train iterations
    # save accuracy
    # trans list into array for cross validation
    # array_trvect=np.array(tr_vectors)
    # array_trlab = np.array(tr_labels)
    #
    tst_accuracy_setallfolds =[]
    tr_accuracy_setallfolds =[]
    iterations_set = [40000]#np.arange(6000,18000,4000)
    ita_set=np.arange(0.01,0.51,0.1)
    #np.random.shuffle(tr_vectors)
    kf = KFold(n_splits=10)#, shuffle=True, # try 20 fold

    #k-fold cross validation
    #for j in iterations_set:
    j = best_iterations
    for h in ita_set:
        print("ieration:%d"%j)
        #KFold(n_splits=10, random_state=None, shuffle=False)
        accuracy_set_tr = []
        accuracy_set_tst = []
        # Session
        # init
    #update
        train_step = tf.train.GradientDescentOptimizer(ita).minimize(cross_entropy)

        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)

        for train_index, test_index in kf.split(tr_vectors):
            X_train, X_test = tr_vectors[train_index], tr_vectors[test_index]
            y_train, y_test = tr_labels[train_index], tr_labels[test_index]
            for i in range(j):
                #random
                batch_xs, batch_ys = getimageFeaturebynxtbatch(X_train,y_train,batchsize)
                sess.run([cross_entropy,train_step], feed_dict={x: batch_xs, y_: batch_ys,keep_prob:keep_pro})
                # if i % 20 == 0:
                #     # correct_prediction = tf.equal(tf.argmax(y_nn, 1), tf.argmax(y_, 1))
                #     # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                #     # print ("Setp: ", i, "Accuracy: ",sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
                #     save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
                #     print("saved to %s" % save_path)
                #     summary = sess.run(merged, {x: batch_xs, y_: batch_ys,keep_prob: 1.0})
                #     writer.add_summary(summary, i)
            correct_prediction = tf.equal(tf.argmax(y_nn, 1), tf.argmax(y_, 1))
            accuracy_validate = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            accuracy_fold_tst= sess.run(accuracy_validate, feed_dict={x: X_test, y_: y_test,keep_prob: 1.0})
            accuracy_fold_tr = sess.run(accuracy_validate, feed_dict={x: X_train, y_: y_train,keep_prob: 1.0})
            print("train accuracy:%f,test accuracy :%f"%(accuracy_fold_tr,accuracy_fold_tst))
            accuracy_set_tr.append(accuracy_fold_tr)
            accuracy_set_tst.append(accuracy_fold_tst)

        tst_accuracy_setallfolds.append(np.mean(accuracy_set_tst))
        tr_accuracy_setallfolds.append(np.mean(accuracy_set_tr))

            # save accuracy of each fold
        #acquire best mean accuracy select best iteration
    # best_iterations=iterations_set[tst_accuracy_setallfolds.index(max(tst_accuracy_setallfolds))]
    #
    # print("best iterations:%d and max test accuracy:%f "%(best_iterations,max(tst_accuracy_setallfolds)))
    #
    # #plot iteration================
    # plt.plot(iterations_set, tr_accuracy_setallfolds, '-', label=u'train accuracy')
    # plt.plot(iterations_set, tst_accuracy_setallfolds, '-', label=u'validate accuracy')
    # plt.xlabel('# of iterations')
    # plt.ylabel('prediction accuracy')
    # plt.legend()
    # plt.show()
    # end=================

    best_ita = ita_set[tst_accuracy_setallfolds.index(max(tst_accuracy_setallfolds))]

    print("best ita:%d and max test accuracy:%f "%(best_ita,max(tst_accuracy_setallfolds)))

    #plot ita==================
    plt.plot(ita_set, tr_accuracy_setallfolds, '-', label=u'train accuracy')
    plt.plot(ita_set, tst_accuracy_setallfolds, '-', label=u'validate accuracy')
    plt.xlabel('ita')
    plt.ylabel('prediction accuracy')
    plt.legend()
    plt.show()

    #run best iteration
    #best_iterations =20000
    # Session
    #best_iterations=150000
    sess = tf.InteractiveSession()
    train_step = tf.train.GradientDescentOptimizer(ita).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    tf.summary.scalar('Loss', cross_entropy)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)

    for i in range(best_iterations):
        # random
        batch_xs, batch_ys = getimageFeaturebynxtbatch(tr_vectors, tr_labels, batchsize)
        sess.run([cross_entropy, train_step], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: keep_pro})
        if i % 20 == 0:
            save_path = saver.save(sess, "models_kin/pretrained_kinship.ckpt", global_step=i)
            print("saved to %s" % save_path)
            summary = sess.run(merged, {x: batch_xs, y_: batch_ys,keep_prob: 1.0})
            writer.add_summary(summary, i)
    correct_prediction = tf.equal(tf.argmax(y_nn, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_tr= sess.run(accuracy,feed_dict={x: tr_vectors, y_: tr_labels,keep_prob: 1.0})
    accuracy_tst= sess.run(accuracy,feed_dict={x: tst_vectors, y_: tst_labels,keep_prob: 1.0})
    accuracy_tstfs= sess.run(accuracy,feed_dict={x: tst_vectorsfs, y_: tst_labelsfs,keep_prob: 1.0})
    accuracy_tstfd = sess.run(accuracy, feed_dict={x: tst_vectorsfd, y_: tst_labelsfd, keep_prob: 1.0})
    accuracy_tstms = sess.run(accuracy, feed_dict={x: tst_vectorsms, y_: tst_labelsms, keep_prob: 1.0})
    accuracy_tstmd= sess.run(accuracy, feed_dict={x: tst_vectorsmd, y_: tst_labelsmd, keep_prob: 1.0})
    print("whole train data accuracy:%f" % accuracy_tr)
    print("whole test data accuracy:%f" % accuracy_tst)
    print("whole f-d test data accuracy:%f" % accuracy_tstfd)
    print("whole f-s test data accuracy:%f" % accuracy_tstfs)
    print("whole m-d test data accuracy:%f" % accuracy_tstmd)
    print("whole m-s test data accuracy:%f" % accuracy_tstms)


    writer.close()
#print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

#test
#input data
# each vector size is 128


#train data  smilefiw
dir='../128vectors-new1/'
tr_vectors,tr_labels =loadInFeatureVect(dir)
tr_vectors,tr_labels = feature_inputdata(tr_vectors,tr_labels)
print(tr_vectors.shape[0])
#test data  kinface
dir1='../128vectors/'
tst_vectors,tst_labels =loadInFeatureVect_test(dir1)
tst_vectors,tst_labels=feature_inputdata(tst_vectors,tst_labels)

dir1='../128vectors-kinface_f-d/'
tst_vectorsfd,tst_labelsfd =loadInFeatureVect_test(dir1)
tst_vectorsfd,tst_labelsfd =feature_inputdata(tst_vectorsfd,tst_labelsfd)
dir1='../128vectors-kinface_f-s/'
tst_vectorsfs,tst_labelsfs =loadInFeatureVect_test(dir1)
tst_vectorsfs,tst_labelsfs =feature_inputdata(tst_vectorsfs,tst_labelsfs)
dir1='../128vectors-kinface_m-d/'
tst_vectorsmd,tst_labelsmd =loadInFeatureVect_test(dir1)
tst_vectorsmd,tst_labelsmd =feature_inputdata(tst_vectorsmd,tst_labelsmd)
dir1='../128vectors-kinface_m-s/'
tst_vectorsms,tst_labelsms =loadInFeatureVect_test(dir1)
tst_vectorsms,tst_labelsms =feature_inputdata(tst_vectorsms,tst_labelsms)


train_NN()







#train_NN with 2 layer(2 cluster + 4 cluster)===============================
labels_kin=[]
# dir='../128vectors-new1-pos/'
# tr_vectors,tr_labels =loadInFeatureVect(dir)
# tr_vectors,tr_labels = feature_inputdata_2layer(tr_vectors,tr_labels)
# print(tr_vectors.shape[0])
# #test data  kinface
# dir1='../128vectors/'
# tst_vectors,tst_labels =loadInFeatureVect_test(dir1)
# tst_vectors,tst_labels=feature_inputdata_2layer(tst_vectors,tst_labels)
#
# dir1='../128vectors-kinface_f-d/'
# tst_vectorsfd,tst_labelsfd =loadInFeatureVect_test(dir1)
# tst_vectorsfd,tst_labelsfd =feature_inputdata_2layer(tst_vectorsfd,tst_labelsfd)
# dir1='../128vectors-kinface_f-s/'
# tst_vectorsfs,tst_labelsfs =loadInFeatureVect_test(dir1)
# tst_vectorsfs,tst_labelsfs =feature_inputdata_2layer(tst_vectorsfs,tst_labelsfs)
# dir1='../128vectors-kinface_m-d/'
# tst_vectorsmd,tst_labelsmd =loadInFeatureVect_test(dir1)
# tst_vectorsmd,tst_labelsmd =feature_inputdata_2layer(tst_vectorsmd,tst_labelsmd)
# dir1='../128vectors-kinface_m-s/'
# tst_vectorsms,tst_labelsms =loadInFeatureVect_test(dir1)
# tst_vectorsms,tst_labelsms =feature_inputdata_2layer(tst_vectorsms,tst_labelsms)

# pred_y = np.load("y_nn_logistic.npy")
#
# j=0
# labels= np.zeros(pred_y.shape[0])# all the label
# labels_kin=[]# label with kinship
# for y in pred_y:
#     indx = y.argmax()
#
#     if indx==0:# have kinship and
#     # collect those data use NN check point to predict the collected data
#         labels_kin.append(j)
#         labels[j] = 1
#         # print(j)
#     else:
#         labels[j]=0
#     j+=1
# # predict those kinships
# kin_tst_vectors=[]
# kin_tst_lables=[]
# for j in labels_kin:
#     kin_tst_vectors.append(tst_vectors[j])
#     kin_tst_lables.append(tst_labels[j])
# labels_kin = np.array(labels_kin)

#train_NN_2layer()










# y_ = tf.placeholder(tf.float32,  4)
# y_nn = tf.placeholder(tf.float32, 4)
# y_nn=[0,0,0,0]
# print(tf.argmax(y_nn, 1))
# print(tf.equal(tf.argmax(y_nn, 1), tf.argmax(y_, 1)))
#tf.equal(tf.argmax(y_nn, 1), tf.argmax(y_, 1))



#print('len of verctor2 = %d'%len(tr_vectors))



