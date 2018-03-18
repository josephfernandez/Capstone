import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.metrics

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

tr_features = np.load(r'c:\users\joe\desktop\capstone_audio\tr_features.npy')
tr_labels = np.load(r'c:\users\joe\desktop\capstone_audio\tr_labels.npy')
ts_features = np.load(r'c:\users\joe\desktop\capstone_audio\ts_features.npy')
ts_labels = np.load(r'c:\users\joe\desktop\capstone_audio\ts_labels.npy')

tr_labels = one_hot_encode(tr_labels)
ts_labels = one_hot_encode(ts_labels)

training_epochs = 200
n_dim = tr_features.shape[1]  #shape[1] returns num of columns, 193
n_classes = 2
n_hidden_units_one = int((n_dim+n_classes)/2)
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01

X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
h_1 = tf.nn.sigmoid(tf.matmul(X,W_1) + b_1)

W = tf.Variable(tf.random_normal([n_hidden_units_one,n_classes], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_1,W) + b)

init = tf.initialize_all_variables()
#view Tensorboard data with following line
writer = tf.summary.FileWriter(r'c:\users\joe\desktop\logs',tf.Session().graph)

cost_function = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None
with tf.Session() as sess:
    sess.run(init)
    # training
    for epoch in range(training_epochs):            
        _,cost = sess.run([optimizer,cost_function],feed_dict={X:tr_features,Y:tr_labels})
        cost_history = np.append(cost_history,cost)
    
    #testing
    y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: ts_features})
    y_true = sess.run(tf.argmax(ts_labels,1))
    print("\nTest accuracy: ",round(sess.run(accuracy,feed_dict={X: ts_features,Y: ts_labels}),3))

fig = plt.figure(figsize=(10,8))
plt.plot(cost_history)
plt.xlabel('training epoch', fontsize=18)
plt.ylabel('cost', fontsize=18)
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()
print("Confusion Matrix:")
print(sklearn.metrics.confusion_matrix(y_true, y_pred, labels=[1,0]))

p,r,f,s = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average="micro")
print ("F-Score:", round(f,3))
print("P:", p)