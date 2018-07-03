import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf


# data prep
data = pd.read_csv('Datasets/breastcancer.csv')
del data['Unnamed: 32']

# Split the features from the target results
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

# Converting labels to binary
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

# split our data into training 70% and testing 30%
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 0)

# #Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)

train_y = train_y[:,None]
test_y = test_y[:,None]


class NeuralNetwork:
    
	def __init__(self,X, y, X_test, y_test, hidden_nodes=12, learning_rate=0.1, epochs=10000):

		self.X = tf.placeholder(tf.float32, [None,30])
		self.Y = tf.placeholder(tf.float32, [None, 1])

		# weight 1
		w1 = tf.Variable(tf.random_normal([30,10], seed=0), name='weight1')

		# bias 1
		b1 = tf.Variable(tf.random_normal([10], seed=0), name='bias1')

		layer1 = tf.nn.relu(tf.matmul(self.X,w1) + b1)

		# weight 2
		W2 = tf.Variable(tf.random_normal([10,1], seed=0), name='weight2')

		# bias 2
		b2 = tf.Variable(tf.random_normal([1], seed=0), name='bias2')

		logits = tf.matmul(layer1,W2) + b2

		hypothesis = tf.nn.relu(logits)

		self.cost_i = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=self.Y)
		self.cost = tf.reduce_mean(self.cost_i)

		self.train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.cost)

		self.prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
		self.correct_prediction = tf.equal(self.prediction, self.Y)
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, dtype=tf.float32))


		self.run_model(epochs )
        
	def run_model(self, epochs):

		with tf.Session() as sess:
		    sess.run(tf.global_variables_initializer())
		    for step in range(epochs):
		        sess.run(self.train, feed_dict={self.X: train_x, self.Y: train_y})
		        if step % 1000 == 0:
		            loss, acc = sess.run([self.cost, self.accuracy], feed_dict={self.X: train_x, self.Y: train_y})
		            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc)) 

		    train_acc = sess.run(self.accuracy, feed_dict={self.X: train_x, self.Y: train_y})
		    print("Training Accuracy =", train_acc)
		    
		    #testing
		    test_acc,test_predict,test_correct = sess.run([self.accuracy,self.prediction,self.correct_prediction], feed_dict={self.X: test_x, self.Y: test_y})
		    print("Test Accuracy =", test_acc)
	    

nn = NeuralNetwork(train_x, train_y, test_x, test_y)

