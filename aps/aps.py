import pandas as pd
import numpy as np
import tensorflow as tf

print('parsing data')

train_df = pd.read_csv('aps_failure_training_set.csv')
test_df = pd.read_csv('aps_failure_test_set.csv')

train_df.replace(to_replace='na', value=-99999, inplace=True)
test_df.replace(to_replace='na', value=-99999, inplace=True)

train_df.replace('neg', 0, inplace=True)
train_df.replace('pos', 1, inplace=True)
test_df.replace('neg', 0, inplace=True)
test_df.replace('pos', 1, inplace=True)

X_train = np.array(train_df.drop('class', axis=1))
y_train = np.array(train_df['class'])
X_test = np.array(test_df.drop('class', axis=1))
y_test = np.array(test_df['class'])



def make_one_hot(x):
	# thank you arduano
	n_values = np.max(x) + 1
	x = np.eye(n_values)[x].astype('float32')
	return x
y_train = make_one_hot(y_train)
y_test = make_one_hot(y_test)
print('parsing done')



class dnn:
    def __init__(
        self,
        n_features,
        n_classes,
        n_layers=3,
        batch_size=100,
        learning_rate=0.01,
        n_epochs=50

        ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.n_nodes_list = n_layers*[500]
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.n_classes])

    # returns rensorflow variable of desired shape
    def get_variable(self, shape):
        return tf.Variable(tf.random_normal(shape))

    def model(self, data):
        n_nodes_prev_layer = self.n_features
        prev_layer = data
        for i in range(self.n_layers):
            weights = self.get_variable([n_nodes_prev_layer, self.n_nodes_list[i]])
            biases = self.get_variable([self.n_nodes_list[i]])
            layer = tf.add(tf.matmul(prev_layer, weights), biases)
            layer = tf.nn.relu(layer)
            n_nodes_prev_layer = self.n_nodes_list[i]
            prev_layer = layer

        # weights and biases for output
        weights = self.get_variable(
            [self.n_nodes_list[len(self.n_nodes_list) - 1], self.n_classes])
        biases = self.get_variable([self.n_classes])
        layer = tf.add(tf.matmul(prev_layer, weights), biases)
        return layer

    def train(self, featureset):
        model = self.model(featureset)
        cost_function = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=model))
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost_function)


        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for i in range(1, self.n_epochs+1):
                avg_cost = 0


                sess.run(optimizer, feed_dict = {self.x: X_train, self.y: y_train})

                avg_cost += sess.run(cost_function, feed_dict={self.x: X_train, self.y: y_train})

            print('epoch', i, 'complete')
            predictions = tf.equal(tf.argmax(model, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(predictions, 'float'))
            print('accuracy:', accuracy.eval({self.x : X_test, self.y: y_test}))
        

d = dnn(170, 2)
x = d.x
print('training')
d.train(x)
print('yeet')