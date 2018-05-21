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


def saveModel (sess):
    saver = tf.train.Saver()
    saver.save(sess, 'C:/Users/ridge/Desktop/machinelearning/aps/model/model.ckpt')

def restoreModel (sess):
    saver = tf.train.Saver()
    saver.restore(sess, 'C:/Users/ridge/Desktop/machinelearning/aps/model/model.ckpt')
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('scrap15', sess.graph)




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



        with tf.Session() as sess:
            # saver = tf.train.Saver()

            sess.run(tf.global_variables_initializer()) # comment this line if restoring model
            # restoreModel(sess)


            for i in range(1, self.n_epochs+1):
                avg_cost = 0
                sess.run(optimizer, feed_dict = {self.x: X_train, self.y: y_train})
                avg_cost += sess.run(cost_function, feed_dict={self.x: X_train, self.y: y_train})

                print('epoch', i, 'of' , self.n_epochs, 'complete')
            predictions = tf.equal(tf.argmax(model, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(predictions, 'float'))
            if i % 20 == 0:
                saveModel(sess)


            # data = [59816,-9999,1010,936,0,0,0,0,0,0,123922,984314,1680050,1135268,92606,14038,1772828,0,0,0,1116,2372,3546760,3053176,652616,423374,0,0,7274,0,1622,432,0,0,0,0,0,6388,1091104,2930694,2012,0,3526,904,1426,223226,2663348,1137664,104,0,0,0,1283806,928212,345132,265930,194770,158262,219942,620264,13880,0,4201350,98,238,880,16,1772828,51468,331744,316130,176000,208420,159380,-9999,-999,-99999,-9999,-9999,100120,59816.46,4201350,4201350,4203050,29967.0,26214,51894,562680,4030198,1209600,114684,0,144,0,3387773.76,38633.28,599624.64,0,0,0,0,14308,475410,1109740,1528024,837114,58942,6220,440,1278,1292,4201350,4,6846,810,70090,345884,191284,2454600,926846,33558,280,0,1516,1398,2050280,64066,0,674,0,46,3413978,2924,414,0,0,60,38710,0,0,0,0,27740,33354,6330,0,0,133542,21290,2718360,435370,0,0,0,0,1179900,1541.32,1678,659550,691580,540820,243270,483302,485332,431376,210074,281662,3232,0,0]
            # pred = sess.run(model, feed_dict={self.x: np.expand_dims(data, 0)})

            print('accuracy:', accuracy.eval({self.x : X_test, self.y: y_test}))
            # saver.save(sess, 'C:/Users/ridge/Desktop/machinelearning/aps/model/model.ckpt')
            

   

        

d = dnn(170, 2, n_epochs=50)
x = d.x
print('training')
d.train(x)
print('done')