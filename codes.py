import cPickle as pickle
import os
import numpy as np
import theano
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def float32(k):
    return np.cast['float32'](k)

class PlotLosses(object):
    def __init__(self, figsize=(8, 6)):
        plt.plot([],[])
    def __call__(self, nn, train_history):
        train_loss=np.array([i["train_loss"] for i in nn.train_history_])
        valid_loss=np.array([i["valid_loss"] for i in nn.train_history_])
        
        plt.gca().cla()
        plt.plot(train_loss, label="train")
        plt.plot(valid_loss, label="test")
        
        plt.legend()
        plt.draw()

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name=name
        self.start, self.stop=start, stop
        self.ls=None
    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls=np.linspace(self.start, self.stop, nn.max_epochs)
        epoch=train_history[-1]['epoch']
        new_value=float32(self.ls[epoch-1])
        getattr(nn, self.name).set_value(new_value)

def load_data(path):
    x_train = np.zeros((50000, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((50000,), dtype="uint8")

    for i in range(1, 6):
        data = unpickle(os.path.join(path, 'data_batch_' + str(i)))
        images = data['data'].reshape(10000, 3, 32, 32)
        labels = data['labels']
        x_train[(i - 1) * 10000:i * 10000, :, :, :] = images
        y_train[(i - 1) * 10000:i * 10000] = labels

    test_data = unpickle(os.path.join(path, 'test_batch'))
    x_test = test_data['data'].reshape(10000, 3, 32, 32)
    y_test = np.array(test_data['labels'])

    return x_train/float32(255), y_train, x_test/float32(255), y_test


def unpickle(file):
    f = open(file, 'rb')
    dict = pickle.load(f)
    f.close()
    return dict


net = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('dense1', layers.DenseLayer),
            ('drop1', layers.DropoutLayer),
            ('dense2', layers.DenseLayer),
            ('drop2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    input_shape=(None, 3, 32, 32),

    conv2d1_num_filters=32,
    conv2d1_filter_size=(5, 5),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    

    maxpool1_pool_size=(2, 2),

    conv2d2_num_filters=32,
    conv2d2_filter_size=(5, 5),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,

    maxpool2_pool_size=(2, 2),

    dense1_num_units=256,
    dense1_nonlinearity=lasagne.nonlinearities.rectify,
    
    drop1_p=0.5,

    dense2_num_units=256,
    
    dense2_nonlinearity=lasagne.nonlinearities.rectify,
    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),
    
    drop2_p=0.5,

    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=10,
    
    update=nesterov_momentum,
    on_epoch_finished=[AdjustVariable('update_learning_rate',
                                      start=0.03, 
                                      stop=0.0001), 
                       AdjustVariable('update_momentum',
                                      start=0.9,
                                      stop=0.999),
                      PlotLosses(figsize=(8, 6))],
    
    max_epochs=100,
    verbose=True,
    
)

net2 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('dense1', layers.DenseLayer),
            ('drop1', layers.DropoutLayer),
            ('dense2', layers.DenseLayer),
            ('drop2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    input_shape=(None, 3, 32, 32),

    conv2d1_num_filters=32,
    conv2d1_filter_size=(3, 3),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),

    

    maxpool1_pool_size=(2, 2),

    conv2d2_num_filters=32,
    conv2d2_filter_size=(4, 4),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d2_W=lasagne.init.GlorotUniform(),

    maxpool2_pool_size=(2, 2),

    dense1_num_units=256,
    dense1_nonlinearity=lasagne.nonlinearities.rectify,
    
    drop1_p=0.5,

    dense2_num_units=256,
    
    dense2_nonlinearity=lasagne.nonlinearities.rectify,
    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),
    
    drop2_p=0.5,

    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=10,
    
    update=nesterov_momentum,
    on_epoch_finished=[AdjustVariable('update_learning_rate',
                                      start=0.03, 
                                      stop=0.0001), 
                       AdjustVariable('update_momentum',
                                      start=0.9,
                                      stop=0.999),
                      PlotLosses(figsize=(8, 6))],
    
    max_epochs=100,
    verbose=True,
    
)

net3 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('conv2d3', layers.Conv2DLayer),
            ('maxpool3', layers.MaxPool2DLayer),           
            ('dense1', layers.DenseLayer),
            ('drop1', layers.DropoutLayer),
            ('dense2', layers.DenseLayer),
            ('drop2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    input_shape=(None, 3, 32, 32),

    conv2d1_num_filters=32,
    conv2d1_filter_size=(3, 3),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),

    

    maxpool1_pool_size=(2, 2),

    conv2d2_num_filters=32,
    conv2d2_filter_size=(2, 2),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d2_W=lasagne.init.GlorotUniform(),

    maxpool2_pool_size=(2, 2),

    conv2d3_num_filters=32,
    conv2d3_filter_size=(2, 2),
    conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d3_W=lasagne.init.GlorotUniform(),

    maxpool3_pool_size=(2, 2),

    dense1_num_units=256,
    dense1_nonlinearity=lasagne.nonlinearities.rectify,
    
    drop1_p=0.5,

    dense2_num_units=256,
    
    dense2_nonlinearity=lasagne.nonlinearities.rectify,
    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),
    
    drop2_p=0.5,

    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=10,
    
    update=nesterov_momentum,
    on_epoch_finished=[AdjustVariable('update_learning_rate',
                                      start=0.03, 
                                      stop=0.0001), 
                       AdjustVariable('update_momentum',
                                      start=0.9,
                                      stop=0.999),
                      PlotLosses(figsize=(8, 6))],
    
    max_epochs=100,
    verbose=True,
    
)

net4 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('conv2d3', layers.Conv2DLayer),
            ('maxpool3', layers.MaxPool2DLayer),           
            ('dense1', layers.DenseLayer),
            ('drop1', layers.DropoutLayer),
            ('dense2', layers.DenseLayer),
            ('drop2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    input_shape=(None, 3, 32, 32),

    conv2d1_num_filters=20,
    conv2d1_filter_size=(3, 3),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),

    

    maxpool1_pool_size=(2, 2),

    conv2d2_num_filters=32,
    conv2d2_filter_size=(2, 2),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d2_W=lasagne.init.GlorotUniform(),

    maxpool2_pool_size=(2, 2),

    conv2d3_num_filters=64,
    conv2d3_filter_size=(2, 2),
    conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d3_W=lasagne.init.GlorotUniform(),

    maxpool3_pool_size=(2, 2),

    dense1_num_units=500,
    dense1_nonlinearity=lasagne.nonlinearities.rectify,
    
    drop1_p=0.5,

    dense2_num_units=500,
    
    dense2_nonlinearity=lasagne.nonlinearities.rectify,
    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),
    
    drop2_p=0.5,

    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=10,
    
    update=nesterov_momentum,
    on_epoch_finished=[AdjustVariable('update_learning_rate',
                                      start=0.03, 
                                      stop=0.0001), 
                       AdjustVariable('update_momentum',
                                      start=0.9,
                                      stop=0.999),
                      PlotLosses(figsize=(8, 6))],
    
    max_epochs=100,
    verbose=True,
    
)

#Load Data
x_train, y_train, x_test, y_test = load_data(os.path.expanduser('~/data/cifar-10-batches-py'))
#For Training---
network = net.fit(x_train, y_train)


#For Showing Image
%matplotlib inline
from PIL import Image
import matplotlib.pyplot as plt
def displayImage(image):
    img=np.ascontiguousarray(image.transpose(1,2,0))
    img = Image.fromarray(img, 'RGB')
    plt.imshow(img)
    plt.show()

#For Predicting and printing Accuracy

predictions=DropoutCIFAR.predict(x_test)
print classification_report(y_test, predictions)
print accuracy_score(y_test, predictions)

#For Confusion Matrix

cm=confusion_matrix(y_test, predictions)
plt.matshow(cm)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

#For loading without normalization
def load_data2(path):
    x_train = np.zeros((50000, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((50000,), dtype="uint8")

    for i in range(1, 6):
        data = unpickle(os.path.join(path, 'data_batch_' + str(i)))
        images = data['data'].reshape(10000, 3, 32, 32)
        labels = data['labels']
        x_train[(i - 1) * 10000:i * 10000, :, :, :] = images
        y_train[(i - 1) * 10000:i * 10000] = labels

    test_data = unpickle(os.path.join(path, 'test_batch'))
    x_test = test_data['data'].reshape(10000, 3, 32, 32)
    y_test = np.array(test_data['labels'])

    return x_train, y_train, x_test, y_test



#For Saving

with open('cifar7.pickle', 'wb') as f:
    pickle.dump(cifar7,f, -1)

#For loadig  back

file_cifar=open('cifar6.pickle', 'r')

#For Visualizig layers

visualize.plot_conv_weights(cifar7.layers_['conv2d2'])