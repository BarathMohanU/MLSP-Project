import h5py
import numpy as np
import gc
from scipy.io import savemat
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

def model_def():
    
    #feed-forward network archietcture
    
    input1 = Input(shape = (100))
    x = Dropout(0.1)(input1)
    x = Dense(40, activation='elu')(x)
    x = Dropout(0.1)(x)
    x = Dense(40, activation='elu')(x)
    x = Dropout(0.1)(x)
    output = Dense(7, activation='softmax')(x)
    
    return Model(inputs=input1, outputs=output)

#read the data
data = h5py.File('data_all_subs.mat')
audio = np.transpose(np.array(data["audio"]))
thinking = np.transpose(np.array(data["thinking"]))
speaking = np.transpose(np.array(data["speaking"]))
face = np.transpose(np.array(data["face"]))
sub = np.squeeze(np.transpose(np.array(data["sub"])))
y = np.squeeze(np.transpose(np.array(data["y"])))
del data

#flatten the data
dim = thinking.shape[0]
thinking = thinking.reshape((dim, -1))
speaking = speaking.reshape((dim, -1))
audio = audio.reshape((dim, -1))
face = face.reshape((dim, -1))

#shuffle the data
p = np.random.RandomState(seed=0).permutation(thinking.shape[0])
thinking = thinking[p,:]
speaking = speaking[p,:]
audio = audio[p,:]
face = face[p,:]
sub = sub[p]
y = y[p]

testacc_ffn = np.zeros((6))
testacc_svm = np.zeros((6))

def corr(x, y):
    
    """
    Computes the Pearson correlation between the features and the class labels
    and ranks them by the value of the correlation coefficient.
    Arguments:
        x - data in the form (samples, features)
        y - 1D vetcor of class labels
    Returns:
        pos - 1D vector of ranked features. The feature number in position 0 
              has the highest correlation and so on.
    """
    cor = np.zeros((x.shape[1]))
    for i in range(x.shape[1]):
        temp = np.corrcoef(x[:,i], y)
        cor[i] = temp[1,0]
    
    cor = np.abs(cor)
    cor = cor[~np.isnan(cor)]
    pos = np.argsort(-1 * cor)
    
    return pos
    
#run for first six subjects as test subjects iteratively
for test_sub in range(14):
    
    #train-test split
    train = np.concatenate((thinking[sub!=test_sub,:], speaking[sub!=test_sub,:], audio[sub!=test_sub,:], face[sub!=test_sub,:]), axis=-1)
    train_y = y[sub!=test_sub]
    
    test = np.concatenate((thinking[sub==test_sub,:], speaking[sub==test_sub,:], audio[sub==test_sub,:], face[sub==test_sub,:]), axis=-1)
    test_y = y[sub==test_sub]

    gc.collect()
    
    #choose top 100 features
    feats = 100
    pos = corr(train, train_y)
    train = train[:,pos[:100]]
    test = test[:, pos[:100]]
    
    #train and evaluate the svm
    kernel = 'rbf'
    svm = SVC(kernel = kernel, decision_function_shape='ovo')
    svm.fit(train, train_y)
    
    testacc_svm[test_sub] = svm.score(test, test_y)
    
    #train and evaluate the neural network
    model = model_def()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    model.fit(train, tf.one_hot(train_y,7), epochs=100, batch_size=32)
    [_,testacc_ffn[test_sub]] = model.evaluate(test, tf.one_hot(test_y,7))
    
    tf.keras.backend.clear_session()
    
#save the accuracies
savemat('testacc_svm.mat', {'testacc_svm': testacc_svm})
savemat('testacc_ffn.mat', {'testacc_ffn': testacc_ffn})