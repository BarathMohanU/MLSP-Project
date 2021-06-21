import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, LSTM, ConvLSTM2D, Conv2D, Lambda
import h5py
import numpy as np
import gc
from scipy.io import savemat

def model_def():
    
    # proposed model architecture
    
    #thinking stage of eeg
    input1 = Input(shape = (19,62,63), name = 'Thinking_EEG')
    x1 = Lambda(lambda x:tf.expand_dims(x, axis=-1), name='Add_axis_1')(input1)
    x1 = ConvLSTM2D(1, (62,1), return_sequences=True, name = 'ConvLSTM_1')(x1)
    x1 = Lambda(lambda x:tf.squeeze(x, axis=[2,4]), name='Squeeze_1')(x1)
    x1 = LSTM(5, name = 'LSTM_1')(x1)
    
    #speaking stage of eeg
    input2 = Input(shape = (19,62,63), name = 'Speaking_EEG')
    x2 = Lambda(lambda x:tf.expand_dims(x, axis=-1), name='Add_axis_2')(input2)
    x2 = ConvLSTM2D(1, (62,1), return_sequences=True, name = 'ConvLSTM_2')(x2)
    x2 = Lambda(lambda x:tf.squeeze(x, axis=[2,4]), name='Squeeze_2')(x2)
    x2 = LSTM(5, name = 'LSTM_2')(x2)
    
    #audio signal
    input3 = Input(shape = (19,63), name = 'Audio')
    x3 = LSTM(5, name = 'LSTM_3')(input3)
    
    #facial data
    input4 = Input(shape = (42,6), name = 'Face')
    x4 = Lambda(lambda x:tf.expand_dims(x, axis=-1), name='Add_axis_4')(input4)
    x4 = Conv2D(1, (1,6), name = 'Conv_4')(x4)
    x4 = Lambda(lambda x:tf.squeeze(x, axis=[2,3]), name='Squeeze_4')(x4)
    x4 = Dense(5, activation='elu', name = 'Dense_face')(x4)
    
    #combined features
    x5 = Concatenate(name = 'Concat')([x1, x2, x3, x4])
    output = Dense(7, activation='softmax', name = 'Output')(x5)
    
    return Model(inputs={"input_1": input1, "input_2": input2, "input_3": input3, "input_4":input4}, outputs=output)

#read the data
data = h5py.File('data_all_subs.mat')
audio = np.transpose(np.array(data["audio"]))
thinking = np.transpose(np.array(data["thinking"]))
speaking = np.transpose(np.array(data["speaking"]))
face = np.transpose(np.array(data["face"]))
sub = np.transpose(np.array(data["sub"]))
y = np.transpose(np.array(data["y"]))
del data

#shuffle the data
p = np.random.RandomState(seed=0).permutation(thinking.shape[0])
thinking = thinking[p,:,:,:]
speaking = speaking[p,:,:,:]
audio = audio[p,:,:]
face = face[p,:,:]
sub = sub[p,:]
y = y[p,:]

testacc_rnn = np.zeros((6))

#run for first six subjects as test subjects iteratively
for test_sub in range(6):
    
    #train-test split
    train_thinking = thinking[np.squeeze(sub!=test_sub),:,:,:]
    train_speaking = speaking[np.squeeze(sub!=test_sub),:,:,:]
    train_audio = audio[np.squeeze(sub!=test_sub),:,:]
    train_face = face[np.squeeze(sub!=test_sub),:,:]
    train_y = y[np.squeeze(sub!=test_sub),:]
    
    test_thinking = thinking[np.squeeze(sub==test_sub),:,:,:]
    test_speaking = speaking[np.squeeze(sub==test_sub),:,:,:]
    test_audio = audio[np.squeeze(sub==test_sub),:,:]
    test_face = face[np.squeeze(sub==test_sub),:,:]
    test_y = y[np.squeeze(sub==test_sub),:]
    
    #call and plot the model
    model = model_def()
    tf.keras.utils.plot_model(model, 'model.png', show_shapes=True, dpi=300)
    
    #compile and fit the model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    model.fit([train_thinking, train_speaking, train_audio, train_face], tf.one_hot(np.squeeze(train_y), 7), epochs=100, batch_size=32)
    
    #evaluate and save the accuracy
    [_,testacc_rnn[test_sub]] = model.evaluate([test_thinking, test_speaking, test_audio, test_face], tf.one_hot(np.squeeze(test_y), 7))
    
    tf.keras.backend.clear_session()
    gc.collect()

#save the accuracy file
savemat('testacc_rnn.mat', {'testacc_rnn': testacc_rnn})