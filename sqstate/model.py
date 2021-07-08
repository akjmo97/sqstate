from sqstate import CURRENT_PATH
import os
import tensorflow as tf
from tensorflow.keras import backend as bak
from tensorflow.keras.layers import Input, Dense, Conv1D, AveragePooling1D, Flatten, BatchNormalization, add, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_custom_objects


def custom_activation(x):
    return bak.relu(x) * 1


h = tf.keras.losses.Huber(delta=1.0)


def custom_loss(y_true, y_pred):
        h_loss  = h(y_true,y_pred)
        return(h_loss)


def Nor_L2(x):
    x = bak.l2_normalize(x, axis=-1)
    return x


def get_model(hyperparameter_file_name: str):
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.7, beta_2=0.9)
    get_custom_objects().update({'custom_activation': Activation(custom_activation)})

    time_in =Input(shape=(2016,1),name='time_in')
    time0=BatchNormalization()(time_in)
    time0 = Conv1D(64,32, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
               bias_initializer='zeros',name='conv1-time0')(time0)
    time0=BatchNormalization()(time0)
    time0=Activation(custom_activation, name='act1-time0')(time0)

    time0X=Conv1D(96,1,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                bias_initializer='zeros',name='conv-time0x')(time0)
    time0X=BatchNormalization()(time0X)

    time1=Conv1D(32,1,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                bias_initializer='zeros',name='conv1-time1')(time0)
    time1=BatchNormalization()(time1)
    time1=Activation(custom_activation, name='act1-time1')(time1)
    time1=Conv1D(32,16,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                bias_initializer='zeros',name='conv2-time1')(time1)
    time1=BatchNormalization()(time1)
    time1=Activation(custom_activation, name='act2-time1')(time1)
    time1=Conv1D(96,1,strides=1, padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                bias_initializer='zeros',name='conv3-time1')(time1)
    time1=BatchNormalization()(time1)
    time2=add([time0X,time1])
    time2=Activation(custom_activation, name='act1-time2')(time2)

    time2=AveragePooling1D(    pool_size=2)(time2)
    time2=BatchNormalization()(time2)
    time2=Activation(custom_activation, name='POOL1-act')(time2)

    time3=Conv1D(32,1,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
             bias_initializer='zeros',name='conv1-time3')(time2)
    time3=BatchNormalization()(time3)
    time3=Activation(custom_activation, name='act1-time3')(time3)
    time3=Conv1D(32,8,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
             bias_initializer='zeros',name='conv2-time3')(time3)
    time3=BatchNormalization()(time3)
    time3=Activation(custom_activation, name='act2-time3')(time3)
    time3=Conv1D(96,1,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
             bias_initializer='zeros',name='conv3-time3')(time3)
    time3=BatchNormalization()(time3)
    time4=add([time2,time3])
    time4=Activation(custom_activation, name='act1-time4')(time4)

    time5=Conv1D(32,1,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                      bias_initializer='zeros',name='conv1-time5')(time4)
    time5=BatchNormalization()(time5)
    time5=Activation(custom_activation, name='act1-time5')(time5)
    time5=Conv1D(32,8,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                      bias_initializer='zeros',name='conv2-time5')(time5)
    time5=BatchNormalization()(time5)
    time5=Activation(custom_activation, name='act2-time5')(time5)
    time5=Conv1D(96,1,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                      bias_initializer='zeros',name='conv3-time5')(time5)
    time5=BatchNormalization()(time5)
    time6=add([time4,time5])
    time6=Activation(custom_activation, name='act1-time6')(time6)

    time6=AveragePooling1D(    pool_size=4)(time6)
    time6=BatchNormalization()(time6)
    time6=Activation(custom_activation, name='POOL2-act')(time6)

    time6X=Conv1D(128,1,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                bias_initializer='zeros',name='conv-time6x')(time6)
    time6X=BatchNormalization()(time6X)

    time7=Conv1D(64,1,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                      bias_initializer='zeros',name='conv1-time7')(time6)
    time7=BatchNormalization()(time7)
    time7=Activation(custom_activation, name='act1-time7')(time7)
    time7=Conv1D(64,4,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                      bias_initializer='zeros',name='conv2-time7')(time7)
    time7=BatchNormalization()(time7)
    time7=Activation(custom_activation, name='act2-time7')(time7)
    time7=Conv1D(128,1,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                      bias_initializer='zeros',name='conv3-time7')(time7)
    time7=BatchNormalization()(time7)
    time8=add([time6X,time7])
    time8=Activation(custom_activation, name='act1-time8')(time8)

    time9=Conv1D(64,1,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                      bias_initializer='zeros',name='conv1-time9')(time8)
    time9=BatchNormalization()(time9)
    time9=Activation(custom_activation, name='act1-time9')(time9)
    time9=Conv1D(64,4,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                      bias_initializer='zeros',name='conv2-time9')(time9)
    time9=BatchNormalization()(time9)
    time9=Activation(custom_activation, name='act2-time9')(time9)
    time9=Conv1D(128,1,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                      bias_initializer='zeros',name='conv3-time9')(time9)
    time9=BatchNormalization()(time9)
    time10=add([time8,time9])
    time10=Activation(custom_activation, name='act1-time10')(time10)

    time11=Conv1D(64,1,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                       bias_initializer='zeros',name='conv1-time11')(time10)
    time11=BatchNormalization()(time11)
    time11=Activation(custom_activation, name='act1-time11')(time11)
    time11=Conv1D(64,4,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                       bias_initializer='zeros',name='conv2-time11')(time11)
    time11=BatchNormalization()(time11)
    time11=Activation(custom_activation, name='act2-time11')(time11)
    time11=Conv1D(128,1,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                       bias_initializer='zeros',name='conv3-time11')(time11)
    time11=BatchNormalization()(time11)
    time12=add([time10,time11])
    time12=Activation(custom_activation, name='act1-time12')(time12)

    time12=AveragePooling1D(    pool_size=8)(time12)
    time12=BatchNormalization()(time12)
    time12=Activation(custom_activation, name='POOL3-act')(time12)

    time12X=Conv1D(196,1,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                bias_initializer='zeros',name='conv-time12x')(time12)
    time12X=BatchNormalization()(time12X)

    time13=Conv1D(96,1,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                       bias_initializer='zeros',name='conv1-time13')(time12)
    time13=BatchNormalization()(time13)
    time13=Activation(custom_activation, name='act1-time13')(time13)
    time13=Conv1D(96,2,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                       bias_initializer='zeros',name='conv2-time13')(time13)
    time13=BatchNormalization()(time13)
    time13=Activation(custom_activation, name='act2-time13')(time13)
    time13=Conv1D(196,1,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                       bias_initializer='zeros',name='conv3-time13')(time13)
    time13=BatchNormalization()(time13)
    time14=add([time12X,time13])
    time14=Activation(custom_activation, name='act1-time14')(time14)

    time15=Conv1D(96,1,strides=1, padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                       bias_initializer='zeros',name='conv1-time15')(time14)
    time15=BatchNormalization()(time15)
    time15=Activation(custom_activation, name='act1-time15')(time15)
    time15=Conv1D(96,2,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                       bias_initializer='zeros',name='conv2-time15')(time15)
    time15=BatchNormalization()(time15)
    time15=Activation(custom_activation, name='act2-time15')(time15)
    time15=Conv1D(196,1,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                       bias_initializer='zeros',name='conv3-time15')(time15)
    time15=BatchNormalization()(time15)
    time16=add([time14,time15])
    time16=Activation(custom_activation, name='act1-time16')(time16)

    time17=Conv1D(96,1,strides=1, padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                       bias_initializer='zeros',name='conv1-time17')(time16)
    time17=BatchNormalization()(time17)
    time17=Activation(custom_activation, name='act1-time17')(time17)
    time17=Conv1D(96,2,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                       bias_initializer='zeros',name='conv2-time17')(time17)
    time17=BatchNormalization()(time17)
    time17=Activation(custom_activation, name='act2-time17')(time17)
    time17=Conv1D(196,1,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                       bias_initializer='zeros',name='conv3-time17')(time17)
    time17=BatchNormalization()(time17)
    time18=add([time16,time17])
    time18=Activation(custom_activation, name='act1-time18')(time18)

    time19=Conv1D(96,1,strides=1, padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                       bias_initializer='zeros',name='conv1-time19')(time18)
    time19=BatchNormalization()(time19)
    time19=Activation(custom_activation, name='act1-time19')(time19)
    time19=Conv1D(96,2,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                       bias_initializer='zeros',name='conv2-time19')(time19)
    time19=BatchNormalization()(time19)
    time19=Activation(custom_activation, name='act2-time19')(time19)
    time19=Conv1D(196,1,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                       bias_initializer='zeros',name='conv3-time19')(time19)
    time19=BatchNormalization()(time19)
    time20=add([time18,time19])
    time20=Activation(custom_activation, name='act1-time20')(time20)

    time21=Conv1D(96,1,strides=1, padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                       bias_initializer='zeros',name='conv1-time21')(time20)
    time21=BatchNormalization()(time21)
    time21=Activation(custom_activation, name='act1-time21')(time21)
    time21=Conv1D(96,2,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                       bias_initializer='zeros',name='conv2-time21')(time21)
    time21=BatchNormalization()(time21)
    time21=Activation(custom_activation, name='act2-time21')(time21)
    time21=Conv1D(196,1,strides=1, padding="same",kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                       bias_initializer='zeros',name='conv3-time21')(time21)
    time21=BatchNormalization()(time21)
    time22=add([time20,time21])
    time22=Activation(custom_activation, name='act1-time22')(time22)

    time22X=AveragePooling1D(    pool_size=2)(time22)
    time22X=BatchNormalization()(time22X)
    time22X=Activation(custom_activation, name='POOL4-act')(time22X)

    cnn1d_feature=Flatten()(time22X)
    DTX_out=Dense(1225,kernel_initializer=tf.random_normal_initializer(stddev=0.001))(cnn1d_feature)
    DTX_out=Activation(custom_activation, name='act1-dtx_out')(DTX_out)
    DTX_out=Dense(1225,activation='linear',kernel_initializer=tf.random_normal_initializer(stddev=0.001))(DTX_out)

    DTX_out= tf.keras.layers.Lambda(Nor_L2, name="lambda_layer")(DTX_out)

    model = Model([time_in], [DTX_out])

    model.compile(
        optimizer=opt,
        loss=custom_loss,
        metrics=['accuracy', 'mae', 'mape', 'mse']
    )
    model.load_weights(os.path.join(CURRENT_PATH, "../model_args", hyperparameter_file_name))

    return model
