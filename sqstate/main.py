import os

import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras import backend as bak
from tensorflow.keras.layers import Dropout, Input, Dense, Conv1D, AveragePooling1D, Flatten, BatchNormalization, add, \
    Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_custom_objects

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))


def custom_activation(x):
    return bak.relu(x) * 1


def custom_loss(y_true, y_pred):
    mse = bak.mean(bak.square(y_true - y_pred), axis=-1)
    sum_constraint = 0.002 * bak.abs(1 * (bak.sum(bak.square(y_pred), axis=-1) - 1))
    return mse + sum_constraint


def get_model():
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.7, beta_2=0.9)
    get_custom_objects().update({'custom_activation': Activation(custom_activation)})

    homodyne_time_in = Input(shape=(4032, 1), name='homodyne_time')
    homodyne_time0 = Conv1D(128, 4, padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                            bias_initializer='zeros')(homodyne_time_in)
    homodyne_time0 = BatchNormalization()(homodyne_time0)
    homodyne_time0 = Activation(custom_activation, name='SpecialActivation1')(homodyne_time0)
    homodyne_time1 = Conv1D(64, 1, strides=1, padding="same",
                            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                            bias_initializer='zeros')(homodyne_time0)
    homodyne_time1 = BatchNormalization()(homodyne_time1)
    homodyne_time1 = Activation(custom_activation, name='SpecialActivation2')(homodyne_time1)
    homodyne_time1 = Conv1D(64, 4, strides=1, padding="same",
                            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                            bias_initializer='zeros')(homodyne_time1)
    homodyne_time1 = BatchNormalization()(homodyne_time1)
    homodyne_time1 = Activation(custom_activation, name='SpecialActivation3')(homodyne_time1)
    homodyne_time1 = Conv1D(128, 1, strides=1, padding="same",
                            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                            bias_initializer='zeros')(homodyne_time1)
    homodyne_time1 = BatchNormalization()(homodyne_time1)
    homodyne_time1 = Activation(custom_activation, name='SpecialActivation4')(homodyne_time1)
    homodyne_time2 = add([homodyne_time0, homodyne_time1])

    homodyne_time2 = AveragePooling1D(pool_size=2)(homodyne_time2)

    homodyne_time3 = Conv1D(64, 1, strides=1, padding="same",
                            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                            bias_initializer='zeros')(homodyne_time2)
    homodyne_time3 = BatchNormalization()(homodyne_time3)
    homodyne_time3 = Activation(custom_activation, name='SpecialActivation5')(homodyne_time3)
    homodyne_time3 = Conv1D(64, 4, strides=1, padding="same",
                            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                            bias_initializer='zeros')(homodyne_time3)
    homodyne_time3 = BatchNormalization()(homodyne_time3)
    homodyne_time3 = Activation(custom_activation, name='SpecialActivation6')(homodyne_time3)
    homodyne_time3 = Conv1D(128, 1, strides=1, padding="same",
                            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                            bias_initializer='zeros')(homodyne_time3)
    homodyne_time3 = BatchNormalization()(homodyne_time3)
    homodyne_time3 = Activation(custom_activation, name='SpecialActivation7')(homodyne_time3)
    homodyne_time4 = add([homodyne_time2, homodyne_time3])
    homodyne_time5 = Conv1D(64, 1, strides=1, padding="same",
                            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                            bias_initializer='zeros')(homodyne_time4)
    homodyne_time5 = BatchNormalization()(homodyne_time5)
    homodyne_time5 = Activation(custom_activation, name='SpecialActivation8')(homodyne_time5)
    homodyne_time5 = Conv1D(64, 4, strides=1, padding="same",
                            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                            bias_initializer='zeros')(homodyne_time5)
    homodyne_time5 = BatchNormalization()(homodyne_time5)
    homodyne_time5 = Activation(custom_activation, name='SpecialActivation9')(homodyne_time5)
    homodyne_time5 = Conv1D(128, 1, strides=1, padding="same",
                            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                            bias_initializer='zeros')(homodyne_time5)
    homodyne_time5 = BatchNormalization()(homodyne_time5)
    homodyne_time5 = Activation(custom_activation, name='SpecialActivation10')(homodyne_time5)
    homodyne_time6 = add([homodyne_time4, homodyne_time5])

    homodyne_time6 = AveragePooling1D(pool_size=8)(homodyne_time6)

    homodyne_time7 = Conv1D(64, 1, strides=1, padding="same",
                            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                            bias_initializer='zeros')(homodyne_time6)
    homodyne_time7 = BatchNormalization()(homodyne_time7)
    homodyne_time7 = Activation(custom_activation, name='SpecialActivation11')(homodyne_time7)
    homodyne_time7 = Conv1D(64, 4, strides=1, padding="same",
                            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                            bias_initializer='zeros')(homodyne_time7)
    homodyne_time7 = BatchNormalization()(homodyne_time7)
    homodyne_time7 = Activation(custom_activation, name='SpecialActivation12')(homodyne_time7)
    homodyne_time7 = Conv1D(128, 1, strides=1, padding="same",
                            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                            bias_initializer='zeros')(homodyne_time7)
    homodyne_time7 = BatchNormalization()(homodyne_time7)
    homodyne_time7 = Activation(custom_activation, name='SpecialActivation13')(homodyne_time7)
    homodyne_time8 = add([homodyne_time6, homodyne_time7])
    homodyne_time9 = Conv1D(64, 1, strides=1, padding="same",
                            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                            bias_initializer='zeros')(homodyne_time8)
    homodyne_time9 = BatchNormalization()(homodyne_time9)
    homodyne_time9 = Activation(custom_activation, name='SpecialActivation14')(homodyne_time9)
    homodyne_time9 = Conv1D(64, 4, strides=1, padding="same",
                            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                            bias_initializer='zeros')(homodyne_time9)
    homodyne_time9 = BatchNormalization()(homodyne_time9)
    homodyne_time9 = Activation(custom_activation, name='SpecialActivation15')(homodyne_time9)
    homodyne_time9 = Conv1D(128, 1, strides=1, padding="same",
                            kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                            bias_initializer='zeros')(homodyne_time9)
    homodyne_time9 = BatchNormalization()(homodyne_time9)
    homodyne_time9 = Activation(custom_activation, name='SpecialActivation16')(homodyne_time9)
    homodyne_time10 = add([homodyne_time8, homodyne_time9])
    homodyne_time11 = Conv1D(64, 1, strides=1, padding="same",
                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                             bias_initializer='zeros')(homodyne_time10)
    homodyne_time11 = BatchNormalization()(homodyne_time11)
    homodyne_time11 = Activation(custom_activation, name='SpecialActivation17')(homodyne_time11)
    homodyne_time11 = Conv1D(64, 4, strides=1, padding="same",
                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                             bias_initializer='zeros')(homodyne_time11)
    homodyne_time11 = BatchNormalization()(homodyne_time11)
    homodyne_time11 = Activation(custom_activation, name='SpecialActivation18')(homodyne_time11)
    homodyne_time11 = Conv1D(128, 1, strides=1, padding="same",
                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                             bias_initializer='zeros')(homodyne_time11)
    homodyne_time11 = BatchNormalization()(homodyne_time11)
    homodyne_time11 = Activation(custom_activation, name='SpecialActivation19')(homodyne_time11)
    homodyne_time12 = add([homodyne_time10, homodyne_time11])

    homodyne_time12 = AveragePooling1D(pool_size=8)(homodyne_time12)

    homodyne_time13 = Conv1D(64, 1, strides=1, padding="same",
                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                             bias_initializer='zeros')(homodyne_time12)
    homodyne_time13 = BatchNormalization()(homodyne_time13)
    homodyne_time13 = Activation(custom_activation, name='SpecialActivation20')(homodyne_time13)
    homodyne_time13 = Conv1D(64, 4, strides=1, padding="same",
                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                             bias_initializer='zeros')(homodyne_time13)
    homodyne_time13 = BatchNormalization()(homodyne_time13)
    homodyne_time13 = Activation(custom_activation, name='SpecialActivation21')(homodyne_time13)
    homodyne_time13 = Conv1D(128, 1, strides=1, padding="same",
                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                             bias_initializer='zeros')(homodyne_time13)
    homodyne_time13 = BatchNormalization()(homodyne_time13)
    homodyne_time13 = Activation(custom_activation, name='SpecialActivation22')(homodyne_time13)
    homodyne_time14 = add([homodyne_time12, homodyne_time13])
    homodyne_time15 = Conv1D(64, 1, strides=1, padding="same",
                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                             bias_initializer='zeros')(homodyne_time14)
    homodyne_time15 = BatchNormalization()(homodyne_time15)
    homodyne_time15 = Activation(custom_activation, name='SpecialActivation23')(homodyne_time15)
    homodyne_time15 = Conv1D(64, 4, strides=1, padding="same",
                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                             bias_initializer='zeros')(homodyne_time15)
    homodyne_time15 = BatchNormalization()(homodyne_time15)
    homodyne_time15 = Activation(custom_activation, name='SpecialActivation24')(homodyne_time15)
    homodyne_time15 = Conv1D(128, 1, strides=1, padding="same",
                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                             bias_initializer='zeros')(homodyne_time15)
    homodyne_time15 = BatchNormalization()(homodyne_time15)
    homodyne_time15 = Activation(custom_activation, name='SpecialActivation25')(homodyne_time15)
    homodyne_time16 = add([homodyne_time14, homodyne_time15])
    homodyne_time17 = Conv1D(64, 1, strides=1, padding="same",
                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                             bias_initializer='zeros')(homodyne_time16)
    homodyne_time17 = BatchNormalization()(homodyne_time17)
    homodyne_time17 = Activation(custom_activation, name='SpecialActivation26')(homodyne_time17)
    homodyne_time17 = Conv1D(64, 4, strides=1, padding="same",
                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                             bias_initializer='zeros')(homodyne_time17)
    homodyne_time17 = BatchNormalization()(homodyne_time17)
    homodyne_time17 = Activation(custom_activation, name='SpecialActivation27')(homodyne_time17)
    homodyne_time17 = Conv1D(128, 1, strides=1, padding="same",
                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                             bias_initializer='zeros')(homodyne_time17)
    homodyne_time17 = BatchNormalization()(homodyne_time17)
    homodyne_time17 = Activation(custom_activation, name='SpecialActivation28')(homodyne_time17)
    homodyne_time18 = add([homodyne_time16, homodyne_time17])
    homodyne_time19 = Conv1D(64, 1, strides=1, padding="same",
                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                             bias_initializer='zeros')(homodyne_time18)
    homodyne_time19 = BatchNormalization()(homodyne_time19)
    homodyne_time19 = Activation(custom_activation, name='SpecialActivation29')(homodyne_time19)
    homodyne_time19 = Conv1D(64, 4, strides=1, padding="same",
                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                             bias_initializer='zeros')(homodyne_time19)
    homodyne_time19 = BatchNormalization()(homodyne_time19)
    homodyne_time19 = Activation(custom_activation, name='SpecialActivation30')(homodyne_time19)
    homodyne_time19 = Conv1D(128, 1, strides=1, padding="same",
                             kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                             bias_initializer='zeros')(homodyne_time19)
    homodyne_time19 = BatchNormalization()(homodyne_time19)
    homodyne_time19 = Activation(custom_activation, name='SpecialActivation31')(homodyne_time19)
    homodyne_time20 = add([homodyne_time18, homodyne_time19])

    homodyne_time20 = AveragePooling1D(pool_size=4)(homodyne_time20)

    homodyne_time = Flatten()(homodyne_time20)
    DTX_out = Dropout(0.5)(homodyne_time)
    DTX_out = Dense(1225 * 4, activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.001))(DTX_out)
    DTX_out = Dense(1225, kernel_initializer=tf.random_normal_initializer(stddev=0.001))(DTX_out)

    model = Model([homodyne_time_in], [DTX_out])

    model.compile(
        optimizer=opt,
        loss=custom_loss,
        metrics=['accuracy', 'mae', 'mape', 'mse']
    )
    model.load_weights(f'{CURRENT_PATH}/my_model_weights_0107.h5')

    return model


def get_1d_density_matrix():
    model = get_model()

    matfn7 = f'{CURRENT_PATH}/test_data_010701.mat'
    data7 = sio.loadmat(matfn7)
    example_batch = np.array(data7['q_value'])  # matlab variable name
    example_batch.astype(np.float32)
    example_batch = example_batch.reshape((1, 4032, 1))
    example_result = model.predict(example_batch)

    example_result_r = example_result[:, 0:630]
    zerod = np.zeros((1, 35))
    example_result_i = example_result[:, 630:1226]
    example_result_if = np.hstack((example_result_i, zerod))
    density_mtx_pred_r = example_result_r
    density_mtx_pred_i = example_result_if
    # sio.savemat('density_mtx_pred_i_0107.mat', mdict={'density_mtx_pred_i': density_mtx_pred_i})
    # sio.savemat('density_mtx_pred_r_0107.mat', mdict={'density_mtx_pred_r': density_mtx_pred_r})

    return density_mtx_pred_r, density_mtx_pred_i
