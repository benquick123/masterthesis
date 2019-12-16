import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras as k

from utils import Log, rampdown, rampup, data_generator, get_image_augmentations
from model_WN import MeanTeacherModel
from preprocess import preprocess, SVHN
from reinforcement import ActorNetwork, CriticNetwork, ReplayBuffer, ddpg_update, get_rewards


def get_teacher_predictions(x_label_aug, y_label, x_unlabel_aug, y_unlabel):
    # i suppose training=True for teacher as well?
    y_label_pred = teacher_model(x_label_aug, training=True)
    y_unlabel_pred = teacher_model(x_unlabel_aug, training=True)
    y_pred_teacher = tf.concat([y_label_pred, y_unlabel_pred], axis=0)
    
    return y_pred_teacher


def update_student(student_model, student_optimizer, loss, gradient_tape):
    gradients = gradient_tape.gradient(loss, student_model.trainable_variables)
    student_optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))


def update_teacher(student_model, teacher_model, ema_decay):
    # maybe normal average for the first epochs have to be used. look at mean-teacher/pytorch/main.py:191
    student_weights = student_model.weights
    teacher_weights = teacher_model.weights
    
    for student_layer, teacher_layer in zip(student_weights, teacher_weights):
        teacher_layer.assign(teacher_layer * ema_decay + student_layer * (1-ema_decay))


def train(plot=False):
    rampup_fun = rampup(RAMPUP_EPOCHS)
    rampdown_fun = rampdown(RAMPDOWN_EPOCHS, N_EPOCH)
    
    # TODO: sort() -- maps cluster_centers to the closest matches in in previous state. 
    #    -- This preserves the ordering of cluster_centers when presented to the RL neural networks.
    # TODO: K = number of groups to divide estimates into (e.g. 100)
    # TODO: ACC_t0 = accuracy(y_valid, model(x_valid))
    
    for epoch in range(N_EPOCH):
        Log._print("EPOCH: %d" % epoch)
        
        student_optimizer = tf.keras.optimizers.Adam(lr=MAX_STUDENT_LR * rampup_fun(epoch) * rampdown_fun(epoch), 
                                             beta_1=BETA1 * rampdown_fun(epoch) + RAMPDOWN_BETA1 * (1-rampdown_fun(epoch)),
                                             beta_2=BETA2 * rampup_fun(epoch) + RAMPUP_BETA2 * (1-rampup_fun(epoch)),
                                             epsilon=EPSILON)
        
        # TODO: y0 = student_model(X_unlabeled)
        # TODO: S_t0 = sort(kmeans.cluster_centers(y0,  n_clusters=K))
        # TODO: A_t0 = RL_model(S_t0) + action_noise() -- this determines the weights of samples
        # TODO: W_t0 = ...get weights for each unlabeled sample based on A_t and cluster_centers
        
        for batch, ((x_label, y_label, label_indices), (x_unlabel, y_unlabel, unlabel_indices)) in enumerate(dataset):
            x_label_aug_student, x_label_aug_teacher, x_unlabel_aug_student, x_unlabel_aug_teacher = get_image_augmentations(x_label, y_label, x_unlabel, y_unlabel)
            
            y_pred_teacher = get_teacher_predictions(x_label_aug_teacher, y_label, x_unlabel_aug_teacher, y_unlabel)
            
            with tf.GradientTape() as g:
                y_label_pred_student = student_model(x_label_aug_student, training=True)
                y_unlabel_pred_student = student_model(x_unlabel_aug_student, training=True)
                y_pred_student = tf.concat([y_label_pred_student, y_unlabel_pred_student], axis=0)
                
                loss_cce = tf.reduce_mean(k.losses.categorical_crossentropy(y_label, y_label_pred_student))
                loss_mse = tf.reduce_mean(k.losses.mean_squared_error(y_pred_student, y_pred_teacher))
                loss = loss_cce + (rampup_fun(epoch) * MAX_CONSISTENCY_WEIGHT * loss_mse)
                
            update_student(student_model, student_optimizer, loss, g)
            update_teacher(student_model, teacher_model, ema_decay=EMA_DECAY * rampup_fun(epoch) + RAMPUP_EMA_DECAY * (1-rampup_fun(epoch)))
            
            Log._print("BATCH: %03d - loss: %.3f (%.3f, %.3f) - consistency_weight: %.3f" % (batch, loss, loss_cce, loss_mse, rampup_fun(epoch) * MAX_CONSISTENCY_WEIGHT), end="\r")
            
            if batch == NUM_STEPS_PER_EPOCH:
                break
            
        print()
        get_accuracy.update_state(tf.argmax(y_train, axis=1), tf.argmax(student_model.predict(x_train, batch_size=BATCH_SIZE), axis=1))
        train_accuracy = get_accuracy.result().numpy()
        get_accuracy.update_state(tf.argmax(y_test, axis=1), tf.argmax(student_model.predict(x_test, batch_size=BATCH_SIZE), axis=1))
        test_accuracy = get_accuracy.result().numpy()
        Log._print("acc (train): %.3f - acc (test): %.3f" % (train_accuracy, test_accuracy))

        # TODO: RL learning rate rampup
        # TODO: y1 = model(X_unlabeled)
        # TODO: S_t1 = sort(kmeans.cluster_centers(y1,  n_clusters=K))    
        # TODO: ACC_t1 = accuracy(y_valid, model(x_valid))
        # TODO: R = ACC_t1 - ACC_t0
        # TODO: update_DDPG(S_t0, A_t0, R, S_t1)
        # TODO: ACC_t0 = ACC_t1


if __name__ == "__main__":
    # logging initialization
    answer = input("Log?[y/N] ")
    if answer in set(["y", "yes", "Y", "YES", "YA", "YISS"]):
        Log.log = True
        Log.current_time = str(datetime.datetime.now()).replace(":", "-").split(".")[0]
        Log.save_self()
        
    # common hyper params
    N_LABEL = 1000
    N_UNLABEL = 73000
    N_VALID = 0
    BATCH_SIZE = 100
    NUM_LABELED_PER_BATCH = 50
    NUM_UNLABELED_PER_BATCH = BATCH_SIZE - NUM_LABELED_PER_BATCH
    N_EPOCH = 1000
    NUM_STEPS_PER_EPOCH = (N_UNLABEL + N_LABEL) // BATCH_SIZE
    
    RAMPUP_EPOCHS = 50
    RAMPDOWN_EPOCHS = 30
    
    # mean teacher hyperparams   
    MAX_CONSISTENCY_WEIGHT = 100.0
    RAMPUP_EMA_DECAY = 0.99
    EMA_DECAY = 0.999
    
    # studen Adam hyperparams
    MAX_STUDENT_LR = 0.003
    RAMPDOWN_BETA1 = 0.5
    RAMPUP_BETA2 = 0.99
    BETA1 = 0.9
    BETA2 = 0.999
    EPSILON = 1e-8
    
    # get accuracy function
    get_accuracy = k.metrics.Accuracy()

    Log._print("N_LABEL = %d, N_UNLABEL = %d, MODEL_LR = %.4f" % (N_LABEL, N_UNLABEL, MAX_STUDENT_LR))
    Log._print("BATCH_SIZE = %d, N_EPOCH = %d, NUM_STEPS = %d" % (BATCH_SIZE, N_EPOCH, NUM_STEPS_PER_EPOCH))
    
    ################################## DATA LOADING ####################################
    # load data
    data_filepath = "/opt/workspace/host_storage_hdd/"
    (x_train, y_train), (x_test, y_test) = SVHN.load_data(path=data_filepath)
    (x_train_labeled, y_train_labeled), (x_train_unlabeled, y_train_unlabeled), (x_valid, y_valid) = preprocess(x_train, y_train, N_LABEL, N_UNLABEL, N_VALID)
    
    # initialize tensorflow dataset
    labeled_dataset = tf.data.Dataset.from_generator(data_generator, output_types=(tf.float32, tf.float32, tf.int32), args=[x_train_labeled, y_train_labeled]).shuffle(N_LABEL*2).batch(NUM_LABELED_PER_BATCH)
    unlabeled_dataset = tf.data.Dataset.from_generator(data_generator, output_types=(tf.float32, tf.float32, tf.int32), args=[x_train_unlabeled, y_train_unlabeled]).shuffle(N_UNLABEL*2).batch(NUM_UNLABELED_PER_BATCH)
    dataset = tf.data.Dataset.zip((labeled_dataset, unlabeled_dataset))
    
    # get input shapes
    input_shape = x_train.shape[1:]
    output_shape = y_train.shape[1]
    
    ############################### MODEL INITIALIZATION ###############################
    # initialize base model, model opzimizer and loss
    student_model = MeanTeacherModel(input_shape, output_shape)
    teacher_model = MeanTeacherModel(input_shape, output_shape)
    
    # initial accuracy
    get_accuracy.update_state(tf.argmax(y_test, axis=1), tf.argmax(student_model.predict(x_test, batch_size=BATCH_SIZE), axis=1))
    Log._print("initial accuracy: %.3f" % (get_accuracy.result().numpy()))
    
    train()