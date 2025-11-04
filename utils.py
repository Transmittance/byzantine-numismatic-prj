# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 02:45:02 2022

@author: Heigrast
"""
import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import time


# Выводит информацию о процессе обучения
def print_info( test_gen, preds, print_code, save_dir, subject ):
    class_dict = test_gen.class_indices
    labels = test_gen.labels
    file_names = test_gen.filenames
    error_list = []
    true_class = []
    pred_class = []
    prob_list  = []
    new_dict   = {}
    error_indices = []
    y_pred = []
    for key,value in class_dict.items():
        new_dict[value] = key             # dictionary {integer of class number: string of class name}
    # store new_dict as a text fine in the save_dir
    classes = list(new_dict.values())     # list of string of class names
    errors = 0
    for i, p in enumerate(preds):
        pred_index = np.argmax(p)
        true_index = labels[i]  # labels are integer values
        if pred_index != true_index: # a misclassification has occurred
            error_list.append(file_names[i])
            true_class.append(new_dict[true_index])
            pred_class.append(new_dict[pred_index])
            prob_list.append(p[pred_index])
            error_indices.append(true_index)
            errors = errors + 1
        y_pred.append(pred_index)
    if print_code != 0:
        if errors > 0:
            if print_code>errors:
                r = errors
            else:
                r = print_code
            msg ='{0:^28s}{1:^28s}{2:^28s}{3:^16s}'.format('Filename', 'Predicted Class' , 'True Class', 'Probability')
            print_in_color(msg, (0,255,0),(55,65,80))
            for i in range(r):
                split1 = os.path.split(error_list[i])
                split2 = os.path.split(split1[0])
                fname  = split2[1] + '/' + split1[1]
                msg    = '{0:^28s}{1:^28s}{2:^28s}{3:4s}{4:^6.4f}'.format(fname, pred_class[i],true_class[i], ' ', prob_list[i])
                print_in_color(msg, (255,255,255), (55,65,60))
                #print(error_list[i]  , pred_class[i], true_class[i], prob_list[i])
        else:
            msg = 'With accuracy of 100 % there are no errors to print'
            print_in_color(msg, (0,255,0),(55,65,80))
    if errors > 0:
        plot_bar   = []
        plot_class = []
        for  key, value in new_dict.items():
            count = error_indices.count(key)
            if count != 0:
                plot_bar.append(count) # list containg how many times a class c had an error
                plot_class.append(value)   # stores the class
        fig=plt.figure()
        fig.set_figheight(len(plot_class)/3)
        fig.set_figwidth(10)
        plt.style.use('fivethirtyeight')
        for i in range(0, len(plot_class)):
            c = plot_class[i]
            x = plot_bar[i]
            plt.barh(c, x, )
            plt.title( ' Errors by Class on Test Set')
    y_true = np.array(labels)
    y_pred = np.array(y_pred)
    if len(classes)<= 30:
        # create a confusion matrix
        cm = confusion_matrix(y_true, y_pred )
        length = len(classes)
        if length < 8:
            fig_width  = 8
            fig_height = 8
        else:
            fig_width = int(length * .5)
            fig_height = int(length * .5)
        plt.figure(figsize = (fig_width, fig_height))
        sns.heatmap(cm, annot = True, vmin = 0, fmt = 'g', cmap = 'Blues', cbar = False)
        plt.xticks(np.arange(length) + .5, classes, rotation = 90)
        plt.yticks(np.arange(length) + .5, classes, rotation =0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    clr = classification_report(y_true, y_pred, target_names=classes)
    print("Classification Report:\n----------------------\n", clr)

# Сохраняет результаты обучения в файл
def saver(save_path, model, model_name, subject, accuracy, img_size, scalar, generator):
    # first save the model
    save_id = str (model_name +  '-' + subject +'-'+ str(accuracy)[:str(accuracy).rfind('.')+3] + '.h5')
    model_save_loc = os.path.join(save_path, save_id)
    model.save(model_save_loc)
    print_in_color ('model was saved as ' + model_save_loc, (0,255,0),(55,65,80))
    # now create the class_df and convert to csv file
    class_dict = generator.class_indices
    height = []
    width = []
    scale = []
    for i in range(len(class_dict)):
        height.append(img_size[0])
        width.append(img_size[1])
        scale.append(scalar)
    Index_series  = pd.Series(list(class_dict.values()), name='class_index')
    Class_series  = pd.Series(list(class_dict.keys()), name='class')
    Height_series = pd.Series(height, name='height')
    Width_series  = pd.Series(width, name='width')
    Scale_series  = pd.Series(scale, name='scale by')

    class_df = pd.concat([Index_series, Class_series, Height_series, Width_series, Scale_series], axis=1)
    csv_name ='class_dict.csv'
    csv_save_loc = os.path.join(save_path, csv_name)
    class_df.to_csv(csv_save_loc, index=False)
    print_in_color ('class csv file was saved as ' + csv_save_loc, (0,255,0),(55,65,80))

    return model_save_loc, csv_save_loc

def print_in_color(txt_msg, fore_tupple, back_tupple,):
    #prints the text_msg in the foreground color specified by fore_tupple with the background specified by back_tupple
    #text_msg is the text, fore_tupple is foreground color tupple (r,g,b), back_tupple is background tupple (r,g,b)
    rf,gf,bf = fore_tupple
    rb,gb,bb = back_tupple
    msg = '{0}' + txt_msg
    mat = '\33[38;2;' + str(rf) +';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' +str(gb) + ';' + str(bb) +'m'
    print(msg.format(mat), flush = True)
    print('\33[0m', flush = True) # returns default print color to back to black
    return

# Класс колбеков, он дает возможность выводить доп информацию во время обучения через переопределение методов родителя
class LRA(keras.callbacks.Callback):
    def __init__(self, model, base_model, patience, stop_patience, threshold, factor, dwell, batches, initial_epoch, epochs, ask_epoch):
        super(LRA, self).__init__()
        # НЕ присваиваем self.model = model — Keras сам выставит self.model
        self.base_model = base_model
        self.patience = patience
        self.stop_patience = stop_patience
        self.threshold = threshold
        self.factor = factor
        self.dwell = dwell
        self.batches = batches
        self.initial_epoch = initial_epoch
        self.epochs = epochs
        self.ask_epoch = ask_epoch
        self.ask_epoch_initial = ask_epoch
        # служебные счётчики
        self.count = 0
        self.stop_count = 0
        self.best_epoch = 1
        self.highest_tracc = 0.0
        self.lowest_vloss = np.inf
        # Эти поля заполним в on_train_begin(), когда self.model уже доступна
        self.initial_lr = None
        self.best_weights = None
        self.initial_weights = None

    def _read_lr(self):
        lr_obj = self.model.optimizer.learning_rate
        # если это Variable/Tensor — читаем через K.get_value; иначе (float/число) — просто каст
        try:
            return float(tf.keras.backend.get_value(lr_obj))
        except Exception:
            return float(lr_obj)

    def _write_lr(self, new_lr: float):
        lr_obj = self.model.optimizer.learning_rate
        # если это Variable — меняем через assign; иначе просто переустанавливаем атрибут
        if hasattr(lr_obj, "assign"):
            lr_obj.assign(float(new_lr))
        else:
            self.model.optimizer.learning_rate = float(new_lr)


    def on_train_begin(self, logs=None):
        # теперь self.model уже установлен Keras
        lr_t = self.model.optimizer.learning_rate
        self.initial_lr = self._read_lr()
        self.initial_weights = self.model.get_weights()
        self.best_weights = self.model.get_weights()

        if self.base_model is not None:
            status = self.base_model.trainable
            msg = 'initializing callback starting train with base_model trainable' if status else \
                  'initializing callback starting training with base_model not trainable'
        else:
            msg = 'initialing callback and starting training'
        print_in_color(msg, (244, 252, 3), (55, 65, 80))
        hdr = '{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:^8s}'.format(
            'Epoch','Loss','Accuracy','V_loss','V_acc','LR','Next LR','Monitor','Duration'
        )
        print_in_color(hdr, (244,252,3), (55,65,80))
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        stop_time = time.time()
        tr_duration = stop_time - self.start_time
        hours = tr_duration // 3600
        minutes = (tr_duration - (hours * 3600)) // 60
        seconds = tr_duration - ((hours * 3600) + (minutes * 60))
        # восстановим лучшие веса
        self.model.set_weights(self.best_weights)
        msg = f'Training is completed - model is set with weights from epoch {self.best_epoch} '
        print_in_color(msg, (0,255,0), (55,65,80))
        msg = f'training elapsed time was {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds)'
        print_in_color(msg, (0,255,0), (55,65,80))

    def on_train_batch_end(self, batch, logs=None):
        acc = logs.get('accuracy') * 100
        loss = logs.get('loss')
        msg = '{0:20s}processing batch {1:4s} of {2:5s} accuracy= {3:8.3f}  loss: {4:8.5f}'.format(
            ' ', str(batch), str(self.batches), acc, loss
        )
        print(msg, '\r', end='')

    def on_epoch_begin(self, epoch, logs=None):
        self.now = time.time()

    def on_epoch_end(self, epoch, logs=None):
        later = time.time()
        duration = later - self.now
        # читаем и будем обновлять learning_rate
        current_lr = self._read_lr()
        lr = current_lr

        v_loss = logs.get('val_loss')
        acc    = logs.get('accuracy')
        v_acc  = logs.get('val_accuracy')
        loss   = logs.get('loss')

        if acc < self.threshold:
            monitor = 'accuracy'
            if acc > self.highest_tracc:
                self.highest_tracc = acc
                self.best_weights = self.model.get_weights()
                self.count = 0
                self.stop_count = 0
                if v_loss < self.lowest_vloss:
                    self.lowest_vloss = v_loss
                color = (0,255,0)
                self.best_epoch = epoch + 1
            else:
                if self.count >= self.patience - 1:
                    color = (245,170,66)
                    lr = lr * self.factor
                    self._write_lr(lr)
                    self.count = 0
                    self.stop_count += 1
                    if self.dwell:
                        self.model.set_weights(self.best_weights)
                    else:
                        if v_loss < self.lowest_vloss:
                            self.lowest_vloss = v_loss
                else:
                    self.count += 1
        else:
            monitor = 'val_loss'
            if v_loss < self.lowest_vloss:
                self.lowest_vloss = v_loss
                self.best_weights = self.model.get_weights()
                self.count = 0
                self.stop_count = 0
                color = (0,255,0)
                self.best_epoch = epoch + 1
            else:
                if self.count >= self.patience - 1:
                    color = (245,170,66)
                    lr = lr * self.factor
                    self._write_lr(lr)
                    self.stop_count += 1
                    self.count = 0
                    if self.dwell:
                        self.model.set_weights(self.best_weights)
                else:
                    self.count += 1
                if acc > self.highest_tracc:
                    self.highest_tracc = acc

        msg = f'{str(epoch+1):^3s}/{str(self.epochs):4s} {loss:^9.3f}{acc*100:^9.3f}{v_loss:^9.5f}{v_acc*100:^9.3f}{current_lr:^9.5f}{lr:^9.5f}{monitor:^11s}{duration:^8.2f}'
        print_in_color(msg, color, (55,65,80))

        with open('epoch statistics.txt', 'a') as epoch_stats:
            epoch_stats.write(msg + '\n')

        if self.stop_count > self.stop_patience - 1:
            msg = f' training has been halted at epoch {epoch + 1} after {self.stop_patience} adjustments of learning rate with no improvement'
            print_in_color(msg, (0,255,255), (55,65,80))
            self.model.stop_training = True
        else:
            if self.ask_epoch is not None and epoch + 1 >= self.ask_epoch:
                msg = 'enter H to halt ,F to fine tune model, or an integer for number of epochs to run then ask again'
                print_in_color(msg, (0,255,255), (55,65,80))
                ans = input('')
                if ans in ('H','h'):
                    msg = f'training has been halted at epoch {epoch + 1} due to user input'
                    print_in_color(msg, (0,255,255), (55,65,80))
                    self.model.stop_training = True
                elif ans in ('F','f'):
                    msg ='setting base_model as trainable for fine tuning of model'
                    self.base_model.trainable = True
                    print_in_color(msg, (0,255,255), (55,65,80))
                    hdr = '{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:^8s}'.format(
                        'Epoch','Loss','Accuracy','V_loss','V_acc','LR','Next LR','Monitor','Duration'
                    )
                    print_in_color(hdr, (244,252,3), (55,65,80))
                    self.count = 0
                    self.stop_count = 0
                    self.ask_epoch = epoch + 1 + self.ask_epoch_initial
                else:
                    ans = int(ans)
                    self.ask_epoch += ans
                    msg = ' training will continue until epoch ' + str(self.ask_epoch)
                    print_in_color(msg, (0,255,255), (55,65,80))
                    hdr = '{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:^8s}'.format(
                        'Epoch','Loss','Accuracy','V_loss','V_acc','LR','Next LR','Monitor','Duration'
                    )
                    print_in_color(hdr, (244,252,3), (55,65,80))
