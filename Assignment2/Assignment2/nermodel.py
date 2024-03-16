import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, SimpleRNN, LSTM, GRU, Dense
from keras.optimizers.legacy import Adam
from keras.callbacks import EarlyStopping
from keras.losses import sparse_categorical_crossentropy
from keras.saving import register_keras_serializable
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from embedding_layer import *




class NERModel():
    def __init__(self, num_classes, num_features=None, layer=SimpleRNN, embed='word2vec', **kwargs):
        super(NERModel, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.num_features = num_features

        self.embedding_layer = EmbeddingLayer(embed)
        self.layer = layer
        
        self.seq = Sequential()
        self.seq.add(self.layer(units=64,return_sequences=True,use_bias=True))
        self.seq.add(Dense(1024, activation='relu', use_bias=True))
        self.seq.add(Dense(256, activation='relu', use_bias=True))
        self.seq.add(Dense(64, activation='relu', use_bias=True))
        self.seq.add(Dense(num_classes, activation='softmax', use_bias=True))

        self.batch_size = None
        self.model = None
    
    def build_model(self):
        X = Input(shape=(None,self.num_features))
        Y = self.seq(X)
        zeros = tf.zeros(shape=(tf.shape(Y)[0],tf.shape(Y)[1],1), dtype=tf.float32)
        Y = tf.concat([Y,zeros], axis=-1)
        self.model = Model(inputs=X, outputs=Y)

    def compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=self.lr),
            loss=sparse_categorical_crossentropy,
            metrics=[self.macro_f1, 'accuracy'],
            run_eagerly=True
        )

    def fit(self, X, y, X_val, y_val, lr, epochs=50, batch_size=32, patience=None):
        self.batch_size = batch_size
        
        self.lr = lr
        self.epochs = epochs

        X, y = self.embedding_layer.fit_transform(X, y)
        X_val, y_val = self.embedding_layer.transform(X_val, y_val)

        if(self.num_features is None):
            self.num_features = X.shape[-1]

        if ((patience is None) or (patience>int(0.5*epochs))):
            patience = 0.1*epochs

        self.build_model()
        self.compile_model()
        
        self.model.fit(
            x=X,
            y=y, 
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=1,
            shuffle=True,
            callbacks=[EarlyStopping(monitor='val_loss',patience=patience, restore_best_weights=True)],
            batch_size=self.batch_size
        )

        self.generate_plots()

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, X_t, y_t):
        X,y = self.embedding_layer.transform(X_t, y_t)
        res = self.model.evaluate(X, y, verbose=0)
        f1 = res[1]
        acc = res[2]
        return f1, acc
    
    def generate_plots(self):
        if self.model is not None:
            self._plot_losses(
                self.model.history.history['loss'],
                'Training Loss',
                self.model.history.history['val_loss'],
                'Validation Loss'
            )
            self._plot_accuracy(
                self.model.history.history['macro_f1'],
                'Training Macro F1',
                self.model.history.history['val_macro_f1'],
                'Validation Macro F1'
            )

    def _plot_losses(self, tr_loss, tr_label, val_loss=None, val_label=None):
        plt.figure(figsize=(10,8))
        plt.plot(tr_loss, label=tr_label)

        if(val_loss is not None):
            plt.plot(val_loss, label=val_label)

        plt.xlabel('Epochs',fontsize=12)
        plt.ylabel('Loss',fontsize=12)
        plt.legend()
        plt.show()

    def _plot_accuracy(self, tr_acc, tr_label, val_acc=None, val_label=None):
        plt.figure(figsize=(10,8))
        plt.plot(tr_acc, label=tr_label)

        if(val_acc is not None):
            plt.plot(val_acc, label=val_label)

        plt.xlabel('Epochs',fontsize=12)
        plt.ylabel('Accuracy',fontsize=12)
        plt.legend()
        plt.show()

    def macro_f1(self, y_true, y_pred):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.reshape(y_pred,[-1])
        y_true = tf.reshape(y_true,[-1])
        
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        f1 = f1_score(tf.keras.backend.eval(y_true), tf.keras.backend.eval(y_pred), average='macro')
        acc = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))
        return f1, acc
    
    def save(self,file_path):
        self.model.save(file_path)
    
    def load(self,file_path):
        self.model = tf.keras.models.load_model(file_path,custom_objects={'macro_f1':self.macro_f1})



from keras.layers import Bidirectional, BatchNormalization
from keras_crf import CRFModel


class BiLSTM_CRF_NERModel():
    def __init__(self, num_classes, num_features=None, embed='word2vec', **kwargs):
        super(BiLSTM_CRF_NERModel, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.num_features = num_features

        self.embedding_layer = EmbeddingLayer(embed)
        self.seq = Sequential()

        self.seq.add(Bidirectional(LSTM(units=128, return_sequences=True ,use_bias=True)))
        self.seq.add(BatchNormalization(axis=-1))
        self.seq.add(Dense(64, activation="relu", use_bias=True))
        self.seq.add(Dense(num_classes, activation="relu", use_bias=True))
        
        self.batch_size = None
        self.model = None
    
    def build_model(self):
        X = Input(shape=(None,self.num_features))
        Y = self.seq(X)
        self.model = CRFModel(Model(inputs=X, outputs=Y),self.num_classes+1)

    def compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=self.lr),
            metrics=[self.macro_f1, 'accuracy'],
            run_eagerly=True
        )

    def fit(self, X, y, X_val, y_val, lr, epochs=50, batch_size=32, patience=None):
        self.batch_size = batch_size
        
        self.lr = lr
        self.epochs = epochs

        X, y = self.embedding_layer.fit_transform(X, y)
        X_val, y_val = self.embedding_layer.transform(X_val, y_val)

        if(self.num_features is None):
            self.num_features = X.shape[-1]

        if ((patience is None) or (patience>int(0.5*epochs))):
            patience = 0.1*epochs

        self.build_model()
        self.compile_model()
        
        self.model.fit(
            x=X,
            y=y, 
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=1,
            shuffle=True,
            callbacks=[EarlyStopping(monitor='val_loss',patience=patience, restore_best_weights=True)],
            batch_size=self.batch_size
        )

        self.generate_plots()

    def predict(self, X_test):
        preds = self.model.predict(X_test)
        preds = preds[1]
        return preds
    
    def evaluate(self, X_t, y_t):
        X,y = self.embedding_layer.transform(X_t, y_t)
        res = self.model.evaluate(X, y, verbose=0)
        f1 = res[1]
        acc = res[2]
        return f1, acc
    
    def generate_plots(self):
        if self.model is not None:
            self._plot_losses(
                self.model.history.history['loss'],
                'Training Loss',
                self.model.history.history['val_loss'],
                'Validation Loss'
            )
            self._plot_accuracy(
                self.model.history.history['decode_sequence_macro_f1'],
                'Training Macro F1',
                self.model.history.history['val_decode_sequence_macro_f1'],
                'Validation Macro F1'
            )

    def _plot_losses(self, tr_loss, tr_label, val_loss=None, val_label=None):
        plt.figure(figsize=(10,8))
        plt.plot(tr_loss, label=tr_label)

        if(val_loss is not None):
            plt.plot(val_loss, label=val_label)

        plt.xlabel('Epochs',fontsize=12)
        plt.ylabel('Loss',fontsize=12)
        plt.legend()
        plt.show()

    def _plot_accuracy(self, tr_acc, tr_label, val_acc=None, val_label=None):
        plt.figure(figsize=(10,8))
        plt.plot(tr_acc, label=tr_label)

        if(val_acc is not None):
            plt.plot(val_acc, label=val_label)

        plt.xlabel('Epochs',fontsize=12)
        plt.ylabel('Accuracy',fontsize=12)
        plt.legend()
        plt.show()
    
    def macro_f1(self, y_true, y_pred):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.reshape(y_pred,[-1])
        y_true = tf.reshape(y_true,[-1])
        
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        f1 = f1_score(tf.keras.backend.eval(y_true), tf.keras.backend.eval(y_pred), average='macro')
        acc = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))
        return f1, acc

    def save(self,file_path):
        self.model.save(file_path)
    
    def load(self,file_path):
        self.model = tf.keras.models.load_model(file_path,custom_objects={'macro_f1':self.macro_f1})