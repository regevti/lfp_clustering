import tensorflow.keras as keras
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class VAE:
    def __init__(self, original_dim, intermediate_dim=64, latent_dim=2):
        self.original_dim = original_dim
        self.encoder = None
        self.vae = self.build(intermediate_dim, latent_dim)

    def build(self, intermediate_dim, latent_dim, n_features=1) -> keras.Model:
        inputs = layers.Input(shape=(self.original_dim, n_features))
        # h = layers.Dense(intermediate_dim, activation='relu')(inputs)
        h = layers.LSTM(intermediate_dim, activation='relu')(inputs)
        z_mean = layers.Dense(latent_dim)(h)
        z_log_sigma = layers.Dense(latent_dim)(h)

        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=0.1)
            return z_mean + K.exp(z_log_sigma) * epsilon

        z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
        # Create encoder
        self.encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

        # Create decoder
        latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
        # x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
        # outputs = layers.Dense(self.original_dim, activation='sigmoid')(x)
        x = layers.RepeatVector(self.original_dim)(latent_inputs)
        x = layers.LSTM(intermediate_dim, activation='relu', return_sequences=True)(x)
        outputs = layers.TimeDistributed(layers.Dense(n_features))(x)
        decoder = keras.Model(latent_inputs, outputs, name='decoder')

        # instantiate VAE model
        outputs = decoder(self.encoder(inputs)[2])
        vae = keras.Model(inputs, outputs, name='vae_mlp')

        # KL loss
        cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        reconstruction_loss = cross_entropy(inputs, outputs)
        reconstruction_loss *= self.original_dim
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)

        vae.compile(optimizer='adam')
        return vae

    def train(self, V, test_size=0.1, epochs=100, batch_size=32, is_print=False):
        assert V.shape[1] == self.original_dim, f'Input shape should be {self.original_dim}'
        if test_size:
            x_train, x_test, _, _ = train_test_split(V, V, test_size=0.1, random_state=42)
            validation_data = (x_test, x_test)
        else:
            x_train, x_test, validation_data = V, V, None
        history = self.vae.fit(x_train, x_train,
                     epochs=epochs,
                     batch_size=batch_size,
                     verbose=0,
                     validation_data=validation_data)

        if is_print:
            self.print_latent(x_test, history, batch_size)

        return history

    def print_latent(self, x_test, history, batch_size=32):
        x_test_encoded = self.encoder.predict(x_test, batch_size=batch_size)[1]
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        axes[0].plot(history.history['loss'], label='loss')
        axes[0].plot(history.history['val_loss'], label='val_loss')
        axes[0].legend()

        if x_test_encoded.shape[1] == 2:
            axes[1].scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
        else:
            X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(x_test_encoded)
            axes[1].scatter(X_embedded[:, 0], X_embedded[:, 1])


class VAE2:
    def __init__(self, original_dim, intermediate_dim=64, latent_dim=2):
        self.original_dim = original_dim
        self.encoder = None
        self.decoder = None
        self.vae = self.build(intermediate_dim, latent_dim)

    def build(self, intermediate_dim, latent_dim, n_features=1) -> keras.Model:
        def set_seed(seed):
            tf.random.set_seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            random.seed(seed)

        def sampling(args):
            z_mean, z_log_sigma = args
            batch_size = tf.shape(z_mean)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
            return z_mean + K.exp(0.5 * z_log_sigma) * epsilon

        def vae_loss(original, out, z_log_sigma, z_mean):
            reconstruction = K.mean(K.square(original - out)) * self.original_dim
            kl = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
            return reconstruction + kl

        # set_seed(33)
        ### encoder ###
        # inp_original = layers.Input(shape=(self.original_dim, 1))
        inp = layers.Input(shape=(self.original_dim, 1), name='enc_input')
        enc = layers.LSTM(64, name='enc_lstm')(inp)
        z = layers.Dense(32, activation="relu", name='enc_dense1')(enc)
        z = layers.LeakyReLU(alpha=0.3)(z)
        z = layers.Dropout(rate=0.2)(z)
        z_mean = layers.Dense(latent_dim, name='z_mean')(z)
        z_log_sigma = layers.Dense(latent_dim, name='z_log_sigma')(z)
        self.encoder = keras.Model([inp], [z_mean, z_log_sigma])

        ### decoder ###
        inp_z = layers.Input(shape=(latent_dim,))
        dec = layers.RepeatVector(self.original_dim)(inp_z)
        dec = layers.LSTM(64, return_sequences=True)(dec)
        out = layers.TimeDistributed(layers.Dense(1))(dec)
        self.decoder = keras.Model([inp_z], out)

        ### encoder + decoder ###
        z_mean, z_log_sigma = self.encoder([inp])
        z = layers.Lambda(sampling)([z_mean, z_log_sigma])
        pred = self.decoder([z])

        vae = keras.Model([inp], pred)
        vae.add_loss(vae_loss(inp, pred, z_log_sigma, z_mean))
        vae.compile(loss=None, optimizer=keras.optimizers.Adam(lr=1e-3))

        return vae

    def train(self, V, test_size=0.1, epochs=100, batch_size=32, is_print=False):
        es = EarlyStopping(patience=10, verbose=1, min_delta=0.001, monitor='val_loss', mode='auto',
                           restore_best_weights=True)
        self.vae.fit(V, V, batch_size=batch_size, epochs=epochs, validation_split=test_size, shuffle=False, callbacks=[es])