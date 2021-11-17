import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard

from model.encoder import Encoder
from model.decoder import Decoder
from model.create_layers import create_bottleneck
from config.hyper_parameters import HyperParameter
from config.file_path import FilePath
from utils.LearningRateHistory import LearningRateHistory


class UNet:
    def __init__(self, input_shape, output_channels, name="UNet"):
        self.input = Input(shape=input_shape)

        self.encoder = Encoder(inputs=self.input)
        pool4, features = self.encoder.get_output()

        self.bottle_neck = create_bottleneck(inputs=pool4)

        self.decoder = Decoder(inputs=self.bottle_neck, conv_layers=features, output_channels=output_channels)
        self.output = self.decoder.get_output()

        self.model = Model(inputs=self.input, outputs=self.output, name=name)
        self.criterion = None
        self.optimizer = None

    def get_model(self):
        return self.model

    def _check_compile(self):
        if self.criterion is None or self.optimizer is None:
            self.criterion = SparseCategoricalCrossentropy()
            self.optimizer = Adam()

        self.model.compile(optimizer=self.optimizer, loss=self.criterion, metrics=['accuracy'])

    def make_callbacks(self):
        callbacks = []

        model_check_point = ModelCheckpoint(filepath=FilePath.MODEL_FILE_PATH,
                                            monitor='val_loss',
                                            save_best_only=True,
                                            verbose=1)

        tensorboard = TensorBoard(log_dir=FilePath.TENSORBOARD_LOG_DIR)
        learning_rate_history = LearningRateHistory(log_dir=FilePath.TENSORBOARD_LEARNING_RATE_LOG_DIR)

        def _learning_rate_scheduler(epoch):
            if epoch+1 <= HyperParameter.NUM_EPOCHS // 5:
                return HyperParameter.LEARNING_RATE
            else:
                return HyperParameter.LEARNING_RATE * tf.math.exp(0.1 * (5 - epoch))

        learning_rate_scheduler = LearningRateScheduler(schedule=_learning_rate_scheduler, verbose=1)

        callbacks.append(model_check_point)
        callbacks.append(tensorboard)
        callbacks.append(learning_rate_scheduler)
        callbacks.append(learning_rate_history)

        return callbacks

    def train_on_epoch(self, train_data, validation_data, epochs, steps_per_epoch, validation_steps, verbose=1):
        self._check_compile()

        history = self.model.fit(train_data,
                                 validation_data=validation_data,
                                 epochs=epochs,
                                 callbacks=self.make_callbacks(),
                                 steps_per_epoch=steps_per_epoch,
                                 validation_steps=validation_steps,
                                 verbose=verbose)

        return history

    def predict(self, test_dataset, steps=None):
        predict_result = None
        if self.model is not None:
            if steps is not None:
                predict_result = self.model.predict(test_dataset, steps)
            else:
                predict_result = self.model.predict(test_dataset)

        return predict_result

    def load_weights(self, model_file_path):
        if self.model is not None:
            self.model.load_weights(filepath=model_file_path)

    def summary(self):
        if self.model is not None:
            self.model.summary()
