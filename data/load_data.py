import tensorflow as tf
import tensorflow_datasets as tfds
from data.image_processing import load_image_train, load_image_test
from config.hyper_parameters import HyperParameter


def load_data():
    dataset, info = tfds.load('oxford_iiit_pet:3.2.0', download=False, with_info=True)

    train_dataset = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = dataset['test'].map(load_image_test)

    print("train dataset length: ", len(train_dataset))
    print("test dataset length: ", len(test_dataset))

    train_dataset = \
        train_dataset.cache().shuffle(HyperParameter.BATCH_SIZE).batch(HyperParameter.BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(HyperParameter.BATCH_SIZE)

    steps_per_epoch = info.splits['train'].num_examples // HyperParameter.BATCH_SIZE
    validation_steps = info.splits['test'].num_examples / HyperParameter.BATCH_SIZE // HyperParameter.VAL_SUB_SPLITS

    return train_dataset, test_dataset, steps_per_epoch, validation_steps, info
