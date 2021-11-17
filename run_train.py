import argparse
from model.unet import UNet
from data.load_data import load_data
from data.class_name import ClassName
from config.hyper_parameters import HyperParameter
from config.file_path import make_directories
from utils.common import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UNet train")
    parser.add_argument("--epochs", help="Training epochs", type=int,
                        required=False, default=HyperParameter.NUM_EPOCHS, metavar="20")

    args = parser.parse_args()
    epochs = args.epochs

    make_directories()
    train_dataset, test_dataset, steps_per_epoch, validation_steps, _ = load_data()

    model = UNet(input_shape=(HyperParameter.IMAGE_WIDTH, HyperParameter.IMAGE_HEIGHT, HyperParameter.IMAGE_CHANNEL),
                 output_channels=len(ClassName.CLASS_NAMES))
    model.summary()

    history = model.train_on_epoch(train_data=train_dataset,
                                   validation_data=test_dataset,
                                   steps_per_epoch=steps_per_epoch,
                                   validation_steps=validation_steps,
                                   epochs=epochs)

    display_loss(history.history)
