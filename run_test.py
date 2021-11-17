import argparse
from glob import glob
import os
from model.unet import UNet
from data.load_data import load_data
from data.class_name import ClassName
from config.hyper_parameters import HyperParameter
from utils.common import *
from utils.predictions import *
from config.file_path import FilePath


def load_model(model_file_dir):
    save_model_file_names = sorted(glob(os.path.join(model_file_dir, "*.h5")))

    if len(save_model_file_names) > 0:
        last_file = save_model_file_names[len(save_model_file_names) - 1]

        model = UNet(
            input_shape=(HyperParameter.IMAGE_WIDTH, HyperParameter.IMAGE_HEIGHT, HyperParameter.IMAGE_CHANNEL),
            output_channels=len(ClassName.CLASS_NAMES))

        model.load_weights(last_file)

        print("{} is loaded.".format(last_file))
        return model
    else:
        raise Exception("It can't find model files.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UNet test")
    parser.add_argument("--model-file-dir", help="Model file directory", type=str,
                        required=True, default=FilePath.MODEL_FILE_BASE_DIR, metavar=FilePath.MODEL_FILE_BASE_DIR)

    args = parser.parse_args()
    model_file_dir = args.model_file_dir

    train_dataset, test_dataset, steps_per_epoch, validation_steps, info = load_data()

    model = load_model(model_file_dir=model_file_dir)
    model.summary()

    y_true_images, y_true_segments = get_test_image_and_annotation_arrays(test_dataset=test_dataset, info=info)

    results = model.predict(test_dataset, steps=info.splits['test'].num_examples // HyperParameter.BATCH_SIZE)
    results = np.argmax(results, axis=3)
    results = results[..., tf.newaxis]

    cls_wise_iou, cls_wise_dice_score = class_wise_metrics(y_true_segments, results)

    for idx, iou in enumerate(cls_wise_iou):
        spaces = ' ' * (10 - len(ClassName.CLASS_NAMES[idx]) + 2)
        print("{}{}{} ".format(ClassName.CLASS_NAMES[idx], spaces, iou))
    # show the Dice Score for each class

    for idx, dice_score in enumerate(cls_wise_dice_score):
        spaces = ' ' * (10 - len(ClassName.CLASS_NAMES[idx]) + 2)
        print("{}{}{} ".format(ClassName.CLASS_NAMES[idx], spaces, dice_score))

    for i in range(10):
        show_index = np.random.choice(info.splits['train'].num_examples, replace=False)
        y_pred_mask = make_predictions(model, y_true_images[show_index])
        iou, dice_score = class_wise_metrics(y_true_segments[show_index], y_pred_mask)
        display_with_metrics([y_true_images[show_index], y_pred_mask, y_true_segments[show_index]],
                             iou,
                             dice_score,
                             ClassName.CLASS_NAMES)

    plt.show()
