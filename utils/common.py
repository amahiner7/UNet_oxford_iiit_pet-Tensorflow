import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def display_loss(history):
    train_loss = history['loss']
    val_loss = history['val_loss']

    # 그래프로 표현
    x_len = np.arange(len(train_loss))
    plt.figure()
    plt.plot(x_len, val_loss, marker='.', c="blue", label='Validation loss')
    plt.plot(x_len, train_loss, marker='.', c="red", label='Train loss')
    # 그래프에 그리드를 주고 레이블을 표시
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.show()

    if history.get('learning_rate') is not None:
        learning_rate = history['learning_rate']
        x_len = np.arange(len(learning_rate))
        plt.clf()
        plt.figure()
        plt.plot(x_len, learning_rate, marker='.', c="yellow", label='Learning rate')
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('Learning rate')
        plt.title('Learning rate')
        plt.show()


def display_with_metrics(display_list, iou_list, dice_score_list, class_names):
    '''displays a list of images/masks and overlays a list of IOU and Dice Scores'''

    metrics_by_id = [(idx, iou, dice_score) for idx, (iou, dice_score) in enumerate(zip(iou_list, dice_score_list)) if
                     iou > 0.0]
    metrics_by_id.sort(key=lambda tup: tup[1], reverse=True)  # sorts in place

    display_string_list = ["{}: IOU: {} Dice Score: {}".format(class_names[idx], iou, dice_score) for
                           idx, iou, dice_score in metrics_by_id]
    display_string = "\n\n".join(display_string_list)

    display(display_list, ["Image", "Predicted Mask", "True Mask"], display_string=display_string)


def display(display_list, titles=[], display_string=None):
    '''displays a list of images/masks'''

    plt.figure(figsize=(15, 15))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
        if display_string and i == 1:
            plt.xlabel(display_string, fontsize=12)
        img_arr = tf.keras.preprocessing.image.array_to_img(display_list[i])
        plt.imshow(img_arr)
