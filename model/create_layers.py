from tensorflow.keras.layers import Conv2D,  ReLU


def create_conv_layers(inputs, filters, kernel_size=(3, 3)):
    outputs = inputs

    for index in range(len(filters)):
        outputs = Conv2D(filters=filters[index],
                         kernel_size=kernel_size,
                         kernel_initializer='he_normal',
                         padding='same')(outputs)

        outputs = ReLU()(outputs)

    return outputs


def create_bottleneck(inputs, filters=[1024, 1024]):
    bottle_neck = create_conv_layers(inputs, filters=filters)

    return bottle_neck
