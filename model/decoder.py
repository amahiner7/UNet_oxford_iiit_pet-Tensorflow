from tensorflow.keras.layers import Conv2D, Conv2DTranspose, concatenate, Dropout
from model.create_layers import create_conv_layers


class Decoder:
    def __init__(self, inputs, conv_layers, output_channels, dropout=0.3, name="Decoder"):
        self.inputs = inputs
        self.dropout = dropout
        self.name = name

        feature1, feature2, feature3, feature4 = conv_layers

        block1 = self._create_block(inputs=inputs, conv_output=feature4, filters=[512, 512], kernel_size=(3, 3),
                                    strides=(2, 2), dropout=0.3)

        block2 = self._create_block(inputs=block1, conv_output=feature3, filters=[256, 256], kernel_size=(3, 3),
                                    strides=(2, 2), dropout=0.3)

        block3 = self._create_block(inputs=block2, conv_output=feature2, filters=[128, 128], kernel_size=(3, 3),
                                    strides=(2, 2), dropout=0.3)

        block4 = self._create_block(inputs=block3, conv_output=feature1, filters=[64, 64], kernel_size=(3, 3),
                                    strides=(2, 2), dropout=0.3)

        self.outputs = Conv2D(filters=output_channels, kernel_size=(1, 1), activation='softmax')(block4)

    def _create_block(self, inputs, conv_output, filters, kernel_size, strides, dropout):
        conv_t = Conv2DTranspose(filters[0], kernel_size, strides=strides, padding='same')(inputs)
        output = concatenate([conv_t, conv_output])
        output = Dropout(dropout)(output)
        output = create_conv_layers(output, filters, kernel_size=(3, 3))

        return output

    def get_output(self):
        return self.outputs
