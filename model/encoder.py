from tensorflow.keras.layers import MaxPooling2D, Dropout
from model.create_layers import create_conv_layers


class Encoder:
    def __init__(self, inputs, dropout=0.3, name="Encoder"):
        self.dropout = dropout
        self.name = name
        self.feature1, self.pool1 = self._create_block(inputs=inputs, filters=[64, 64])
        self.feature2, self.pool2 = self._create_block(inputs=self.pool1, filters=[128, 128])
        self.feature3, self.pool3 = self._create_block(inputs=self.pool2, filters=[256, 256])
        self.feature4, self.pool4 = self._create_block(inputs=self.pool3, filters=[512, 512])

    def _create_block(self, inputs, filters, pool_size=(2, 2)):
        feature = create_conv_layers(inputs=inputs, filters=filters)
        pooling = MaxPooling2D(pool_size)(feature)
        pooling = Dropout(rate=self.dropout)(pooling)

        return feature, pooling

    def get_output(self):
        return self.pool4, (self.feature1, self.feature2, self.feature3, self.feature4)
