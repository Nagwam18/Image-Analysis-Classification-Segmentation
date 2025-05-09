from tensorflow.keras import layers, models

def unet_model(input_shape=(512, 512, 1)):
    inputs = layers.Input(input_shape)
    x = layers.Rescaling(1./255)(inputs)

    # Encoder
    f1, p1 = downsample_block(x, 32)
    f2, p2 = downsample_block(p1, 64)
    f3, p3 = downsample_block(p2, 128)
    f4, p4 = downsample_block(p3, 256)

    # Bottleneck
    bottleneck = double_conv(p4, 512)

    # Decoder
    u6 = upsample_block(bottleneck, 256, f4)
    u7 = upsample_block(u6, 128, f3)
    u8 = upsample_block(u7, 64, f2)
    u9 = upsample_block(u8, 32, f1)

    outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u9)

    model = models.Model(inputs, outputs, name='LUNG_UNet')
    return model
