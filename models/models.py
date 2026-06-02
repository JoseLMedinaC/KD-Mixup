import tensorflow as tf
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras import layers
import keras_hub

def build_mobilenetv2(input_shape=(224, 224, 3), num_classes=10, dropout_rate=0.3):
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        input_shape=input_shape,
        weights='imagenet',
        pooling="avg"
    )
    x = tf.keras.layers.Dense(256, use_bias=False)(base.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    out = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs=base.input, outputs=out, name="StudentMobileNet"), base

def build_resnet50(input_shape=(224, 224, 3), num_classes=10, dropout_rate=0.3):
    base = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=input_shape,
        weights='imagenet',
        pooling="avg"
    )
    x = tf.keras.layers.Dense(256, use_bias=False)(base.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    out = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs=base.input, outputs=out, name="StudentResNet50"), base


def build_vit_base(input_shape=(224, 224, 3), num_classes=100, dropout_rate=0.3):
    inputs = tf.keras.Input(shape=input_shape, name="input_raw224")
    
    backbone = keras_hub.models.ViTBackbone.from_preset("vit_base_patch16_224_imagenet")
    x = backbone(inputs)  # shape: (B, 197, 768)
    # Extraer solo el CLS token (posición 0)
    x = x[:, 0, :]        # shape: (B, 768)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    logits = tf.keras.layers.Dense(num_classes, activation=None, name="logits")(x)
    softmax_out = tf.keras.layers.Activation("softmax", name="predictions")(logits)
    return tf.keras.Model(inputs=inputs, outputs=softmax_out, name="ViTBase")

def build_efficientnetv2b0(input_shape=(224, 224, 3), num_classes=10, dropout_rate=0.3):
    inputs = tf.keras.Input(shape=input_shape, name="input_raw224")

    # Identity layer (para dejar explícito que no hay normalización aquí)
    x = tf.keras.layers.Lambda(lambda im: im, name="identity")(inputs)

    # Base model sin top, con pesos de ImageNet
    base = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        input_tensor=x,
        weights="imagenet"
    )

    # Head del modelo
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    logits = tf.keras.layers.Dense(num_classes, activation=None, name="logits")(x)
    softmax_out = tf.keras.layers.Activation("softmax", name="predictions")(logits)

    # SOLO softmax como salida
    model = tf.keras.Model(inputs=inputs, outputs=softmax_out, name="EffNetV2B0")
    return model

def build_resnet152v2(input_shape=(224, 224, 3), num_classes=10, dropout_rate=0.3):
    inputs = tf.keras.Input(shape=input_shape, name="input_raw224")
    x = tf.keras.layers.Lambda(lambda im: im, name="identity")(inputs)

    base = tf.keras.applications.ResNet152V2(
        include_top=False,
        input_tensor=x,
        weights="imagenet"
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    logits = tf.keras.layers.Dense(num_classes, activation=None, name="logits")(x)
    softmax_out = tf.keras.layers.Activation("softmax", name="predictions")(logits)
    # SOLO softmax como salida
    model = tf.keras.Model(inputs=inputs, outputs=softmax_out, name="ResNet152V2")
    return model

def build_convnexttiny(input_shape=(224, 224, 3), num_classes=10, dropout_rate=0.3):
    inputs = tf.keras.Input(shape=input_shape, name="input_raw224")
    x = tf.keras.layers.Lambda(lambda im: im, name="identity")(inputs)

    base = ConvNeXtTiny(
        include_top=False,
        input_tensor=x,
        weights="imagenet"
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    logits = tf.keras.layers.Dense(num_classes, activation=None, name="logits")(x)
    softmax_out = tf.keras.layers.Activation("softmax", name="predictions")(logits)

    model = tf.keras.Model(inputs=inputs, outputs=softmax_out, name="ConvNeXtTiny")
    return model

def build_convnextlarge(input_shape=(224, 224, 3), num_classes=10, dropout_rate=0.3):
    inputs = tf.keras.Input(shape=input_shape, name="input_raw224")
    x = tf.keras.layers.Lambda(lambda im: im, name="identity")(inputs)
    base = tf.keras.applications.ConvNeXtLarge(
        include_top=False,
        input_tensor=x,
        weights="imagenet"
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    logits = tf.keras.layers.Dense(num_classes, activation=None, name="logits")(x)
    softmax_out = tf.keras.layers.Activation("softmax", name="predictions")(logits)
    # SOLO softmax como salida
    model = tf.keras.Model(inputs=inputs, outputs=softmax_out, name="ConvNeXtLarge")
    return model

def conv_bn_relu(x, filters, kernel_size, strides=1):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)

def build_mobileone_s0(input_shape=(224, 224, 3), include_top=False, pretrained=False, pooling=None):
    inputs = tf.keras.Input(shape=input_shape)
    x = conv_bn_relu(inputs, 64, 3)
    x = conv_bn_relu(x, 64, 3)
    x = conv_bn_relu(x, 128, 3)
    x = conv_bn_relu(x, 128, 3)
    x = conv_bn_relu(x, 256, 3)
    x = conv_bn_relu(x, 256, 3)
    x = conv_bn_relu(x, 512, 3)
    if pooling == "avg":
        x = layers.GlobalAveragePooling2D()(x)
    elif include_top:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1000, activation='softmax')(x)
    return tf.keras.Model(inputs, x, name="MobileOne_S0")

def ResNet152V2IN():
    base = tf.keras.applications.ResNet152V2(
        include_top=True,
        weights="imagenet",
        input_shape=(224,224,3)
    )
    return base

def ConvNextLargeIN():
    base = tf.keras.applications.ConvNeXtLarge(
        include_top=True,
        weights="imagenet",
        input_shape=(224,224,3)
    )
    return base

def ConvNextTinyIN():
    base = tf.keras.applications.ConvNeXtTiny(
        include_top=True,
        weights="imagenet",
        input_shape=(224,224,3)
    )
    return base
