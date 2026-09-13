"""LITE Multivariate classifier with a joint-attention (SE) block."""

__all__ = ["LITE_MV_SE_CLASSIFIER"]

import tensorflow as tf

from deep_rehab_pile.classifiers._lite_mv import LITE_MV_CLASSIFIER


class LITE_MV_SE_CLASSIFIER(LITE_MV_CLASSIFIER):
    """Define LITE Multivariate Classifier with joint-attention re-weighting.

    Identical to :class:`LITE_MV_CLASSIFIER`, except the raw input
    channels (joints x dimensions) are first re-weighted by a
    Squeeze-and-Excitation (SE) block before being fed to the LITE
    inception module. The SE block learns, per exercise, which joints
    matter most for the classification, addressing the lack of any
    joint-level importance mechanism in the original LITE-MV
    architecture.

    Parameters
    ----------
    output_dir: str
        Directory to save the output results.
    best_file_name: str
        Name of the file to save the best model.
    init_file_name: str,
        The name of the init model to save.
    length_TS: int,
        The length of input skeleton sequence.
    n_joints: int,
        The number of joints in the skeleton.
    n_dim: int,
        The number of dimensions per joint.
    batch_size: int, optional, default=64
        Batch size for training.
    epochs: int, optional, default=1500
        Number of epochs for training.
    n_filters : int or list of int32, default = 32
        The number of filters used in one lite layer, if not a list, the same
        number of filters is used in all lite layers.
    kernel_size : int or list of int, default = 40
        The head kernel size used for each lite layer, if not a list, the same
        is used in all lite layers.
    strides : int or list of int, default = 1
        The strides of kernels in convolution layers for each lite layer,
        if not a list, the same is used in all lite layers.
    activation : str or list of str, default = 'relu'
        The activation function used in each lite layer, if not a list,
        the same is used in all lite layers.
    reduction_ratio : int, default = 4
        The channel reduction ratio used in the squeeze-and-excitation
        bottleneck, following [1]_.

    References
    ----------
    .. [1] Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation
    networks." Proceedings of the IEEE conference on computer vision
    and pattern recognition. 2018.
    """

    def __init__(
        self,
        output_dir: str,
        best_file_name: str,
        init_file_name: str,
        length_TS: int,
        n_joints: int,
        n_dim: int,
        batch_size: int = 64,
        epochs: int = 1500,
        n_filters: int = 32,
        kernel_size: int = 40,
        strides: int = 1,
        activation: str = "relu",
        reduction_ratio: int = 4,
    ):
        super().__init__(
            output_dir=output_dir,
            best_file_name=best_file_name,
            init_file_name=init_file_name,
            length_TS=length_TS,
            n_joints=n_joints,
            n_dim=n_dim,
            batch_size=batch_size,
            epochs=epochs,
            n_filters=n_filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
        )

        self.reduction_ratio = reduction_ratio

    def _squeeze_and_excitation(self, input_tensor, n_channels):
        reduced_units = max(1, n_channels // self.reduction_ratio)

        squeeze = tf.keras.layers.GlobalAveragePooling1D()(input_tensor)

        excitation = tf.keras.layers.Dense(units=reduced_units, activation="relu")(
            squeeze
        )
        excitation = tf.keras.layers.Dense(units=n_channels, activation="sigmoid")(
            excitation
        )
        excitation = tf.keras.layers.Reshape((1, n_channels))(excitation)

        return tf.keras.layers.Multiply()([input_tensor, excitation])

    def _build_model(self, return_model: bool = False, compile_model: bool = True):
        self.n_channels = self.n_joints * self.n_dim

        input_layer = tf.keras.layers.Input((self.length_TS, self.n_channels))

        joint_attention = self._squeeze_and_excitation(
            input_tensor=input_layer, n_channels=self.n_channels
        )

        inception = self._inception_module(
            input_tensor=joint_attention,
            dilation_rate=1,
            use_custom_filters=True,
            use_multiplexing=True,
        )

        _kernel_size = self.kernel_size // 2

        input_tensor = inception

        dilation_rate = 1

        for i in range(2):
            dilation_rate = 2 ** (i + 1)

            x = self._fcn_module(
                input_tensor=input_tensor,
                kernel_size=_kernel_size // (2**i),
                n_filters=self.n_filters,
                dilation_rate=dilation_rate,
            )

            input_tensor = x

        gap = tf.keras.layers.GlobalAveragePooling1D()(x)

        output_layer = tf.keras.layers.Dense(
            units=self.n_classes, activation="softmax"
        )(gap)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        if compile_model:
            model.compile(loss="categorical_crossentropy", optimizer="Adam")

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=50, min_lr=1e-4
            )

            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.output_dir + self.best_file_name + ".keras",
                monitor="loss",
                save_best_only=True,
            )

            self.callbacks = [reduce_lr, model_checkpoint]

        if return_model:
            return model
        else:
            self.model = model
