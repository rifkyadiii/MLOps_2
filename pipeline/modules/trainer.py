"""
Trainer module for TFX pipeline.
Handles dataset loading, model building, tuning, and training.
"""

import json
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft
from keras_tuner import HyperParameters
from keras_tuner.tuners import RandomSearch
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult

LABEL_KEY = 'price_range'
BATCH_SIZE = 64


def _gzip_reader_fn(filenames):
    """Read TFRecord GZIP files."""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def _input_fn(file_pattern, tf_transform_output, batch_size=BATCH_SIZE, shuffle=False):
    """
    Read and parse transformed TFRecords with label separation.
    """
    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    feature_spec[LABEL_KEY] = tf.io.FixedLenFeature([], tf.int64)

    def _parse_batch(serialized_batch):
        example = tf.io.parse_example(serialized_batch, feature_spec)
        label = example.pop(LABEL_KEY)
        return example, label

    files = tf.io.gfile.glob(file_pattern)
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.interleave(
        _gzip_reader_fn,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if shuffle:
        ds = ds.shuffle(2048)
    ds = ds.batch(batch_size)
    ds = ds.map(_parse_batch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds.repeat()


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """
    Create a serving function for tf.Example inputs.
    """
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
        ]
    )
    def serve_tf_examples_fn(serialized_tf_examples):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_feature_spec.pop(LABEL_KEY, None)
        parsed = tf.io.parse_example(serialized_tf_examples, raw_feature_spec)
        transformed = model.tft_layer(parsed)
        return model(transformed)

    return serve_tf_examples_fn


def model_builder(hp: HyperParameters, transform_graph_path: str) -> keras.Model:
    """
    Build a Keras model with tunable hyperparameters.
    """
    tf_transform_output = tft.TFTransformOutput(transform_graph_path)
    feature_keys = [
        k for k in tf_transform_output.transformed_feature_spec().keys()
        if k != LABEL_KEY
    ]

    inputs = {k: keras.layers.Input(shape=(1,), name=k) for k in feature_keys}
    x = keras.layers.Concatenate()(list(inputs.values()))

    units = hp.Int('units', min_value=128, max_value=512, step=32)
    x = keras.layers.Dense(units=units, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(units=units // 2, activation='relu')(x)

    outputs = keras.layers.Dense(4, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
    )
    return model


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """
    TFX Tuner entrypoint.
    Returns a TunerFnResult object with tuner and fit_kwargs.
    """
    tft_out = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_ds = _input_fn(fn_args.train_files, tft_out, shuffle=True)
    eval_ds = _input_fn(fn_args.eval_files, tft_out, shuffle=False)

    tuner = RandomSearch(
        hypermodel=lambda hp: model_builder(hp, fn_args.transform_graph_path),
        objective='val_sparse_categorical_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory=fn_args.working_dir,
        project_name='model_tuning',
        overwrite=True,
    )

    fit_kwargs = {
        'x': train_ds,
        'validation_data': eval_ds,
        'steps_per_epoch': fn_args.train_steps,
        'validation_steps': fn_args.eval_steps,
        'epochs': 10,
    }

    return TunerFnResult(tuner=tuner, fit_kwargs=fit_kwargs)


def run_fn(fn_args: FnArgs):
    """
    TFX Trainer entrypoint.
    Trains the model with best hyperparameters from tuner and saves it.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_ds = _input_fn(fn_args.train_files, tf_transform_output, shuffle=True)
    eval_ds = _input_fn(fn_args.eval_files, tf_transform_output, shuffle=False)

    if isinstance(fn_args.hyperparameters, dict):
        best_hp = HyperParameters.from_config(fn_args.hyperparameters)
    else:
        best_hp = HyperParameters.from_config(
            json.loads(fn_args.hyperparameters)
        )

    model = model_builder(best_hp, fn_args.transform_graph_path)

    model.fit(
        train_ds,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_ds,
        validation_steps=fn_args.eval_steps,
        epochs=25,
    )

    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures={
            'serving_default': _get_serve_tf_examples_fn(
                model, tf_transform_output
            )
        },
    )