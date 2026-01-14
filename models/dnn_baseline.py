# models/dnn_baseline.py
import tensorflow as tf
tf1 = tf.compat.v1

def build_dnn_baseline(cfg):
    tf1.reset_default_graph()

    channel_input = tf1.placeholder(tf.float64, shape=None, name="channel_input")
    user_weights  = tf1.placeholder(tf.float64, shape=None, name="user_weights")

    x = tf.reshape(channel_input, [cfg.nr_of_samples_per_batch, -1])

    with tf1.variable_scope("dnn_baseline"):
        h = x
        for li, width in enumerate(cfg.dnn_hidden):
            h = tf1.keras.layers.Dense(
                width, activation="relu", dtype="float64", name=f"fc{li+1}"
            )(h)

        out_dim = cfg.nr_of_users * (2 * cfg.nr_of_BS_antennas) * 1
        y = tf1.keras.layers.Dense(
            out_dim, activation=None, dtype="float64", name="out"
        )(h)

    precoder = tf.reshape(
        y,
        [cfg.nr_of_samples_per_batch, cfg.nr_of_users, 2 * cfg.nr_of_BS_antennas, 1],
        name="precoder_raw"
    )

    power = tf.reduce_sum(tf.square(precoder), axis=[1,2,3], keepdims=True) + 1e-12
    scale = tf.sqrt(tf.cast(cfg.total_power, tf.float64) / power)
    precoder = tf.identity(precoder * scale, name="precoder")

    return {
        "channel_input": channel_input,
        "user_weights": user_weights,
        "precoder": precoder,
    }
