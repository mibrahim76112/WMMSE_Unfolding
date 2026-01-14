# models/dnn_baseline.py
import tensorflow as tf
tf1 = tf.compat.v1

def build_dnn_baseline(cfg):
    #tf1.reset_default_graph()

    M2 = 2 * cfg.nr_of_BS_antennas
    K  = cfg.nr_of_users
    B  = cfg.nr_of_samples_per_batch

    channel_input = tf1.placeholder(
        tf.float64,
        shape=[B, K, M2, 2],
        name="channel_input"
    )
    user_weights = tf1.placeholder(
        tf.float64,
        shape=[B, K, 1],
        name="user_weights"
    )

    in_dim = K * M2 * 2
    x = tf.reshape(channel_input, [B, in_dim])

    with tf1.variable_scope("dnn_baseline"):
        h = x
        for li, width in enumerate(cfg.dnn_hidden):
            h = tf1.keras.layers.Dense(
                width, activation="relu", dtype="float64", name=f"fc{li+1}"
            )(h)

        out_dim = K * M2 * 1
        y = tf1.keras.layers.Dense(
            out_dim, activation=None, dtype="float64", name="out"
        )(h)

    precoder = tf.reshape(y, [B, K, M2, 1], name="precoder_raw")

    # enforce total power constraint per sample
    power = tf.reduce_sum(tf.square(precoder), axis=[1,2,3], keepdims=True) + 1e-12
    scale = tf.sqrt(tf.cast(cfg.total_power, tf.float64) / power)
    precoder = tf.identity(precoder * scale, name="precoder")

    return {
        "channel_input": channel_input,
        "user_weights": user_weights,
        "precoder": precoder,
    }
