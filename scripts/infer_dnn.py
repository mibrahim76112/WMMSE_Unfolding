import os
import numpy as np
import tensorflow as tf

from config import ExperimentConfig
from models.dnn_baseline import build_dnn_baseline
from utils import compute_channel

tf1 = tf.compat.v1

def main():
    cfg = ExperimentConfig()
    tf1.disable_eager_execution()

    g = build_dnn_baseline(cfg)
    saver = tf1.train.Saver()

    ckpt = os.environ.get("CKPT_PATH", "/content/checkpoints_dnn/dnn_baseline.ckpt")

    # make one test batch
    batch_ch = []
    for _ in range(cfg.nr_of_samples_per_batch):
        ch_nn, _, _, _ = compute_channel(
            cfg.nr_of_BS_antennas, cfg.nr_of_users, cfg.total_power,
            cfg.path_loss_option, cfg.path_loss_range
        )
        batch_ch.append(ch_nn)

    with tf1.Session() as sess:
        sess.run(tf1.global_variables_initializer())
        saver.restore(sess, ckpt)

        prec = sess.run(
            g["precoder"],
            feed_dict={g["channel_input"]: batch_ch, g["user_weights"]: cfg.user_weights_batch()}
        )
        print("precoder shape:", prec.shape)

if __name__ == "__main__":
    main()
