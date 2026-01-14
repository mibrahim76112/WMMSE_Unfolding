# scripts/train_dnn.py
import os, time
import numpy as np
import tensorflow as tf

from config import ExperimentConfig
from models.dnn_baseline import build_dnn_baseline
from utils import compute_channel, compute_WSR_nn

tf1 = tf.compat.v1

def main():
    cfg = ExperimentConfig()
    tf1.disable_eager_execution()

    graph = build_dnn_baseline(cfg)

    # WSR objective (unsupervised)
    WSR = compute_WSR_nn(
        graph["user_weights"],
        graph["channel_input"],
        graph["precoder"],
        cfg.noise_power,
        cfg.nr_of_users,
        cfg.nr_of_samples_per_batch
    )
    loss = -WSR
    opt = tf1.train.AdamOptimizer(cfg.learning_rate).minimize(loss)

    out_dir = os.environ.get("OUT_DIR", os.path.join(os.getcwd(), "checkpoints_dnn"))
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "dnn_baseline.ckpt")
    saver = tf1.train.Saver(max_to_keep=3)

    np.random.seed(cfg.train_seed)

    with tf1.Session() as sess:
        sess.run(tf1.global_variables_initializer())
        t0 = time.time()

        for it in range(cfg.nr_of_batches_training):
            batch_ch = []
            for _ in range(cfg.nr_of_samples_per_batch):
                channel_nn, _, _, _ = compute_channel(
                    cfg.nr_of_BS_antennas, cfg.nr_of_users, cfg.total_power,
                    cfg.path_loss_option, cfg.path_loss_range
                )
                batch_ch.append(channel_nn)

            feed = {
                graph["channel_input"]: batch_ch,
                graph["user_weights"]: cfg.user_weights_batch(),
            }

            sess.run(opt, feed_dict=feed)

            if (it + 1) % 500 == 0:
                wsr_val = sess.run(WSR, feed_dict=feed)
                print(f"iter {it+1}/{cfg.nr_of_batches_training} | WSR={wsr_val/cfg.nr_of_samples_per_batch:.4f}")

        print("Training took:", time.time() - t0)
        saver.save(sess, ckpt_path)
        print("Saved checkpoint to:", ckpt_path)

if __name__ == "__main__":
    main()
