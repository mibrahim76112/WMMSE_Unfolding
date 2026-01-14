# scripts/eval_compare_hist.py
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config import ExperimentConfig
from models.unfolded_graph import build_unfolded_graph
from models.dnn_baseline import build_dnn_baseline
from utils import compute_channel, compute_WSR_nn

tf1 = tf.compat.v1

def sample_batch(cfg, path_loss_option=None):
    if path_loss_option is None:
        path_loss_option = cfg.path_loss_option

    batch_ch = []
    batch_init = []
    for _ in range(cfg.nr_of_samples_per_batch):
        ch_nn, init_tp, _, _ = compute_channel(
            cfg.nr_of_BS_antennas, cfg.nr_of_users, cfg.total_power,
            path_loss_option, cfg.path_loss_range
        )
        batch_ch.append(ch_nn)
        batch_init.append(init_tp)
    return batch_ch, batch_init

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_unfolded", required=True)
    ap.add_argument("--ckpt_dnn", required=True)
    ap.add_argument("--n_batches", type=int, default=200)
    ap.add_argument("--path_loss_option", type=int, default=0)
    ap.add_argument("--out_png", default="hist_compare.png")
    args = ap.parse_args()

    cfg = ExperimentConfig()
    tf1.disable_eager_execution()

    # Build unfolded graph
    g_unf = build_unfolded_graph(cfg)
    WSR_unf = compute_WSR_nn(
        g_unf["user_weights"], g_unf["channel_input"], g_unf["final_precoder"],
        cfg.noise_power, cfg.nr_of_users, cfg.nr_of_samples_per_batch
    ) / cfg.nr_of_samples_per_batch
    saver_unf = tf1.train.Saver(var_list=tf1.get_collection(tf1.GraphKeys.GLOBAL_VARIABLES))

    # Build DNN graph in a separate graph
    g2 = tf1.Graph()
    with g2.as_default():
        g_dnn = build_dnn_baseline(cfg)
        WSR_dnn = compute_WSR_nn(
            g_dnn["user_weights"], g_dnn["channel_input"], g_dnn["precoder"],
            cfg.noise_power, cfg.nr_of_users, cfg.nr_of_samples_per_batch
        ) / cfg.nr_of_samples_per_batch
        saver_dnn = tf1.train.Saver()

    wsr_unf_all = []
    wsr_dnn_all = []

    # Eval unfolded
    with tf1.Session() as sess:
        sess.run(tf1.global_variables_initializer())
        saver_unf.restore(sess, args.ckpt_unfolded)

        for _ in range(args.n_batches):
            batch_ch, batch_init = sample_batch(cfg, bool(args.path_loss_option))
            feed = {
                g_unf["channel_input"]: batch_ch,
                g_unf["initial_tp"]: batch_init,
                g_unf["user_weights"]: cfg.user_weights_batch(),
            }
            wsr_unf_all.append(sess.run(WSR_unf, feed_dict=feed))

    # Eval DNN
    with tf1.Session(graph=g2) as sess:
        sess.run(tf1.global_variables_initializer())
        saver_dnn.restore(sess, args.ckpt_dnn)

        for _ in range(args.n_batches):
            batch_ch, _ = sample_batch(cfg, bool(args.path_loss_option))
            feed = {
                g_dnn["channel_input"]: batch_ch,
                g_dnn["user_weights"]: cfg.user_weights_batch(),
            }
            wsr_dnn_all.append(sess.run(WSR_dnn, feed_dict=feed))

    wsr_unf_all = np.array(wsr_unf_all)
    wsr_dnn_all = np.array(wsr_dnn_all)

    plt.figure()
    plt.hist(wsr_unf_all, bins=60, alpha=0.7, label="Deep unfolding")
    plt.hist(wsr_dnn_all, bins=60, alpha=0.7, label="DNN baseline")
    plt.xlabel("WSR (bits per channel use)")
    plt.ylabel("Number of batches")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print("Saved:", args.out_png)

if __name__ == "__main__":
    g_unf_graph = tf1.Graph()
    g_dnn_graph = tf1.Graph()

    main()
