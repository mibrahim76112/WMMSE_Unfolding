
import os
import numpy as np
import tensorflow as tf

from config import ExperimentConfig
from models.unfolded_graph import build_unfolded_graph
from utils import compute_channel

tf1 = tf.compat.v1

def main():
    cfg = ExperimentConfig()
    tf1.disable_eager_execution()

    graph = build_unfolded_graph(cfg)

    ckpt = os.environ.get("CKPT_PATH", os.path.join(os.getcwd(), "checkpoints", "unfolded_wmmse.ckpt"))
    saver = tf1.train.Saver()

    # Build a small test batch
    np.random.seed(cfg.test_seed)
    batch_for_testing = []
    initial_tp_batch = []
    for _ in range(cfg.nr_of_samples_per_batch):
        channel_nn, init_tp, _, _ = compute_channel(
            cfg.nr_of_BS_antennas, cfg.nr_of_users, cfg.total_power,
            cfg.path_loss_option, cfg.path_loss_range
        )
        batch_for_testing.append(channel_nn)
        initial_tp_batch.append(init_tp)

    feed = {
        graph["channel_input"]: batch_for_testing,
        graph["initial_tp"]: initial_tp_batch,
        graph["user_weights"]: cfg.user_weights_batch(),
    }

    with tf1.Session() as sess:
        sess.run(tf1.global_variables_initializer())
        saver.restore(sess, ckpt)

        wsr = sess.run(graph["WSR_final"], feed_dict=feed)
        precoder = sess.run(graph["final_precoder"], feed_dict=feed)

    print("Restored checkpoint:", ckpt)
    print("WSR_final:", float(wsr))
    print("final_precoder shape:", np.array(precoder).shape)

if __name__ == "__main__":
    main()
