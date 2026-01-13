
import os
import time
import numpy as np
import tensorflow as tf

from wmmse_unfolded_project.config import ExperimentConfig
from wmmse_unfolded_project.models.unfolded_graph import build_unfolded_graph
from wmmse_unfolded_project.utils import (
    compute_channel,
    run_WMMSE,
    zero_forcing,
    regularized_zero_forcing,
    compute_weighted_sum_rate,
)

tf1 = tf.compat.v1

def main():
    cfg = ExperimentConfig()

    # TF1 graph mode
    tf1.disable_eager_execution()

    graph = build_unfolded_graph(cfg)

    out_dir = os.environ.get("OUT_DIR", os.path.join(os.getcwd(), "checkpoints"))
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "unfolded_wmmse.ckpt")

    saver = tf1.train.Saver(max_to_keep=3)

    WSR_WMMSE = []
    WSR_ZF = []
    WSR_RZF = []
    WSR_nn = []
    training_loss = []

    np.random.seed(cfg.train_seed)

    with tf1.Session() as sess:
        print("start of session")
        start_of_time = time.time()
        sess.run(tf1.global_variables_initializer())

        # Training loop
        for i in range(cfg.nr_of_batches_training):
            batch_for_training = []
            initial_transmitter_precoder_batch = []

            for _ in range(cfg.nr_of_samples_per_batch):
                channel_realization_nn, init_transmitter_precoder, _, _ = compute_channel(
                    cfg.nr_of_BS_antennas, cfg.nr_of_users, cfg.total_power,
                    cfg.path_loss_option, cfg.path_loss_range
                )
                batch_for_training.append(channel_realization_nn)
                initial_transmitter_precoder_batch.append(init_transmitter_precoder)

            feed = {
                graph["channel_input"]: batch_for_training,
                graph["initial_tp"]: initial_transmitter_precoder_batch,
                graph["user_weights"]: cfg.user_weights_batch(),
            }

            sess.run(graph["optimizer"], feed_dict=feed)
            training_loss.append(-1.0 * sess.run(graph["WSR"], feed_dict=feed))

            if (i + 1) % 500 == 0:
                print(f"batch {i+1}/{cfg.nr_of_batches_training} | loss={training_loss[-1]:.4f}")

        print("step size1", sess.run(graph["all_step_size1"], feed_dict={graph["user_weights"]: cfg.user_weights_batch(),
                                                                        graph["channel_input"]: batch_for_training,
                                                                        graph["initial_tp"]: initial_transmitter_precoder_batch}))
        print("step size2", sess.run(graph["all_step_size2"], feed_dict={graph["user_weights"]: cfg.user_weights_batch(),
                                                                        graph["channel_input"]: batch_for_training,
                                                                        graph["initial_tp"]: initial_transmitter_precoder_batch}))
        print("step size3", sess.run(graph["all_step_size3"], feed_dict={graph["user_weights"]: cfg.user_weights_batch(),
                                                                        graph["channel_input"]: batch_for_training,
                                                                        graph["initial_tp"]: initial_transmitter_precoder_batch}))
        print("step size4", sess.run(graph["all_step_size4"], feed_dict={graph["user_weights"]: cfg.user_weights_batch(),
                                                                        graph["channel_input"]: batch_for_training,
                                                                        graph["initial_tp"]: initial_transmitter_precoder_batch}))

        print("Training took:", time.time() - start_of_time)

        # Save checkpoint (weights)
        saver.save(sess, ckpt_path)
        print("Saved checkpoint to:", ckpt_path)

        # Testing
        np.random.seed(cfg.test_seed)

        for _ in range(cfg.nr_of_batches_test):
            batch_for_testing = []
            initial_transmitter_precoder_batch = []

            WSR_WMMSE_batch = 0.0
            WSR_ZF_batch = 0.0
            WSR_RZF_batch = 0.0

            for _ in range(cfg.nr_of_samples_per_batch):
                channel_realization_nn, init_transmitter_precoder, channel_realization_regular, reg_param = compute_channel(
                    cfg.nr_of_BS_antennas, cfg.nr_of_users, cfg.total_power,
                    cfg.path_loss_option, cfg.path_loss_range
                )

                # WMMSE baseline
                _, _, _, WSR_WMMSE_one = run_WMMSE(
                    cfg.epsilon, channel_realization_regular, list(cfg.scheduled_users),
                    cfg.total_power, cfg.noise_power, cfg.user_weights_regular(),
                    cfg.nr_of_iterations_wmmse - 1, log=False
                )
                WSR_WMMSE_batch += WSR_WMMSE_one

                # ZF baseline
                ZF_solution = zero_forcing(channel_realization_regular, cfg.total_power)
                WSR_ZF_batch += compute_weighted_sum_rate(
                    cfg.user_weights_regular(), channel_realization_regular,
                    ZF_solution, cfg.noise_power, list(cfg.scheduled_users)
                )

                # RZF baseline
                RZF_solution = regularized_zero_forcing(channel_realization_regular, cfg.total_power, reg_param, cfg.path_loss_option)
                WSR_RZF_batch += compute_weighted_sum_rate(
                    cfg.user_weights_regular(), channel_realization_regular,
                    RZF_solution, cfg.noise_power, list(cfg.scheduled_users)
                )

                batch_for_testing.append(channel_realization_nn)
                initial_transmitter_precoder_batch.append(init_transmitter_precoder)

            feed_test = {
                graph["channel_input"]: batch_for_testing,
                graph["initial_tp"]: initial_transmitter_precoder_batch,
                graph["user_weights"]: cfg.user_weights_batch(),
            }

            WSR_nn.append(sess.run(graph["WSR_final"], feed_dict=feed_test))
            WSR_WMMSE.append(WSR_WMMSE_batch / cfg.nr_of_samples_per_batch)
            WSR_ZF.append(WSR_ZF_batch / cfg.nr_of_samples_per_batch)
            WSR_RZF.append(WSR_RZF_batch / cfg.nr_of_samples_per_batch)

    print("The WSR achieved with the deep unfolded WMMSE algorithm is:", float(np.mean(WSR_nn)))
    print("The WSR achieved with the WMMSE algorithm is:", float(np.mean(WSR_WMMSE)))
    print("The WSR achieved with the zero forcing is:", float(np.mean(WSR_ZF)))
    print("The WSR achieved with the regularized zero forcing is:", float(np.mean(WSR_RZF)))

if __name__ == "__main__":
    main()
