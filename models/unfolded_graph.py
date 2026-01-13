
import tensorflow as tf
from typing import Dict, Any, Tuple
from config import ExperimentConfig
from utils import compute_WSR_nn, PGD_step

tf1 = tf.compat.v1

def build_unfolded_graph(cfg: ExperimentConfig):
    """Build the TF1-style graph for the deep-unfolded WMMSE from the notebook.

    Returns a dict with placeholders and key tensors/ops:
      - channel_input, initial_tp
      - final_precoder
      - WSR (sum across layers), WSR_final (final layer averaged over batch)
      - optimizer op
      - all_step_size1..4 tensors
    """
    tf1.reset_default_graph()

    channel_input = tf1.placeholder(tf.float64, shape=None, name='channel_input')
    initial_tp = tf1.placeholder(tf.float64, shape=None, name='initial_transmitter_precoder')
    initial_transmitter_precoder = initial_tp

    # Step size initializers: replicate notebook behaviour but safe for L>1
    step_size1_init = [1.0] * cfg.nr_of_iterations_nn
    step_size2_init = [1.0] * cfg.nr_of_iterations_nn
    step_size3_init = [1.0] * cfg.nr_of_iterations_nn
    step_size4_init = [1.0] * cfg.nr_of_iterations_nn

    all_step_size1_temp = []
    all_step_size2_temp = []
    all_step_size3_temp = []
    all_step_size4_temp = []

    profit = []

    # user weights placeholder-like constant (passed from python as numpy in feed_dict)
    # Here we assume user_weights is provided externally, consistent with the notebook.
    user_weights = tf1.placeholder(tf.float64, shape=None, name='user_weights')

    for loop in range(0, cfg.nr_of_iterations_nn):

        # Compute per-sample interference terms (kept close to notebook structure)
        user_interference2 = []
        for batch_index in range(cfg.nr_of_samples_per_batch):
            user_interference_single = []
            for i in range(cfg.nr_of_users):
                temp = 0.0
                for j in range(cfg.nr_of_users):
                    temp = temp + tf.reduce_sum(
                        (tf.matmul(tf.transpose(channel_input[batch_index, i, :, :]),
                                   initial_transmitter_precoder[batch_index, j, :, :])) ** 2
                    )
                user_interference_single.append(temp + cfg.noise_power)
            user_interference2.append(user_interference_single)

        user_interference2 = tf.stack(user_interference2)  # shape: (B, K)

        user_interference_exp2 = tf.tile(
            tf.expand_dims(tf.tile(tf.expand_dims(user_interference2, -1), [1, 1, 2]), -1),
            [1, 1, 1, 1]
        )

        receiver_precoder_temp = tf.matmul(
            tf.transpose(channel_input, perm=[0, 1, 3, 2]),
            initial_transmitter_precoder
        )
        receiver_precoder = tf.divide(receiver_precoder_temp, user_interference_exp2)

        # Optimize the mmse weights
        self_interference = tf.reduce_sum(
            (tf.matmul(tf.transpose(channel_input, perm=[0, 1, 3, 2]),
                       initial_transmitter_precoder)) ** 2,
            axis=2
        )

        inter_user_interference_total = []
        for batch_index in range(cfg.nr_of_samples_per_batch):
            inter_user_interference_temp = []
            for i in range(cfg.nr_of_users):
                temp = 0.0
                for j in range(cfg.nr_of_users):
                    if j != i:
                        temp = temp + tf.reduce_sum(
                            (tf.matmul(tf.transpose(channel_input[batch_index, i, :, :]),
                                       initial_transmitter_precoder[batch_index, j, :, :])) ** 2
                        )
                inter_user_interference_temp.append(temp + cfg.noise_power)
            inter_user_interference = tf.reshape(
                tf.stack(inter_user_interference_temp),
                (cfg.nr_of_users, 1)
            )
            inter_user_interference_total.append(inter_user_interference)

        inter_user_interference_total = tf.stack(inter_user_interference_total)  # (B, K, 1)
        mse_weights = tf.divide(self_interference, inter_user_interference_total) + 1.0

        # Optimize the transmitter precoder through PGD (4 steps)
        transmitter_precoder1, step_size1 = PGD_step(
            step_size1_init[loop], 'PGD_step1',
            mse_weights, user_weights, receiver_precoder, channel_input,
            initial_transmitter_precoder, cfg.total_power,
            cfg.nr_of_users, cfg.nr_of_BS_antennas, cfg.nr_of_samples_per_batch
        )

 
        transmitter_precoder2, step_size2 = PGD_step(
            step_size2_init[loop], 'PGD_step2',
            mse_weights, user_weights, receiver_precoder, channel_input,
            transmitter_precoder1, cfg.total_power,
            cfg.nr_of_users, cfg.nr_of_BS_antennas, cfg.nr_of_samples_per_batch
        )
        transmitter_precoder3, step_size3 = PGD_step(
            step_size3_init[loop], 'PGD_step3',
            mse_weights, user_weights, receiver_precoder, channel_input,
            transmitter_precoder2, cfg.total_power,
            cfg.nr_of_users, cfg.nr_of_BS_antennas, cfg.nr_of_samples_per_batch
        )
        transmitter_precoder, step_size4 = PGD_step(
            step_size4_init[loop], 'PGD_step4',
            mse_weights, user_weights, receiver_precoder, channel_input,
            transmitter_precoder3, cfg.total_power,
            cfg.nr_of_users, cfg.nr_of_BS_antennas, cfg.nr_of_samples_per_batch
        )


        initial_transmitter_precoder = transmitter_precoder

        all_step_size1_temp.append(step_size1)
        all_step_size2_temp.append(step_size2)
        all_step_size3_temp.append(step_size3)
        all_step_size4_temp.append(step_size4)

      #  profit.append(compute_WSR_nn(user_weights, channel_input, initial_transmitter_precoder, cfg.noise_power, cfg.nr_of_users))
        profit.append(compute_WSR_nn(
            user_weights, channel_input, initial_transmitter_precoder,
            cfg.noise_power, cfg.nr_of_users, cfg.nr_of_samples_per_batch
        ))

    all_step_size1 = tf.stack(all_step_size1_temp)
    all_step_size2 = tf.stack(all_step_size2_temp)
    all_step_size3 = tf.stack(all_step_size3_temp)
    all_step_size4 = tf.stack(all_step_size4_temp)

    final_precoder = initial_transmitter_precoder

    WSR = tf.reduce_sum(tf.stack(profit))
   # WSR_final = compute_WSR_nn(user_weights, channel_input, final_precoder, cfg.noise_power, cfg.nr_of_users) / cfg.nr_of_samples_per_batch
    WSR_final = compute_WSR_nn(
        user_weights, channel_input, final_precoder,
        cfg.noise_power, cfg.nr_of_users, cfg.nr_of_samples_per_batch
    ) / cfg.nr_of_samples_per_batch


    optimizer = tf1.train.AdamOptimizer(learning_rate=cfg.learning_rate).minimize(-WSR)

    return {
        "channel_input": channel_input,
        "initial_tp": initial_tp,
        "user_weights": user_weights,
        "final_precoder": final_precoder,
        "WSR": WSR,
        "WSR_final": WSR_final,
        "optimizer": optimizer,
        "all_step_size1": all_step_size1,
        "all_step_size2": all_step_size2,
        "all_step_size3": all_step_size3,
        "all_step_size4": all_step_size4,
    }
