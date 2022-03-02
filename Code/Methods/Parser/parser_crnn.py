#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import argparse


def getCRNNParser(parser):

    parser.add_argument(
        "--training_loss",
        help="training_loss",
        type=str,
        required=False,
        default="mse",
    )

    parser.add_argument("--precision",
                        help="precision",
                        type=str,
                        required=False,
                        default="double")

    parser.add_argument("--AE_convolutional",
                        help="AE_convolutional",
                        default=0,
                        type=int,
                        required=False)

    parser.add_argument("--RNN_convolutional",
                        help="RNN_convolutional",
                        default=0,
                        type=int,
                        required=False)
    parser.add_argument("--RNN_kernel_size",
                        help="RNN_kernel_size",
                        default=0,
                        type=int,
                        required=False)
    parser.add_argument("--RNN_trainable_init_hidden_state",
                        help="RNN_trainable_init_hidden_state",
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument("--RNN_statefull",
                        help="RNN_statefull",
                        type=int,
                        required=False,
                        default=1)

    parser.add_argument(
        "--output_forecasting_loss",
        help="loss of RNN forecasting (if 0, autoencoder mode)",
        type=int,
        default=1,
        required=False)
    parser.add_argument(
        "--latent_forecasting_loss",
        help="loss of dynamic consistency in the latent dynamics",
        type=int,
        default=0,
        required=False)
    parser.add_argument("--reconstruction_loss",
                        help="reconstruction_loss",
                        type=int,
                        default=0,
                        required=False)

    parser.add_argument("--RNN_cell_type",
                        help="type of the rnn cell",
                        type=str,
                        required=False,
                        default="lstm")
    parser.add_argument("--RNN_activation_str",
                        help="RNN_activation_str",
                        type=str,
                        required=False,
                        default="tanh")
    parser.add_argument("--RNN_activation_str_output",
                        help="RNN_activation_str_output",
                        type=str,
                        required=False,
                        default="tanh")
    parser.add_argument('--RNN_layers_size',
                        type=int,
                        help='size of the RNN layers',
                        required=False,
                        default=0)
    parser.add_argument('--RNN_layers_num',
                        type=int,
                        help='number of the RNN layers',
                        required=False,
                        default=0)

    parser.add_argument('--AE_layers_size',
                        type=int,
                        help='The size of the autoencoder layers',
                        required=False,
                        default=0)
    parser.add_argument('--AE_layers_num',
                        type=int,
                        help='The number of the autoencoder layers',
                        required=False,
                        default=0)
    parser.add_argument("--activation_str_general",
                        help="Activation of Autoencoder/MLP/MDN layers",
                        type=str,
                        required=False,
                        default="selu")
    parser.add_argument("--activation_str_output",
                        help="Activation of OUTPUT in Autoencoder layers",
                        type=str,
                        required=False,
                        default="identity")
    parser.add_argument("--AE_batch_norm",
                        help="batch normalization",
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument("--AE_batch_norm_affine",
                        help="AE_batch_norm_affine",
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument("--AE_conv_transpose",
                        help="transpose convolutions",
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument("--AE_pool_type",
                        help="polling operation",
                        type=str,
                        required=False,
                        default="avg")
    parser.add_argument("--AE_conv_architecture",
                        help="AE_conv_architecture",
                        type=str,
                        required=False,
                        default=None)
    parser.add_argument("--AE_size_factor",
                        help="AE_size_factor",
                        type=int,
                        required=False,
                        default=1)
    parser.add_argument("--AE_interp_subsampling_input",
                        help="AE_interp_subsampling_input",
                        type=int,
                        required=False,
                        default=1)

    parser.add_argument("--latent_state_dim",
                        help="latent_state_dim",
                        type=int,
                        required=False,
                        default=0)

    parser.add_argument("--zoneout_keep_prob",
                        help="zoneout_keep_prob",
                        type=float,
                        required=False,
                        default=1)
    parser.add_argument("--dropout_keep_prob",
                        help="dropout_keep_prob",
                        type=float,
                        required=False,
                        default=1)

    parser.add_argument("--noise_level",
                        help="noise_level",
                        type=float,
                        required=False,
                        default=0.0)
    parser.add_argument("--noise_level_AE",
                        help="noise_level_AE",
                        type=float,
                        required=False,
                        default=0.0)

    parser.add_argument("--sequence_length",
                        help="sequence_length",
                        type=int,
                        required=True)
    parser.add_argument(
        "--prediction_length",
        help="prediction_length",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--n_warmup_train",
        help="n_warmup_train",
        type=int,
        required=False,
        default=0,
    )

    parser.add_argument("--scaler", help="scaler", type=str, required=True)

    parser.add_argument("--learning_rate",
                        help="learning_rate",
                        type=float,
                        required=True)
    parser.add_argument(
        "--lr_reduction_factor",
        help="lr_reduction_factor",
        type=float,
        required=False,
        default=0.5,
    )
    parser.add_argument("--weight_decay",
                        help="weight_decay",
                        type=float,
                        required=False,
                        default=0.0)
    parser.add_argument("--batch_size",
                        help="batch_size",
                        type=int,
                        required=True)
    parser.add_argument("--overfitting_patience",
                        help="overfitting_patience",
                        type=int,
                        required=True)
    parser.add_argument("--max_epochs",
                        help="max_epochs",
                        type=int,
                        required=True)
    parser.add_argument("--max_rounds",
                        help="max_rounds",
                        type=int,
                        required=True)
    parser.add_argument("--retrain",
                        help="retrain",
                        type=int,
                        default=0,
                        required=False)

    parser.add_argument("--reference_train_time",
                        help="The reference train time in hours",
                        type=float,
                        default=24.0,
                        required=False)
    parser.add_argument(
        "--buffer_train_time",
        help="The buffer train time to save the model in hours",
        type=float,
        default=0.5,
        required=False)

    parser.add_argument("--optimizer_str",
                        help="adam or sgd with cyclical learning rate",
                        type=str,
                        default="adam",
                        required=False)
    parser.add_argument(
        "--iterative_propagation_during_training_is_latent",
        help=
        "Unplug the encoder and propagate only the latent state during iterative forecasting (only used for training).",
        type=int,
        default=0,
        required=False)

    parser.add_argument("--train_AE_only",
                        help="train_AE_only.",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--train_RNN_only",
                        help="train_RNN_only.",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--load_trained_AE",
                        help="load_trained_AE.",
                        type=int,
                        default=0,
                        required=False)

    parser.add_argument("--cudnn_benchmark",
                        help="cudnn_benchmark",
                        type=int,
                        default=0,
                        required=False)

    parser.add_argument("--learning_rate_in_name",
                        help="learning_rate_in_name",
                        type=int,
                        default=1,
                        required=False)

    parser.add_argument("--iterative_loss_schedule_and_gradient",
                        help="iterative_loss_schedule_and_gradient",
                        type=str,
                        default="none",
                        required=False)
    parser.add_argument(
        "--iterative_loss_validation",
        help="iterative_loss_validation",
        type=int,
        default=0,
        required=False,
    )

    parser.add_argument("--random_seed_in_name",
                        help="random_seed_in_name",
                        type=int,
                        default=0,
                        required=False)

    parser.add_argument("--random_seed_in_AE_name",
                        help="random_seed_in_AE_name",
                        type=int,
                        default=0,
                        required=False)

    parser.add_argument("--learning_rate_AE",
                        help="learning_rate_AE",
                        type=float,
                        required=False)

    parser.add_argument(
        "--multinode",
        help="Training on multiple nodes",
        type=int,
        required=False,
        default=0,
    )

    parser.add_argument(
        "--hvd_compression",
        help="Horovod compression algorithm [fp16 or none]",
        type=int,
        required=False,
        default=0,
    )

    parser.add_argument(
        "--hvd_adasum",
        help="whether to use AdaSum or not",
        type=int,
        required=False,
        default=0,
    )

    parser.add_argument(
        "--beta_vae",
        help="Beta variational layer in the latent space.",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--beta_vae_weight_max",
        help="Weight of Beta-VAE (max towards end of training)",
        type=float,
        required=False,
        default=0.0,
    )

    parser.add_argument(
        "--c1_latent_smoothness_loss",
        help="c1_latent_smoothness_loss",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--c1_latent_smoothness_loss_factor",
        help="c1_latent_smoothness_loss_factor",
        type=float,
        required=False,
        default=0.0,
    )
    parser.add_argument(
        "--latent_space_scaler",
        help="latent_space_scaler, MinMaxZeroOne, Standard",
        type=str,
        required=False,
        default="MinMaxZeroOne",
    )

    return parser
