#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import argparse


def getParserRNN(parser):
    parser.add_argument(
        "--training_loss",
        help="training_loss",
        type=str,
        required=False,
        default="mse",
    )
    parser.add_argument(
        "--precision",
        help="precision",
        type=str,
        required=False,
        default="double",
    )

    parser.add_argument("--noise_level",
                        help="noise_level",
                        type=float,
                        required=False,
                        default=0.0)

    parser.add_argument(
        "--train_residual_AE",
        help="train the auxiliary autoencoder",
        type=float,
        required=False,
        default=0,
    )

    parser.add_argument(
        "--latent_forecasting_loss",
        help="loss of dynamic consistency in the latent dynamics",
        type=int,
        default=1,
        required=False,
    )
    parser.add_argument(
        "--output_forecasting_loss",
        help="loss of RNN forecasting (if 0, autoencoder mode)",
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
    parser.add_argument("--zoneout_keep_prob",
                        help="zoneout_keep_prob",
                        type=float,
                        required=False,
                        default=1)
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
    parser.add_argument("--retrain", help="retrain", type=int, required=False)

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

    parser.add_argument("--random_seed_in_name",
                        help="random_seed_in_name",
                        type=int,
                        default=0,
                        required=False)

    parser.add_argument(
        "--latent_space_scaler",
        help="latent_space_scaler, MinMaxZeroOne, Standard",
        type=str,
        required=False,
        default="MinMaxZeroOne",
    )

    return parser
