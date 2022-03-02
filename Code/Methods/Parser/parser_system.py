#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import argparse


def getParserSystem(parser):

    parser.add_argument("--mode",
                        help="train, test, all",
                        type=str,
                        required=True)

    parser.add_argument("--system_name",
                        help="system_name",
                        type=str,
                        required=True)

    parser.add_argument("--input_dim",
                        help="input_dim",
                        type=int,
                        required=True)

    parser.add_argument("--write_to_log",
                        help="write_to_log",
                        type=int,
                        required=False,
                        default=0)

    parser.add_argument("--channels",
                        help="Channels in case input more than 1-D.",
                        type=int,
                        required=False,
                        default=0)

    parser.add_argument("--Dx",
                        help="Channel dimension 1. (last)",
                        type=int,
                        required=False,
                        default=0)

    parser.add_argument("--Dy",
                        help="Channel dimension 2. (second last)",
                        type=int,
                        required=False,
                        default=0)

    parser.add_argument("--Dz",
                        help="Channel dimension 3. (third last)",
                        type=int,
                        required=False,
                        default=0)

    parser.add_argument(
        "--num_test_ICS",
        help="num_test_ICS",
        type=int,
        required=False,
        default=0,
    )

    parser.add_argument("--prediction_horizon",
                        help="prediction_horizon",
                        type=int,
                        required=False,
                        default=0)

    parser.add_argument("--display_output",
                        help="control the verbosity level of output",
                        type=int,
                        required=False,
                        default=1)

    parser.add_argument("--random_seed",
                        help="random_seed",
                        type=int,
                        default=1,
                        required=False)

    parser.add_argument("--test_on_train",
                        help="test_on_train.",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--test_on_val",
                        help="test_on_val.",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--test_on_test",
                        help="test_on_test.",
                        type=int,
                        default=0,
                        required=False)

    parser.add_argument("--compute_spectrum",
                        help="compute_spectrum.",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--plot_state_distributions",
                        help="plot_state_distributions.",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--plot_system",
                        help="plot_system.",
                        type=int,
                        default=1,
                        required=False)
    parser.add_argument("--plot_errors_in_time",
                        help="plot_errors_in_time.",
                        type=int,
                        default=1,
                        required=False)
    parser.add_argument("--plot_testing_ics_examples",
                        help="plot_testing_ics_examples.",
                        type=int,
                        default=1,
                        required=False)
    parser.add_argument("--plot_latent_dynamics",
                        help="plot_latent_dynamics",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument(
        "--plotting",
        help="plotting",
        type=int,
        default=1,
        required=False,
    )

    parser.add_argument("--truncate_timesteps",
                        help="truncate_timesteps",
                        type=int,
                        default=0,
                        required=False)

    parser.add_argument("--truncate_data_batches",
                        help="truncate_data_batches",
                        type=int,
                        default=0,
                        required=False)

    parser.add_argument("--save_format",
                        help="save format, hickle or pickle",
                        type=str,
                        default="pickle",
                        required=False)
    parser.add_argument("--make_videos",
                        help="make_videos.",
                        type=int,
                        default=0,
                        required=False)

    parser.add_argument(
        "--debug",
        help="debug",
        type=int,
        required=False,
        default=0,
    )

    parser.add_argument(
        "--gpu_monitor_every",
        help=
        "gpu_monitor_every in seconds (separate thread monitoring the memory)",
        type=int,
        required=False,
        default=10,
    )
    return parser
