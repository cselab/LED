#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import argparse


def getParserTesting(parser):

    parser.add_argument(
        "--n_warmup",
        help="n_warmup",
        type=int,
        required=False,
        default=None,
    )

    parser.add_argument(
        "--iterative_state_forecasting",
        help=
        "to test the model in iterative forecasting, propagating the output state of the model.",
        type=int,
        default=0,
        required=False)

    parser.add_argument(
        "--iterative_latent_forecasting",
        help=
        "to test the model in iterative forecasting, propagating the latent space od the model.",
        type=int,
        default=0,
        required=False)

    parser.add_argument("--teacher_forcing_forecasting",
                        help="to test the the model in teacher forcing.",
                        type=int,
                        default=0,
                        required=False)

    return parser
