#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import argparse


def getParserRC(parser):
    parser.add_argument(
        "--rc_solver",
        help="rc_solver",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--rc_approx_reservoir_size",
        help="rc_approx_reservoir_size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--rc_degree",
        help="rc_degree",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--rc_radius",
        help="rc_radius",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--rc_sigma_input",
        help="rc_sigma_input",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--rc_dynamics_length",
        help="rc_dynamics_length",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--rc_regularization",
        help="rc_regularization",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--rc_noise_level_per_mill",
        help="rc_noise_level_per_mill",
        type=int,
        required=True,
    )

    return parser
