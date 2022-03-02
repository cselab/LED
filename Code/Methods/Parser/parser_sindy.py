#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import argparse


def getParserSINDy(parser):
    """ SINDY MODEL PARSER """
    parser.add_argument(
        "--sindy_integrator_type",
        help="sindy_integrator_type, discrete or continuous",
        type=str,
        required=False,
        default="continuous",
    )
    parser.add_argument(
        "--sindy_degree",
        help="sindy_degree, order of polynomial approximation",
        type=int,
        required=False,
        default=5,
    )
    parser.add_argument(
        "--sindy_threshold",
        help="sindy_threshold",
        type=float,
        required=False,
        default=0.001,
    )
    parser.add_argument(
        "--sindy_library",
        help="sindy_library, fourier, poly",
        type=str,
        required=False,
        default="fourier",
    )
    parser.add_argument(
        "--sindy_interp_factor",
        help="sindy_interp_factor",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--sindy_smoother_window_size",
        help="sindy_smoother_window_size",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--sindy_smoother_polyorder",
        help="sindy_smoother_polyorder",
        type=int,
        required=False,
        default=3,
    )

    return parser
