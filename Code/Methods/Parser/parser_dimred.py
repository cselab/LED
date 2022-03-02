#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import argparse


def getParserDimRed(parser):

    parser.add_argument("--dimred_method",
                        help="dimensionality reduction method",
                        type=str,
                        required=True,
                        default="pca")

    parser.add_argument(
        "--diffmaps_weight",
        help=
        "hyperparameter of diffusion maps that multiplies the median of the data",
        type=float,
        required=False,
        default=5.0,
    )
    parser.add_argument(
        "--diffmaps_num_neighbors",
        help="number of neighbors considered during lifting",
        type=int,
        required=False,
        default=10,
    )

    parser.add_argument("--latent_state_dim",
                        help="latent_state_dim",
                        type=int,
                        required=False,
                        default=0)

    parser.add_argument("--scaler", help="scaler", type=str, required=True)

    return parser
