#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import argparse
from . import parser_system
from . import parser_testing
from . import parser_dimred
from . import parser_rc
from . import parser_sindy
from . import parser_rnn
from . import parser_ae
from . import parser_crnn


def defineParser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Selection of the model.',
                                       dest='model_name')

    ######################################
    # Different model implementations
    ######################################

    dimred = subparsers.add_parser("dimred")
    dimred = parser_system.getParserSystem(dimred)
    dimred = parser_dimred.getParserDimRed(dimred)

    dimred_rc = subparsers.add_parser("dimred_rc")
    dimred_rc = parser_system.getParserSystem(dimred_rc)
    dimred_rc = parser_dimred.getParserDimRed(dimred_rc)
    dimred_rc = parser_testing.getParserTesting(dimred_rc)
    dimred_rc = parser_ae.getParserAE(dimred_rc)
    dimred_rc = parser_rc.getParserRC(dimred_rc)

    dimred_sindy = subparsers.add_parser("dimred_sindy")
    dimred_sindy = parser_system.getParserSystem(dimred_sindy)
    dimred_sindy = parser_dimred.getParserDimRed(dimred_sindy)
    dimred_sindy = parser_testing.getParserTesting(dimred_sindy)
    dimred_sindy = parser_ae.getParserAE(dimred_sindy)
    dimred_sindy = parser_sindy.getParserSINDy(dimred_sindy)

    dimred_rnn = subparsers.add_parser("dimred_rnn")
    dimred_rnn = parser_system.getParserSystem(dimred_rnn)
    dimred_rnn = parser_dimred.getParserDimRed(dimred_rnn)
    dimred_rnn = parser_testing.getParserTesting(dimred_rnn)
    dimred_rnn = parser_rnn.getParserRNN(dimred_rnn)

    crnn = subparsers.add_parser("crnn")
    crnn = parser_system.getParserSystem(crnn)
    crnn = parser_testing.getParserTesting(crnn)
    crnn = parser_crnn.getCRNNParser(crnn)
    """ Multiscale parameters """
    parser.add_argument(
        "--multiscale_testing",
        help="Whether to perform the multiscale testing.",
        type=int,
        required=False,
        default=0,
    )

    parser.add_argument(
        "--multiscale_macro_steps_list",
        action='append',
        help="multiscale testing, list of macro steps to perform",
        type=int,
        default=[],
        required=False,
    )
    parser.add_argument(
        "--multiscale_micro_steps_list",
        action='append',
        help="multiscale testing, list of micro steps to perform",
        type=int,
        default=[],
        required=False,
    )
    parser.add_argument(
        "--plot_multiscale_results_comparison",
        help="plot_multiscale_results_comparison.",
        type=int,
        default=0,
        required=False,
    )

    return parser
