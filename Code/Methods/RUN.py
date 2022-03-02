#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import Codebase
import Parser.argparser as argparser
from Config.global_conf import global_params

import sys
import torch

print("[RUN] Python : {:}".format(sys.version))
print("[RUN] Torch  : {:}".format(torch.__version__))


def getModel(params):

    str_ = "[RUN] " + "#" * 10 + "    {:}    ".format(
        params["model_name"]) + "#" * 10
    print("[RUN] " + "#" * len(str_))
    print(str_)
    print("[RUN] " + "#" * len(str_))

    if params["model_name"] == "dimred":
        from Codebase.Networks import dimred
        return Codebase.Networks.dimred.dimred(params)

    if params["model_name"] == "crnn":
        from Codebase.Networks import crnn
        return Codebase.Networks.crnn.crnn(params)

    elif params["model_name"] == "dimred_rc":
        from Codebase.Networks import dimred_rc
        return Codebase.Networks.dimred_rc.dimred_rc(params)

    elif params["model_name"] == "dimred_sindy":
        from Codebase.Networks import dimred_sindy
        return Codebase.Networks.dimred_sindy.dimred_sindy(params)

    elif params["model_name"] == "dimred_rnn":
        from Codebase.Networks import dimred_rnn
        return Codebase.Networks.dimred_rnn.dimred_rnn(params)

    else:
        raise ValueError(
            "     [RUN] Model {:} not found.\n     [RUN] Implemented models are:\n| dimred | crnn | dimred_sindy | dimred_rc | dimred_rnn |"
            .format(params["model_name"]))


def runModel(params_dict):

    if params_dict["mode"] in ["debug"]:
        debugModel(params_dict)

    if params_dict["mode"] in ["plotTraining"]:
        plotTraining(params_dict)

    if params_dict["mode"] in ["train", "all"]:
        trainModel(params_dict)

    if params_dict["mode"] in ["test", "all", "test_only"]:
        testModel(params_dict)

    if params_dict["mode"] in ["test", "plot", "all"]:
        plotModel(params_dict)

    """ Multiscale testing """
    if params_dict["mode"] in ["multiscale"]:
        testModelMultiscale(params_dict)
    """ Multiscale testing """
    if params_dict["mode"] in ["multiscale", "plotMultiscale"]:
        plotModelMultiscale(params_dict)

    """ Multiscale testing """
    if params_dict["mode"] in ["debugMultiscale"]:
        debugModelMultiscale(params_dict)

    return 0


def testModelMultiscale(params_dict):
    model = getModel(params_dict)
    from Codebase.Multiscale import utils_multiscale
    multiscale_testing = utils_multiscale.multiscaleTestingClass(
        model, params_dict)
    multiscale_testing.test()
    del model
    return 0


def plotModelMultiscale(params_dict):
    model = getModel(params_dict)
    from Codebase.Multiscale import utils_multiscale
    multiscale_testing = utils_multiscale.multiscaleTestingClass(
        model, params_dict)
    multiscale_testing.plot()
    del model
    return 0

def debugModelMultiscale(params_dict):
    model = getModel(params_dict)
    from Codebase.Multiscale import utils_multiscale
    multiscale_testing = utils_multiscale.multiscaleTestingClass(
        model, params_dict)
    multiscale_testing.debug()
    del model
    return 0

def debugModel(params_dict):
    model = getModel(params_dict)
    model.debug()
    del model
    return 0


def plotTraining(params_dict):
    model = getModel(params_dict)
    model.plotTraining()
    del model
    return 0


def trainModel(params_dict):
    model = getModel(params_dict)
    model.train()
    del model
    return 0


def testModel(params_dict):
    model = getModel(params_dict)
    model.test()
    del model
    return 0


def plotModel(params_dict):
    model = getModel(params_dict)
    model.plot()
    del model
    return 0


def main():
    parser = argparser.defineParser()
    args = parser.parse_args()
    args_dict = args.__dict__

    # for key in args_dict:
    # print(key)

    # DEFINE PATHS AND DIRECTORIES
    args_dict["saving_path"] = global_params.saving_path.format(
        args_dict["system_name"])
    args_dict["model_dir"] = global_params.model_dir
    args_dict["fig_dir"] = global_params.fig_dir
    args_dict["results_dir"] = global_params.results_dir
    args_dict["logfile_dir"] = global_params.logfile_dir
    args_dict["data_path_train"] = global_params.data_path_train.format(
        args.system_name)
    args_dict["data_path_val"] = global_params.data_path_val.format(
        args.system_name)
    args_dict["data_path_test"] = global_params.data_path_test.format(
        args.system_name)
    args_dict["data_path_gen"] = global_params.data_path_gen.format(
        args.system_name)
    args_dict["worker_id"] = 0
    args_dict["project_path"] = global_params.project_path

    args_dict["home_path"] = global_params.home_path
    args_dict["scratch_path"] = global_params.scratch_path
    args_dict["cluster"] = global_params.cluster

    runModel(args_dict)


if __name__ == '__main__':
    main()
