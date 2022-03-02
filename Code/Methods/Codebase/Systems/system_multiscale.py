#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

######################################################
## PROCESSING UTILITIES FOR EACH SYSTEM
######################################################

from .FHN import microdynamics_fhn as microdynamics_fhn
from .KS import microdynamics_ks as microdynamics_ks
from .cylRe import utils_cylRe as utils_cylRe


def getMicrodynamicsInfo(model):
    from .. import Utils as utils
    import os

    microdynamics_info_dict = {}
    """ Adding the necessary data/info for running the micro dynamics """
    if "cylRe" in model.system_name:
        data_path = os.path.join(os.getcwd(), model.data_path_gen,
                                 'sim_micro_params.pickle')
        assert os.path.exists(
            data_path
        ), "[getMicrodynamicsInfo()] Data file {:} not found.".format(
            data_path)
        sim_micro_params = utils.loadData(data_path,
                                          "pickle",
                                          add_file_format=False)

        data_file = os.path.join(os.getcwd(), model.data_path_gen,
                                 'sim_micro_data.h5')
        assert os.path.exists(
            data_file
        ), "[getMicrodynamicsInfo()] Data file {:} not found.".format(
            data_file)
        sim_micro_data = utils.getDataHDF5(data_file)

        home_path = model.params["home_path"]
        scratch_path = model.params["scratch_path"]
        cluster = model.params["cluster"]

        if cluster == 'barry':
            cubism_path_launch = scratch_path + "/CubismUP_2D"
            cubism_path_save = scratch_path + "/CubismUP_2D" + "/runs"

        elif cluster == 'local':
            cubism_path_launch = home_path + "/CubismUP_2D"
            cubism_path_save = home_path + "/CubismUP_2D" + "/runs"

        elif cluster == 'daint':
            cubism_path_launch = home_path + "/CubismUP_2D"
            cubism_path_save = scratch_path + "/CUP2D"

        else:
            raise ValueError(
                "Cubism path not set. Unsupported cluster {:}.".format(
                    cluster))

        assert os.path.isdir(
            cubism_path_launch
        ), "[getMicrodynamicsInfo()] Cubism directory {:} not found (to create the launch script).".format(
            cubism_path_launch)
        assert os.path.isdir(
            cubism_path_save
        ), "[getMicrodynamicsInfo()] Cubism directory {:} not found (to save the runs).".format(
            cubism_path_save)

        print(
            "[getMicrodynamicsInfo] Cubism path {:} found (to create the launch script)."
            .format(cubism_path_launch))
        print(
            "[getMicrodynamicsInfo] Cubism path {:} found (to save the runs).".
            format(cubism_path_save))

        microdynamics_info_dict.update({
            'sim_micro_params': sim_micro_params,
            'sim_micro_data': sim_micro_data,
            'cubism_path_launch': cubism_path_launch,
            'cubism_path_save': cubism_path_save,
        })

    return microdynamics_info_dict


def evolveSystem(
    mclass,
    initial_state,
    tend,
    dt_coarse,
    t0=0,
    round_=None,
    micro_steps=None,
    macro_steps=None,
):

    if mclass.model.system_name in ["KSGP64L22", "KSGP64L22Large"]:
        u = microdynamics_ks.evolveKSGP64L22(initial_state, tend, dt_coarse)

    elif "FHN" in mclass.model.system_name:
        u = microdynamics_fhn.evolveFitzHughNagumo(initial_state, tend,
                                                   dt_coarse)

    elif "cylRe" in mclass.model.system_name:
        u = utils_cylRe.evolveCUP2D(mclass, initial_state, tend, dt_coarse, prediction_step=t0,
                                    round_=round_, micro_steps=micro_steps, macro_steps=macro_steps)

    else:
        raise ValueError(
            "Do not know how to evolve the micro dynamics of system {:}.".
            format(mclass.model.system_name))

    return u


def addFieldsToCompare(model, fields_to_compare):

    if model.system_name == "FHN":
        pass
        """
        For FHN system comparing additionally the activator and inhibitor MNAD errors (Mean normalized absolute difference)
        """
        fields_to_compare.append("mnad_act")
        fields_to_compare.append("mnad_in")

    elif model.system_name in ["KSGP64L22", "KSGP64L22Large"]:
        """
        For the Kuramoto-Sivashinsky comparing the errors on the distribution and the errors over initial conditions
        """
        fields_to_compare.append("state_dist_L1_hist_error")
        fields_to_compare.append("state_dist_wasserstein_distance")
        # fields_to_compare.append("rmnse_avg_over_ics")

    return fields_to_compare
