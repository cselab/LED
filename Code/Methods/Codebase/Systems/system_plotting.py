#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

from .. import Utils as utils
import warnings

######################################################
## PLOTTING UTILITIES FOR EACH SYSTEM
######################################################


def plotSystem(model, results, set_name, testing_mode):
    # if model.data_info_dict["structured"]:
    #     warnings.warn("[plotSystem()] Warning: structured data (memory intensive). No plotting of state distribution.")
    #     return 0

    if model.system_name in ["KSGP64L22", "KSGP64L22Large"]:
        from .KS import utils_plotting_ks as utils_plotting_ks
        utils_plotting_ks.plotStateDistributionsSystemKS(
            model, results, set_name, testing_mode)

    if model.system_name in [
            "cylRe100", "cylRe1000", "cylRe100HR", "cylRe1000HR",
            "cylRe100HR_demo", "cylRe1000HR_demo", "cylRe100HRDt005",
            "cylRe1000HRDt005", "cylRe1000HRLarge",
    ]:
        from .cylRe import utils_cylRe_plotting as utils_cylRe_plotting
        # utils_cylRe_plotting.plotDrugCoefficient(model, results, set_name, testing_mode)
        utils_cylRe_plotting.plotSystemCUP2D(model, results, set_name, testing_mode)

    return 0
