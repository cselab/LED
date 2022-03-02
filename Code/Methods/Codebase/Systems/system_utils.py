#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python


def checkSystemName(model):
    isGood = False
    if model.system_name in [
            "FHN",
            "FHNStructured",
            "Dummy",
            "DummyStructured",
            "KSGP64L22",
            "KSGP64L22Large",
            "cylRe100",
            "cylRe1000",
            "cylRe100HR",
            "cylRe1000HR",
            "cylRe100HRDt005",
            "cylRe1000HRDt005",
            "cylRe100HR_demo",
            "cylRe1000HR_demo",
            "cylRe100_vortPres_veryLowRes",
            "cylRe1000HRLarge",
    ]:
        isGood = True
    return isGood


def addResultsSystem(model, results, testing_mode):
    print("[system_utils] # addResultsSystem() #")
    if "FHN" in model.system_name:
        from .FHN import utils_processing_fhn as utils_processing_fhn
        results = utils_processing_fhn.addResultsSystemFHN(
            model, results, testing_mode)

    if model.system_name in ["KSGP64L22", "KSGP64L22Large"]:
        from .KS import utils_processing_ks as utils_processing_ks
        results = utils_processing_ks.addResultsSystemKS(
            model, results, testing_mode)

    if model.system_name in [
            "cylRe100", "cylRe1000", "cylRe100HR", "cylRe1000HR",
            "cylRe100HR_demo", "cylRe1000HR_demo", "cylRe100HRDt005",
            "cylRe1000HRDt005", "cylRe1000HRLarge",
    ]:
        from .cylRe import utils_cylRe as utils_cylRe
        # results = utils_cylRe.addResultsSystemCylReStatistics(model, results, testing_mode)
        results = utils_cylRe.addResultsSystemCylRe(model, results,
                                                    testing_mode)

    return results


def computeStateDistributionStatisticsSystem(model, state_dist_statistics,
                                             targets_all, predictions_all):
    print("[system_utils] # computeStateDistributionStatisticsSystem() #")

    # Adding system specific state distributions (e.g. Ux-Uxx plot in Kuramoto-Sivashisnky)
    if model.system_name in ["KSGP64L22", "KSGP64L22Large"]:
        from .KS import utils_processing_ks as utils_processing_ks
        state_dist_statistics = utils_processing_ks.computeStateDistributionStatisticsSystemKS(
            state_dist_statistics, targets_all, predictions_all)
        state_dist_statistics = utils_processing_ks.computeStateDistributionStatisticsSystemKSUxUxx(
            state_dist_statistics, targets_all, predictions_all)

    return state_dist_statistics
