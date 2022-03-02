#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import numpy as np
import os
import h5py

import torch
import torch.nn as nn

import shutil

from ... import Utils as utils

import subprocess
import shlex

from . import utils_cylRe_drug as utils_cylRe_drug


def addResultsSystemCylReStatistics(model, results, testing_mode):
    print("[utils_cylRe] addResultsSystemCylReStatistics()")
    print(
        "[utils_cylRe] Computing error on state (pressure/drag/velocity) distribution"
    )

    if ("autoencoder" in testing_mode) or ("dimred" in testing_mode):
        targets_all = results["inputs_all"]
        predictions_all = results["outputs_all"]
    else:
        targets_all = results["targets_all"]
        predictions_all = results["predictions_all"]
    shape_ = np.shape(targets_all)
    n_ics = shape_[0]
    T = shape_[1]

    prediction_values = []
    target_values = []

    for n in range(n_ics):
        for t in range(T):
            prediction = predictions_all[n][t]
            target = targets_all[n][t]

            if model.data_info_dict["structured"]:
                target = utils.getDataHDF5Field(target[0], target[1])
                prediction = utils.getDataHDF5Field(prediction[0],
                                                    prediction[1])

            prediction = np.reshape(prediction, (np.shape(prediction)[0], -1))
            target = np.reshape(target, (np.shape(target)[0], -1))

            if n == 0 and t == 0:
                prediction_values = prediction
                target_values = target
            else:
                prediction_values = np.concatenate(
                    (prediction_values, prediction), axis=1)
                target_values = np.concatenate((target_values, target), axis=1)

    prediction_values = np.array(prediction_values)
    target_values = np.array(target_values)

    L1_hist_error = []
    wasserstein_distance = []
    KS_error = []

    D = np.shape(prediction_values)[0]
    assert D == 4

    for d in range(D):
        predictions_ = prediction_values[d]
        targets_ = target_values[d]
        N_samples = np.shape(predictions_)[0]
        min_ = np.min([np.min(targets_), np.min(predictions_)])
        max_ = np.max([np.max(targets_), np.max(predictions_)])
        bounds = [min_, max_]
        LL = max_ - min_
        nbins = utils.getNumberOfBins(N_samples, LL)
        # L1_hist_error, error_vec, density_target, density_pred, bin_centers = evaluateL1HistErrorVector(targets_, predictions_, nbins, bounds)
        hist_data = utils.evaluateL1HistErrorVector(targets_, predictions_,
                                                    nbins, bounds)
        L1_hist_error_ic = hist_data[0]
        wasserstein_distance_ic = utils.evaluateWassersteinDistance(
            targets_, predictions_)
        KS_error_ic = utils.evaluateKSError(targets_, predictions_)

        L1_hist_error.append(L1_hist_error_ic)
        wasserstein_distance.append(wasserstein_distance_ic)
        KS_error.append(KS_error_ic)

    L1_hist_error = np.array(L1_hist_error)
    wasserstein_distance = np.array(wasserstein_distance)
    KS_error = np.array(KS_error)

    L1_hist_error_mean = np.mean(L1_hist_error)
    wasserstein_distance_mean = np.mean(wasserstein_distance)
    KS_error_mean = np.mean(KS_error)

    print(
        "[utils_cylRe] Wasserstein distance (mean over channels) = {:}".format(
            wasserstein_distance_mean))
    print("[utils_cylRe] KS_error (mean over channels) = {:}".format(
        KS_error_mean))
    print("[utils_cylRe] L1_hist_error (mean over channels) = {:}".format(
        L1_hist_error_mean))
    results.update({
        "L1_hist_error_mean": L1_hist_error_mean,
        "wasserstein_distance_mean": wasserstein_distance_mean,
    })

    results["fields_2_save_2_logfile"].append("L1_hist_error_mean")
    results["fields_2_save_2_logfile"].append("wasserstein_distance_mean")

    return results


def addResultsSystemCylRe(model, results, testing_mode):
    print("[utils_cylRe] addResultsSystemCylRe()")
    print("[utils_cylRe] Computing drag coefficient")

    if ("autoencoder" in testing_mode) or ("dimred" in testing_mode):
        targets_all = results["inputs_all"]
        predictions_all = results["outputs_all"]
    else:
        targets_all = results["targets_all"]
        predictions_all = results["predictions_all"]

    shape_ = np.shape(targets_all)
    num_ics = shape_[0]
    T = shape_[1]

    drag_coef_pred = []
    drag_coef_targ = []

    sim_micro_params = model.data_info_dict["sim_micro_params"]
    sim_micro_data = model.data_info_dict["sim_micro_data"]

    # print("sim_micro_params:")
    # for key in sim_micro_params:
    #     print(key)
    # print("sim_micro_data:")
    # for key in sim_micro_data:
    #     print(key)

    # sim_micro_params:
    # spatial_subsampling
    # uinfx
    # uinfy
    # x
    # y
    # xlab
    # ylab
    # u
    # v
    # omega
    # Nx
    # Ny
    # Nz
    # BPDX
    # BPDY
    # CFL
    # XPOS
    # ANGLE
    # XVEL
    # RADIUS
    # NU
    # rho

    # sim_micro_data:
    # chi
    # chi_sub
    # dx
    # dy
    # vx
    # vx_sub
    # vy
    # vy_sub
    # vz

    NU = sim_micro_params["NU"]
    mu = NU

    rho = sim_micro_params["rho"]

    RADIUS = sim_micro_params["RADIUS"]
    x = sim_micro_params["x"]
    y = sim_micro_params["y"]
    uinfx = sim_micro_params["uinfx"]
    uinfy = sim_micro_params["uinfy"]

    vx_sub = sim_micro_data["vx_sub"]
    vy_sub = sim_micro_data["vy_sub"]

    gridx = vx_sub
    gridy = vy_sub
    Dx = gridx[-1] - gridx[0]
    Dy = gridy[-1] - gridy[0]
    gridx = (gridx[1:] + gridx[:-1]) / 2.0
    gridy = (gridy[1:] + gridy[:-1]) / 2.0
    dx = gridx[1] - gridx[0]
    dy = gridy[1] - gridy[0]

    R = RADIUS
    cyl_x = x
    cyl_y = y

    vel_x_inf = uinfx
    vel_y_inf = uinfy

    center = [cyl_x, cyl_y]

    for ic in range(num_ics):
        drag_ic_pred = []
        drag_ic_targ = []
        for t in range(T):

            prediction = predictions_all[ic][t]
            target = targets_all[ic][t]

            if model.data_info_dict["structured"]:
                target = utils.getDataHDF5Field(target[0], target[1])
                prediction = utils.getDataHDF5Field(prediction[0],
                                                    prediction[1])

            pressure_field_pred = prediction[0]
            velocity_field_pred = prediction[1:3]

            drag_coef_pred_ = computeDragCoefficient(pressure_field_pred,
                                                     velocity_field_pred,
                                                     gridx, gridy, center, R,
                                                     mu, vel_x_inf, rho)

            pressure_field_targ = target[0]
            velocity_field_targ = target[1:3]

            drag_coef_targ_ = computeDragCoefficient(pressure_field_targ,
                                                     velocity_field_targ,
                                                     gridx, gridy, center, R,
                                                     mu, vel_x_inf, rho)

            drag_ic_pred.append(drag_coef_pred_)
            drag_ic_targ.append(drag_coef_targ_)

        drag_coef_targ.append(drag_ic_targ)
        drag_coef_pred.append(drag_ic_pred)

    drag_coef_targ = np.array(drag_coef_targ)
    drag_coef_pred = np.array(drag_coef_pred)

    drag_coef_error_abs_all = np.abs(drag_coef_targ - drag_coef_pred)
    drag_coef_error_rel_all = drag_coef_error_abs_all / drag_coef_targ

    results["drag_coef_error_abs_all"] = drag_coef_error_abs_all
    results["drag_coef_error_rel_all"] = drag_coef_error_rel_all

    drag_coef_error_abs = np.mean(drag_coef_error_abs_all)
    drag_coef_error_rel = np.mean(drag_coef_error_rel_all)

    results["drag_coef_targ"] = drag_coef_targ
    results["drag_coef_pred"] = drag_coef_pred
    results["drag_coef_error_abs"] = drag_coef_error_abs
    results["drag_coef_error_rel"] = drag_coef_error_rel

    results["fields_2_save_2_logfile"].append("drag_coef_error_rel")
    results["fields_2_save_2_logfile"].append("drag_coef_error_abs")

    return results


def computeDragCoefficient(pressure_field, velocity_field, gridx, gridy,
                           center, R, mu, vel_x_inf, rho):

    assert len(np.shape(pressure_field)) == 2
    assert len(np.shape(velocity_field)) == 3
    assert np.shape(velocity_field)[0] == 2

    F_p = utils_cylRe_drug.integrateFp(pressure_field, gridx, gridy, center, R)
    # print("Pressure forces:")
    # print(F_p)

    F_m = utils_cylRe_drug.integrateFm(velocity_field, gridx, gridy, center, R,
                                       mu)
    # print("Viscous forces:")
    # print(F_m)

    # print("Total forces:")
    F = F_p + F_m
    # print(F)

    # print("Drag:")
    D = F[0]
    # print(D)

    # print("Drag coefficient:")
    u_inf_norm_2 = np.power(vel_x_inf, 2)
    A = 2 * R
    Cd = 2 * D / (rho * u_inf_norm_2 * A)
    # print(Cd)
    return Cd


def makeDir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        print("[evolveCUP2D] Creation of the directory %s failed" % path)
    else:
        print("[evolveCUP2D] Successfully created the directory %s " % path)


def getHDF5Files(directory):
    list_of_files = []
    for file in os.listdir(directory):
        if file.endswith(".h5"):
            file_path = os.path.join(directory, file)
            list_of_files.append(file_path)
    return list_of_files


def getFieldData(filename, keep_indexes=None, spatial_subsampling=[2, 2]):
    with h5py.File(filename, "r") as f:
        dict_of_values = {}
        for key in f.keys():
            dict_of_values[key] = np.array(f[key])

        temp = dict_of_values["data"]
        _, Dx, Dy, C = np.shape(temp)

        subsamplingx, subsamplingy = spatial_subsampling

        Dx_trim = 0
        idx_start = Dx_trim
        idx_end = Dx - Dx_trim
        #
        Dy_trim = 0
        idy_start = Dy_trim
        idy_end = Dy - Dy_trim

        # Get relevant data
        frame = temp[0, idx_start:idx_end, idy_start:idy_end]

        frame = frame[::subsamplingx, ::subsamplingy]
        # print(np.shape(frame))
        # Reordering
        # print(np.shape(frame))
        frame_reord = frame
        frame_reord = np.swapaxes(frame_reord, 0, 2)
        frame_reord = np.swapaxes(frame_reord, 1, 2)
        if keep_indexes is not None:
            frame_reord = frame_reord[keep_indexes]
    return frame_reord


def parseCubismOutput(data_dir, spatial_subsampling):
    # print("# parseCubismOutput() #")
    # print("Reading from directory: {:}".format(data_dir))
    list_of_files = getHDF5Files(data_dir)

    # Get only the velocity files
    list_of_files_velocity = [
        temp for temp in list_of_files if "velChi_avemaria_" in temp
    ]
    list_of_files_pressure = [
        temp for temp in list_of_files if "pres_avemaria_" in temp
    ]
    list_of_files_vorticity = [
        temp for temp in list_of_files if "tmp_avemaria_" in temp
    ]

    list_of_files_velocity.sort()
    list_of_files_pressure.sort()
    list_of_files_vorticity.sort()

    # print("Number of velocity timesteps found: {:}".format(len(list_of_files_velocity)))
    # print("Number of pressure timesteps found: {:}".format(len(list_of_files_pressure)))
    # print("Number of vorticity timesteps found: {:}".format(len(list_of_files_vorticity)))

    # Removing the first file (initial state)
    list_of_files_velocity = list_of_files_velocity[1:]

    assert len(list_of_files_velocity) == len(list_of_files_pressure)
    assert len(list_of_files_velocity) == len(list_of_files_vorticity)

    trajectory = []
    for fl in range(len(list_of_files_velocity)):
        # print("Adding file {:}/{:} - {:.2f}%.".format(fl, len(list_of_files_velocity), 100.*fl/len(list_of_files_velocity)))
        file_velocity = list_of_files_velocity[fl]
        file_pressure = list_of_files_pressure[fl]
        file_vorticity = list_of_files_vorticity[fl]

        timestep_velocity = int(file_velocity.split("_")[-1].split(".h5")[0])
        timestep_pressure = int(file_pressure.split("_")[-1].split(".h5")[0])
        timestep_vorticity = int(file_vorticity.split("_")[-1].split(".h5")[0])

        assert timestep_velocity == timestep_pressure, "Timestep in velocity file {:}, mismatch with timestep in pressure file {:}.".format(
            timestep_velocity, timestep_pressure)
        assert timestep_velocity == timestep_vorticity, "Timestep in velocity file {:}, mismatch with timestep in vorticity file {:}.".format(
            timestep_velocity, timestep_vorticity)

        data_pressure = getFieldData(file_pressure,
                                     spatial_subsampling=spatial_subsampling)
        data_velocity = getFieldData(file_velocity,
                                     keep_indexes=[0, 1],
                                     spatial_subsampling=spatial_subsampling)
        data_vorticity = getFieldData(file_vorticity,
                                      spatial_subsampling=spatial_subsampling)
        data_frame = np.concatenate(
            (data_pressure, data_velocity, data_vorticity), axis=0)
        # print(np.shape(data_frame))
        trajectory.append(data_frame)

    trajectory = np.array(trajectory)
    # print(np.shape(trajectory))
    return trajectory


def upsample(field, scale=1):
    # print("# upsample() #")
    # print(np.shape(field))
    field = torch.tensor(field)
    upsample_operator = nn.Upsample(scale_factor=scale,
                                    mode='bilinear',
                                    align_corners=False)
    # upsample_operator = nn.Upsample(scale_factor=scale, mode='nearest', align_corners=False)
    field_upsampled = upsample_operator(field)
    # print(field_upsampled.size())
    field_upsampled = field_upsampled.cpu().detach().numpy()
    # print(np.shape(field_upsampled))
    return field_upsampled


def evolveCUP2D(
    mclass,
    init_state,
    total_time,
    dt_model,
    prediction_step=0,
    round_=None,
    micro_steps=None,
    macro_steps=None,
):

    model = mclass.model
    microdynamics_info_dict = mclass.microdynamics_info_dict

    assert "sim_micro_params" in microdynamics_info_dict
    assert "cubism_path_launch" in microdynamics_info_dict
    assert "cubism_path_save" in microdynamics_info_dict

    cubism_path_launch = microdynamics_info_dict["cubism_path_launch"]
    cubism_path_save = microdynamics_info_dict["cubism_path_save"]

    # print(np.shape(init_state))
    # print(total_time)
    # print(dt_model)
    # nn = int(total_time//dt_model)
    # trajectory = []
    # for i in range(nn):
    #     trajectory.append(init_state)
    # trajectory = np.array(trajectory)
    # trajectory = trajectory[:,0]
    # return trajectory
    """ Loading micro scale simulation data """
    sim_micro_params = microdynamics_info_dict["sim_micro_params"]

    spatial_subsampling = sim_micro_params["spatial_subsampling"]

    # assert spatial_subsampling[0] == 4
    # assert spatial_subsampling[1] == 4

    uinfx = sim_micro_params["uinfx"]
    uinfy = sim_micro_params["uinfy"]
    x = sim_micro_params["x"]
    y = sim_micro_params["y"]
    xlab = sim_micro_params["xlab"]
    ylab = sim_micro_params["ylab"]
    u = sim_micro_params["u"]
    v = sim_micro_params["v"]
    omega = sim_micro_params["omega"]

    Nx = sim_micro_params["Nx"]
    Ny = sim_micro_params["Ny"]
    Nz = sim_micro_params["Nz"]

    BPDX = sim_micro_params["BPDX"]
    BPDY = sim_micro_params["BPDY"]
    CFL = sim_micro_params["CFL"]
    XPOS = sim_micro_params["XPOS"]
    ANGLE = sim_micro_params["ANGLE"]
    XVEL = sim_micro_params["XVEL"]
    RADIUS = sim_micro_params["RADIUS"]
    NU = sim_micro_params["NU"]

    system_name = model.system_name

    steps = int(total_time / dt_model)

    cubism_path_runs = cubism_path_save
    cubism_path_launch = cubism_path_launch + "/launch"

    # Create directory for restart
    run_name = "LED_{:}_round{:}_micro{:}_macro{:}_restart_step{:}".format(
        system_name, round_, micro_steps, macro_steps, prediction_step)
    micro_sim_path = cubism_path_runs + "/" + run_name

    if os.path.exists(micro_sim_path):
        print("[evolveCUP2D] Removing micro scale simulation path {:}.".format(
            micro_sim_path))
        # os.remove(micro_sim_path)
        shutil.rmtree(micro_sim_path)

    makeDir(micro_sim_path)

    time_start = prediction_step * dt_model

    # stepid = prediction_step
    stepid = 200 + prediction_step
    ##############################
    # Create field.restart file
    ##############################
    fname = micro_sim_path + "/field.restart"
    try:
        file = open(fname, "w")
        file.write("time: {:.20e}\n".format(time_start))
        file.write("stepid: {:d}\n".format(stepid))
        file.write("uinfx: {:.20e}\n".format(uinfx))
        file.write("uinfy: {:.20e}\n".format(uinfy))
        file.close()
    except OSError:
        print("[evolveCUP2D] Creation of %s file failed" % fname)
    else:
        print("[evolveCUP2D] Successfully created the file %s" % fname)

    ##############################
    # Create shape_0.restart file
    ##############################

    fname = micro_sim_path + "/shape_0.restart"
    try:
        file = open(fname, "w")
        file.write("{:7s}{:.20e}\n".format("x:", x))
        file.write("{:7s}{:.20e}\n".format("y:", y))
        file.write("{:7s}{:.20e}\n".format("xlab:", xlab))
        file.write("{:7s}{:.20e}\n".format("ylab:", ylab))
        file.write("{:7s}{:.20e}\n".format("u:", u))
        file.write("{:7s}{:.20e}\n".format("v:", v))
        file.write("{:7s}{:.20e}\n".format("omega:", omega))
        file.close()
    except OSError:
        print("[evolveCUP2D] Creation of %s file failed" % fname)
    else:
        print("[evolveCUP2D] Successfully created the file %s" % fname)

    ##############################
    # Create .xmf file
    ##############################
    fname = micro_sim_path + "/velChi_avemaria_{:07d}.xmf".format(stepid)
    file = open(fname, "w")
    file.write("<?xml version=\"1.0\" ?>\n")
    file.write("<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n")
    file.write("<Xdmf Version=\"2.0\">\n")
    file.write(" <Domain>\n")
    file.write("   <Grid GridType=\"Uniform\">\n")
    file.write("     <Time Value=\"{:e}\"/>\n".format(time_start))
    file.write("\n")

    file.write(
        "     <Topology TopologyType=\"3DRectMesh\" Dimensions=\"{:} {:} {:}\"/>\n"
        .format(Nz, Ny, Nx))
    file.write("\n")

    file.write("     <Geometry GeometryType=\"VxVyVz\">\n")
    file.write(
        "       <DataItem Name=\"mesh_vx\" Dimensions=\"{:}\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n"
        .format(Nx))
    file.write("        velChi_avemaria_{:07d}.h5:/vx\n".format(stepid))
    file.write("       </DataItem>\n")
    file.write(
        "       <DataItem Name=\"mesh_vy\" Dimensions=\"{:}\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n"
        .format(Ny))
    file.write("        velChi_avemaria_{:07d}.h5:/vy\n".format(stepid))
    file.write("       </DataItem>\n")
    file.write(
        "       <DataItem Name=\"mesh_vz\" Dimensions=\"{:}\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n"
        .format(Nz))
    file.write("        velChi_avemaria_{:07d}.h5:/vz\n".format(stepid))
    file.write("       </DataItem>\n")
    file.write("     </Geometry>\n")
    file.write("\n")

    file.write(
        "     <Attribute Name=\"data\" AttributeType=\"Vector\" Center=\"Cell\">\n"
    )
    file.write(
        "       <DataItem Dimensions=\"{:} {:} {:} {:}\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n"
        .format(Nz - 1, Ny - 1, Nx - 1, 3))
    file.write("        velChi_avemaria_{:07d}.h5:/data\n".format(stepid))
    file.write("       </DataItem>\n")
    file.write("     </Attribute>\n")
    file.write("   </Grid>\n")
    file.write(" </Domain>\n")
    file.write("</Xdmf>\n")

    file.close()

    ############################################################
    # Create the initial velocity file (with the mesh/chi/etc.)
    ############################################################
    # Get the mesh data
    sim_micro_data = microdynamics_info_dict["sim_micro_data"]
    # filename = multiscale_model.data_path_gen + "/velChi_mesh.h5"
    # dict_of_values = utils.getDataHDF5(filename)
    # print(sim_micro_data.keys())

    # Create the velocity file
    fname_initial_velocity = micro_sim_path + "/velChi_avemaria_{:07d}.h5".format(
        stepid)
    with h5py.File(fname_initial_velocity, "w") as file:
        vx = sim_micro_data["vx"]
        vy = sim_micro_data["vy"]
        vz = sim_micro_data["vz"]
        chi = sim_micro_data["chi"]

        # data_frame = np.concatenate((data_pressure (1), data_velocity (2), data_vorticity (1)), axis=0)
        idx2keep = [1, 2]

        # Removing pressure and vorticity
        init_state_vel = init_state[:, idx2keep]
        # upsumpling
        if spatial_subsampling[0] > 1:
            init_state_vel = upsample(init_state_vel,
                                      scale=tuple(spatial_subsampling))
        # print("np.shape(init_state_vel)")
        # print(np.shape(init_state_vel))
        init_state_vel = np.swapaxes(init_state_vel, 1, 2)
        init_state_vel = np.swapaxes(init_state_vel, 2, 3)
        # print(np.shape(init_state_vel))
        # print(np.shape(chi))

        # Adding the chi field
        data = np.concatenate((init_state_vel, chi), axis=3)
        # print(np.shape(data))

        dset = file.create_dataset("vx",
                                   sim_micro_data["vx"].shape,
                                   dtype='f',
                                   data=vx)
        dset = file.create_dataset("vy",
                                   sim_micro_data["vy"].shape,
                                   dtype='f',
                                   data=vy)
        dset = file.create_dataset("vz",
                                   sim_micro_data["vz"].shape,
                                   dtype='f',
                                   data=vz)
        dset = file.create_dataset("data", data.shape, dtype='f', data=data)

    tend = time_start + total_time

    ##############################
    # Create launch script
    # launchDiskRe1000_restart_script
    ##############################
    launch_script = "launch_{:}_round{:}_micro{:}_macro{:}_restart_step{:}.sh".format(
        system_name, round_, micro_steps, macro_steps, prediction_step)
    fname = cubism_path_launch + "/" + launch_script

    if os.path.isfile(fname):
        print("[evolveCUP2D] Removing launchscript {:}.".format(launch_script))
        os.remove(fname)

    file = open(fname, "w")
    file.write('#!/usr/bin/env bash\n')
    file.write("# Defaults for Options\n")
    file.write("BPDX=${BPDX:-" + "{:}".format(BPDX) + "}\n")
    file.write("BPDY=${BPDY:-" + "{:}".format(BPDY) + "}\n")
    file.write("EXTENT=${EXTENT:-1}\n")
    file.write("CFL=${CFL:-" + "{:}".format(CFL) + "}\n")
    file.write("# Defaults for Objects\n")
    file.write("XPOS=${XPOS:-" + "{:}".format(XPOS) + "}\n")
    file.write("ANGLE=${ANGLE:-" + "{:}".format(ANGLE) + "}\n")
    file.write("XVEL=${XVEL:-" + "{:}".format(XVEL) + "}\n")
    file.write("RADIUS=${RADIUS:-" + "{:}".format(RADIUS) + "}\n")
    file.write("NU=${NU:-" + "{:.10f}".format(NU) + "}\n")
    # file.write("OPTIONS=\"-bpdx $BPDX -bpdy $BPDY -extent $EXTENT -CFL $CFL -tdump {:} -nu $NU -tend {:} -poissonType dirichlet -iterativePensalization 0 -muteAll 0 -verbose 0 -restart 1\"\n".format(dt_model, tend))
    file.write(
        "OPTIONS=\"-bpdx $BPDX -bpdy $BPDY -extent $EXTENT -CFL $CFL -tdump {:} -nu $NU -tend {:} -poissonType dirichlet -iterativePensalization 0 -muteAll 0 -verbose 1 -restart 1\"\n"
        .format(dt_model, tend))
    file.write("# bForced, tAccel is needed here!\n")
    file.write(
        "OBJECTS=\"disk radius=$RADIUS angle=$ANGLE xpos=$XPOS bForced=1 bFixed=1 xvel=$XVEL tAccel=0\"\n"
    )
    file.write("\n")
    # file.write("source launchCommon.sh\n")
    file.write("source launchCommonDaintLED.sh\n")
    os.chmod(fname, 0o777)
    file.close()

    # print(os.environ['HOST'])
    """ Change to Cubism up directory, and launch the restart script """
    command_to_run = "cd {:}; pwd; ./{:} {:}".format(cubism_path_launch,
                                                     launch_script, run_name)
    print("[evolveCUP2D] Running OS command: {:}".format(command_to_run))

    try:
        result = subprocess.check_output(command_to_run,
                                         encoding="utf-8",
                                         shell=True)
        print("[evolveCUP2D] Command output:\n")
        print(result)
    except subprocess.CalledProcessError as error_:
        print("[evolveCUP2D] Command error:\n")
        print(error_.output)
        raise ValueError("[evolveCUP2D] Cubism command failed.")

    trajectory = parseCubismOutput(micro_sim_path, spatial_subsampling)

    print("[evolveCUP2D] np.shape(init_state)={:}\n".format(
        np.shape(init_state)))
    print("[evolveCUP2D] np.shape(trajectory)={:}\n".format(
        np.shape(trajectory)))

    if os.path.exists(micro_sim_path):
        print("[evolveCUP2D] Removing micro scale simulation path {:}.".format(micro_sim_path))
        # os.remove(micro_sim_path)
        shutil.rmtree(micro_sim_path)

    if os.path.isfile(fname):
        print("[evolveCUP2D] Removing launchscript {:}.".format(launch_script))
        os.remove(fname)

    return trajectory







