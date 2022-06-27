from itertools import product

hyper_params_dict = {
    "mode": ["all"],
    # "system_name": ["cylRe100HR", "cylRe1000HR", "cylRe100HRDt005", "cylRe1000HRDt005"],
    "system_name": ["cylRe1000HR"],
    "cudnn_benchmark": [1],
    "write_to_log": [1],
    "channels": [2],
    "input_dim": [4],
    "Dx": [1024],
    "Dy": [512],
    "optimizer_str": ["adabelief"],
    "output_forecasting_loss": [0],
    "latent_forecasting_loss": [0],
    "reconstruction_loss": [1],
    "activation_str_general": ["celu"],
    # "activation_str_output": ["tanhplus", "identity"],
    "activation_str_output": ["tanhplus"],
    "precision": ["single", "double"],
    "AE_size_factor": [1, 2],
    "AE_convolutional": [1],
    "AE_batch_norm": [1],
    # "AE_batch_norm_affine": [0, 1],
    "AE_batch_norm_affine": [0],
    "AE_conv_transpose": [0, 1],
    "AE_pool_type": ["avg"],
    "AE_interp_subsampling_input": [1, 2, 4],
    "AE_conv_architecture": [
    "conv_latent_1",
    "conv_latent_2",
    "conv_latent_3",
    # "conv_latent_4",
    # "conv_latent_5",
    # "conv_latent_6",
    ],
    # "latent_state_dim": [2, 4, 6, 8, 10, 12, 14, 16],
    "latent_state_dim": [10],
    "noise_level": [0.0],
    "sequence_length": [1],
    "n_warmup_train": [0],
    "n_warmup": [0],
    "scaler": ["MinMaxZeroOne"],
    "learning_rate": [0.001],
    "weight_decay": [0.0, 0.00001],
    "dropout_keep_prob": [1.0],
    # "batch_size": [16],
    "batch_size": [32],
    "overfitting_patience":[40],
    "max_epochs": [20000],
    "max_rounds": [20],
    "display_output": [1],
    "random_seed": [10],
    "random_seed_in_name": [1],
    "make_videos": [0],
    "retrain": [0],
    "compute_spectrum": [0],
    "teacher_forcing_forecasting": [1],
    "iterative_latent_forecasting": [1],
    "iterative_state_forecasting": [0],
    "test_on_test": [1],
    "test_on_val": [1],
    "test_on_train": [0],
    "plot_state_distributions": [0],
    "plot_system": [1],
    "plot_latent_dynamics": [1],
    "plot_testing_ics_examples": [1],
    "reference_train_time": [24.0],
    "buffer_train_time": [6.0], # For testing & saving!
    "c1_latent_smoothness_loss": [0],
    "c1_latent_smoothness_loss_factor": [0.0],
    "iterative_loss_schedule_and_gradient": ["none"],
    "iterative_loss_validation": [0],
    "write_to_log": [1],
    "iterative_propagation_during_training_is_latent": [1],
    "num_test_ICS": [1],
    # "prediction_horizon": [1000],
    # "prediction_horizon": [10],
    "prediction_horizon": [100],
    "plotting":[1],
}

# hyper_params_dict["mode"] = ["all"]
# hyper_params_dict["retrain"] = [0]
# hyper_params_dict["buffer_train_time"] = [4.5] # For testing

# hyper_params_dict["mode"] = ["test"]
# hyper_params_dict["mode"] = ["plot"]
# hyper_params_dict["mode"] = ["plotTraining"]
# hyper_params_dict["plotting"] = [1]
# hyper_params_dict["plot_testing_ics_examples"] = [1]
# hyper_params_dict["plot_system"] = [1]

max_jobs_per_run = 24
# max_jobs_per_run = 40
# max_jobs_per_run = 1



# ##################################################
# ##################################################
# ## FOR PLOTTING PURPOSES
# ##################################################
# ##################################################
# hyper_params_dict["make_videos"]        = [0]
# hyper_params_dict["write_to_log"]       = [0]
# # hyper_params_dict["mode"]               = ["test"]
# hyper_params_dict["mode"]               = ["plot"]
# hyper_params_dict["latent_state_dim"]   = [3]
# hyper_params_dict["system_name"]        = ["cylRe1000HR"]
# hyper_params_dict["plotting"]                   = [1]
# hyper_params_dict["plot_testing_ics_examples"]  = [1]
# hyper_params_dict["plot_latent_dynamics"]       = [1]
# hyper_params_dict["plot_system"]                = [1]
# hyper_params_dict["prediction_horizon"] = [10]
# hyper_params_dict["AE_conv_transpose"]  = [0]
# hyper_params_dict["random_seed"]        = [10]
# max_jobs_per_run = 1




# List of all hyper parameter combinations (list of dictionaries) to run
hyp_dict_list = [
    dict(zip(hyper_params_dict.keys(), v))
    for v in product(*hyper_params_dict.values())
]

# TESTING:
# For prediction_horizon=10: 24 minutes
# For prediction_horizon=100: 240 minutes = 4 hours

# ##################################################
# ##################################################
# ## FOR DEBUGGING PURPOSES
# ##################################################
# ##################################################
# hyp_dict_list = hyp_dict_list[-1:]
# hyp_dict_list[0]["mode"]       = "all"
# # hyp_dict_list[0]["mode"]       = "test"
# hyp_dict_list[0]["batch_size"]     = 32
# hyp_dict_list[0]["truncate_data_batches"]     = 32
# hyp_dict_list[0]["truncate_timesteps"]     = 3
# # hyp_dict_list[0]["mode"]       = "test"
# hyp_dict_list[0]["max_epochs"]       = 2
# hyp_dict_list[0]["num_test_ICS"]     = 1
# hyp_dict_list[0]["write_to_log"]     = 0
# # hyp_dict_list[0]["write_to_log"]     = 1
# hyp_dict_list[0]["prediction_horizon"]     = 2
# hyp_dict_list[0]["test_on_test"]     = 1
# hyp_dict_list[0]["test_on_val"]     = 1
# max_jobs_per_run = 1







print("NUMBER OF HYPER-PARAMETER COMBINATIONS:")
print(len(hyp_dict_list))

# print(ark)



#####################################################################
# RUNNING ALL
#####################################################################
PATH = "/users/pvlachas/LED/Code/Methods/"
model_name = "crnn"



total_jobs = len(hyp_dict_list)
if total_jobs % max_jobs_per_run != 0:
    raise ValueError("total_jobs (={:}) % max_jobs_per_run (={:}) != 0".format(
        total_jobs, max_jobs_per_run))

num_runs = int(total_jobs // max_jobs_per_run)
print("Total number of runs {:}".format(num_runs))

for run_num in range(num_runs):
    with open("./Tasks/WW_B{:}_greasy_AECNN_tasks.txt".format(run_num + 1),
              "w") as file:
        for dict_num in range(run_num * max_jobs_per_run,
                              (run_num + 1) * max_jobs_per_run):
            hyper_param_dict_case = hyp_dict_list[dict_num]
            
            command = "[@ {:} @] python3 RUN.py {:}".format(PATH, model_name)
            for key, value in hyper_param_dict_case.items():
                # print(key)
                command += " --{:} {:}".format(key, value)
            command += ";\n"
            file.write(command)






