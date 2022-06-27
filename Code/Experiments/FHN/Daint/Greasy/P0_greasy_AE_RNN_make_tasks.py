from itertools import product

# |     17 | 9.39735e-07 | GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-CHANNELS_2-8-16-32-4-KERNELS_5-5-5-5-BN_0-TRANS_1-POOL_avg-ACT_celu-DKP_1.0-LD_2  |

hyper_params_dict = {
    "mode": ["all"],
    "system_name": ["FHN"],
    "cudnn_benchmark": [1],
    "write_to_log": [1],
    "channels": [1],
    "input_dim": [2],
    "Dx": [101],
    "optimizer_str": ["adabelief"],
    "output_forecasting_loss": [0],
    "latent_forecasting_loss": [0],
    "reconstruction_loss": [1],
    "activation_str_general": ["celu"],
    "activation_str_output": ["tanhplus"],
    "AE_convolutional": [0],
    "AE_layers_num": [3],
    "AE_layers_size": [100],
    "dropout_keep_prob": [1.0],
    "weight_decay": [0.0],
    "latent_state_dim": [2, 3, 4],
    "noise_level": [0.0],
    "sequence_length": [1],
    "n_warmup_train": [0],
    "n_warmup": [0],
    "scaler": ["MinMaxZeroOne"],
    "learning_rate": [0.001],
    "batch_size": [32],
    "overfitting_patience":[10],
    "max_epochs": [10000],
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
    "test_on_train": [1],
    "plot_state_distributions": [0],
    "plot_system": [0],
    "plot_latent_dynamics": [0],
    "plot_testing_ics_examples": [0],
    "reference_train_time": [20.0],
    "buffer_train_time": [1.0],
    "c1_latent_smoothness_loss": [0],
    "c1_latent_smoothness_loss_factor": [0.0],
    "iterative_loss_schedule_and_gradient": ["none"],
    "iterative_loss_validation": [0],
    "write_to_log": [1],
    "iterative_propagation_during_training_is_latent": [1],
    "num_test_ICS": [32],
    "prediction_horizon": [8000],
    "plotting":[0],
}

##################################################
##################################################
## RNN
##################################################
##################################################
IS_MULTISCALE_TESTING = False
hyper_params_dict["random_seed_in_AE_name"]     = hyper_params_dict["random_seed"]
hyper_params_dict["random_seed"]                = [7]
hyper_params_dict["overfitting_patience"]       = [20]
hyper_params_dict["sequence_length"]            = [20, 40, 60]
hyper_params_dict["n_warmup"]                   = [20]
hyper_params_dict["retrain"]                    = [0]
hyper_params_dict["train_RNN_only"]             = [1]
hyper_params_dict["load_trained_AE"]            = [1]
hyper_params_dict["reconstruction_loss"]        = [0]
hyper_params_dict["output_forecasting_loss"]    = [0, 1]
hyper_params_dict["latent_forecasting_loss"]    = [1]

hyper_params_dict["RNN_cell_type"]              = ["lstm"]
hyper_params_dict["RNN_layers_num"]             = [1]
hyper_params_dict["RNN_layers_size"]            = [16, 32, 64]
hyper_params_dict["RNN_activation_str_output"]  = ["tanhplus"]
hyper_params_dict["write_to_log"]               = [1]
hyper_params_dict["mode"]                       = ["all"]

hyper_params_dict["num_test_ICS"]                       = [32]
hyper_params_dict["prediction_horizon"]                 = [8000]


max_jobs_per_run = 54
# max_jobs_per_run = 1



# ##################################################
# ##################################################
# ## MULTISCALE TESTING
# ##################################################
# ##################################################
# IS_MULTISCALE_TESTING = True
# # |     43 | 0.0278205  | GPU-ARNN-scaler_MinMaxZeroOne-OPT_adabelief-LR_0.001-NL_0.0-L2_0.0-RS_10-AUTO_3x100-ACT_celu-DKP_1.0-LD_2-PRETRAIN-AE_1-RS_7-C_lstm-R_1x32-SL_40-LFO_0-LFL_1                |
# hyper_params_dict["overfitting_patience"]       = [20]
# hyper_params_dict["sequence_length"]            = [40]
# hyper_params_dict["n_warmup"]                   = [20]
# hyper_params_dict["retrain"]                    = [0]
# hyper_params_dict["train_RNN_only"]             = [1]
# hyper_params_dict["load_trained_AE"]            = [1]
# hyper_params_dict["reconstruction_loss"]        = [0]
# hyper_params_dict["output_forecasting_loss"]    = [0]
# hyper_params_dict["latent_forecasting_loss"]    = [1]

# hyper_params_dict["latent_state_dim"]           = [2]

# hyper_params_dict["RNN_cell_type"]              = ["lstm"]
# hyper_params_dict["RNN_layers_num"]             = [1]
# hyper_params_dict["RNN_layers_size"]            = [32]
# hyper_params_dict["RNN_activation_str_output"]  = ["tanhplus"]

# hyper_params_dict["write_to_log"]               = [1]
# hyper_params_dict["mode"]                       = ["multiscale"]

# multiscale_micro_steps=10
# multiscale_macro_steps_list=[10, 50, 100, 200, 1000]
# plot_multiscale_results_comparison=1
# max_jobs_per_run = 1




# List of all hyper parameter combinations (list of dictionaries) to run
hyper_params_dictionary_list = [
    dict(zip(hyper_params_dict.keys(), v))
    for v in product(*hyper_params_dict.values())
]

# ##################################################
# ##################################################
# ## FOR PLOTTING PURPOSES
# ##################################################
# ##################################################
# hyper_params_dict["make_videos"]       = [1]
# hyper_params_dict["write_to_log"]     = [0]
# hyper_params_dict["mode"]       = ["plot"]
# hyper_params_dict["write_to_log"] = [0]


# ##################################################
# ##################################################
# ## FOR DEBUGGING PURPOSES
# ##################################################
# ##################################################
# hyper_params_dictionary_list = hyper_params_dictionary_list[-1:]
# hyper_params_dictionary_list[0]["mode"]       = "all"
# hyper_params_dictionary_list[0]["truncate_data_batches"]     = 64
# # hyper_params_dictionary_list[0]["mode"]       = "test"
# hyper_params_dictionary_list[0]["max_epochs"]       = 1
# hyper_params_dictionary_list[0]["num_test_ICS"]     = 1
# hyper_params_dictionary_list[0]["write_to_log"]     = 0
# # hyper_params_dictionary_list[0]["write_to_log"]     = 1
# hyper_params_dictionary_list[0]["prediction_horizon"]     = 200
# hyper_params_dictionary_list[0]["test_on_test"]     = 1
# hyper_params_dictionary_list[0]["test_on_val"]     = 1
# max_jobs_per_run = 1











print("NUMBER OF HYPER-PARAMETER COMBINATIONS:")
print(len(hyper_params_dictionary_list))

# print(ark)





#####################################################################
# RUNNING ALL
#####################################################################
PATH = "/users/pvlachas/LED/Code/Methods/"
model_name = "crnn"



total_jobs = len(hyper_params_dictionary_list)
if total_jobs % max_jobs_per_run != 0:
    raise ValueError("total_jobs (={:}) % max_jobs_per_run (={:}) != 0".format(
        total_jobs, max_jobs_per_run))

num_runs = int(total_jobs // max_jobs_per_run)
print("Total number of runs {:}".format(num_runs))

if not IS_MULTISCALE_TESTING:
    for run_num in range(num_runs):
        with open("./Tasks/P0_B{:}_greasy_AE_RNN_tasks.txt".format(run_num + 1),
                  "w") as file:
            for dict_num in range(run_num * max_jobs_per_run,
                                  (run_num + 1) * max_jobs_per_run):
                hyper_param_dict_case = hyper_params_dictionary_list[dict_num]

                command = "[@ {:} @] python3 RUN.py {:}".format(PATH, model_name)
                for key, value in hyper_param_dict_case.items():
                    # print(key)
                    command += " --{:} {:}".format(key, value)
                command += ";\n"
                file.write(command)
else:
    for run_num in range(num_runs):
        with open("./Tasks/P0T_B{:}_greasy_AE_RNN_tasks.txt".format(run_num + 1),
                  "w") as file:
            for dict_num in range(run_num * max_jobs_per_run,
                                  (run_num + 1) * max_jobs_per_run):
                hyper_param_dict_case = hyper_params_dictionary_list[dict_num]

                # command = "[@ {:} @] python3 RUN.py {:}".format(PATH, model_name)
                command = "[@ {:} @] python3 RUN.py ".format(PATH)
                command += "--multiscale_testing 1 "
                command += "--plot_multiscale_results_comparison {:} ".format(plot_multiscale_results_comparison)
                command += "--multiscale_micro_steps_list {:} ".format(multiscale_micro_steps)
                for macro_step in multiscale_macro_steps_list:
                    command += "--multiscale_macro_steps_list {:} ".format(macro_step)

                command += "{:} ".format(model_name)

                for key, value in hyper_param_dict_case.items():
                    # print(key)
                    command += " --{:} {:}".format(key, value)
                command += ";\n"
                file.write(command)




