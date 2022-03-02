from itertools import product

hyper_params_dict = {
    # "mode": ["all"],
    # "mode": ["test"],
    "system_name": ["cylRe100HR", "cylRe1000HR", "cylRe100HRDt005", "cylRe1000HRDt005"],
    # "system_name": ["cylRe1000HR"],
    "write_to_log": [1],
    "channels": [2],
    "input_dim": [4],
    "Dx": [1024],
    "Dy": [512],
    # "perform_dim_red": [1],
    "batch_size": [16],
    "dimred_method": ["pca"],
    # "latent_state_dim": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 32],
    "scaler": ["MinMaxZeroOne"],
    "display_output": [1],
    "num_test_ICS": [1],
    "prediction_horizon": [100],
    "test_on_test": [1],
    "test_on_val": [1],
    "test_on_train": [0],
    "plot_state_distributions": [0],
    "plot_system": [1],
    "plot_latent_dynamics": [1],
    "plot_testing_ics_examples": [1],
    "plotting":[1],
}

##################################################
##################################################
## RNN
##################################################
##################################################
IS_MULTISCALE_TESTING = False
hyper_params_dict["random_seed"]                = [17]
hyper_params_dict["random_seed_in_name"]        = [1]
hyper_params_dict["overfitting_patience"]       = [40]
hyper_params_dict["retrain"]                    = [0]

hyper_params_dict["RNN_activation_str"]         = ["tanh"]
hyper_params_dict["RNN_activation_str_output"]  = ["tanhplus"]
hyper_params_dict["RNN_cell_type"]              = ["lstm"]
hyper_params_dict["RNN_layers_size"]            = [16, 32, 64]
hyper_params_dict["RNN_layers_num"]             = [1]

hyper_params_dict["output_forecasting_loss"]    = [0]
hyper_params_dict["latent_forecasting_loss"]    = [1]

hyper_params_dict["sequence_length"]            = [10, 25]
hyper_params_dict["n_warmup"]                   = [10]

hyper_params_dict["max_epochs"]     = [10000]
hyper_params_dict["max_rounds"]     = [10]
hyper_params_dict["learning_rate"]  = [0.01]
hyper_params_dict["batch_size"]     = [16]
hyper_params_dict["optimizer_str"]  = ["adabelief"]

hyper_params_dict["iterative_latent_forecasting"]   = [1]
hyper_params_dict["write_to_log"]                   = [1]
# hyper_params_dict["mode"]                           = ["all"]

##################################################
## TEST
##################################################
hyper_params_dict["mode"]                           = ["test"]
hyper_params_dict["num_test_ICS"]                   = [10]
hyper_params_dict["prediction_horizon"]             = [100]

max_jobs_per_run = 24

# max_jobs_per_run = 4


# 20 steps: 10 minutes
# 4 * 500 steps : 4 * 500 / 10 = 200 minutes = 3.5 hours

# List of all hyper parameter combinations (list of dictionaries) to run
hyp_dict_list = [
    dict(zip(hyper_params_dict.keys(), v))
    for v in product(*hyper_params_dict.values())
]

for i in range(len(hyp_dict_list)):
    if "cylRe100HR" in hyp_dict_list[i]["system_name"]:
        hyp_dict_list[i]["latent_state_dim"]             = 4

    elif "cylRe1000HR" in hyp_dict_list[i]["system_name"]:
        hyp_dict_list[i]["latent_state_dim"]             = 10

    else:
        raise ValueError("Not implemented.")
















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
# hyp_dict_list = hyp_dict_list[-1:]
# hyp_dict_list[0]["mode"]       = "test"
# # hyp_dict_list[0]["mode"]       = "all"
# # hyp_dict_list[0]["mode"]       = "train"
# hyp_dict_list[0]["batch_size"]     = 16
# hyp_dict_list[0]["max_epochs"]     = 1
# hyp_dict_list[0]["truncate_data_batches"]     = 16
# hyp_dict_list[0]["truncate_timesteps"]     = 21
# hyp_dict_list[0]["sequence_length"]     = 10
# hyp_dict_list[0]["RNN_layers_size"]     = 16
# hyp_dict_list[0]["num_test_ICS"]     = 1
# hyp_dict_list[0]["write_to_log"]     = 0
# # hyp_dict_list[0]["write_to_log"]     = 1
# hyp_dict_list[0]["prediction_horizon"]     = 2
# hyp_dict_list[0]["test_on_test"]     = 1
# hyp_dict_list[0]["test_on_val"]     = 1

# hyp_dict_list[0]["iterative_latent_forecasting"]     = 1
# # hyp_dict_list[0]["teacher_forcing_forecasting"]     = 1

# hyp_dict_list[0]["latent_state_dim"]     = 5
# hyp_dict_list[0]["system_name"]     = "cylRe1000HRDt005"

# max_jobs_per_run = 1






print("NUMBER OF HYPER-PARAMETER COMBINATIONS:")
print(len(hyp_dict_list))

# print(ark)



#####################################################################
# RUNNING ALL
#####################################################################
PATH = "/users/pvlachas/STF/Code/Methods/"
model_name = "dimred_rnn"



total_jobs = len(hyp_dict_list)
if total_jobs % max_jobs_per_run != 0:
    raise ValueError("total_jobs (={:}) % max_jobs_per_run (={:}) != 0".format(
        total_jobs, max_jobs_per_run))

num_runs = int(total_jobs // max_jobs_per_run)
print("Total number of runs {:}".format(num_runs))



for run_num in range(num_runs):
    with open("./Tasks/A1P1_B{:}_greasy_PCA_RNN_tasks.txt".format(run_num + 1),
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
