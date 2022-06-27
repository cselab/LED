from itertools import product

hyper_params_dict = {
    "mode": ["all"],
    "system_name": ["FHN"],
    "write_to_log": [1],
    "channels": [1],
    "input_dim": [2],
    "perform_dim_red": [1],
    "batch_size": [32],
    "Dx": [101],
    "dimred_method": ["pca"],
    "latent_state_dim": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 24, 28, 32, 36, 40, 64],
    "scaler": ["MinMaxZeroOne"],
    "display_output": [1],
    "num_test_ICS": [100],
    "prediction_horizon": [500],
    "plotting":[0],
    "test_on_test": [1],
    "test_on_val": [1],
    "test_on_train": [1],
}

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


max_jobs_per_run = 20
# max_jobs_per_run = 1


# ##################################################
# ##################################################
# ## FOR DEBUGGING PURPOSES
# ##################################################
# ##################################################
# hyper_params_dictionary_list = hyper_params_dictionary_list[-1:]
# hyper_params_dictionary_list[0]["mode"]       = "all"
# # hyper_params_dictionary_list[0]["mode"]       = "train"
# # hyper_params_dictionary_list[0]["truncate_data_batches"]     = 2
# hyper_params_dictionary_list[0]["batch_size"]     = 32
# # hyper_params_dictionary_list[0]["mode"]       = "test"
# hyper_params_dictionary_list[0]["num_test_ICS"]     = 1
# hyper_params_dictionary_list[0]["write_to_log"]     = 0
# hyper_params_dictionary_list[0]["prediction_horizon"]     = 100
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
model_name = "dimred_rnn"



total_jobs = len(hyper_params_dictionary_list)
if total_jobs % max_jobs_per_run != 0:
    raise ValueError("total_jobs (={:}) % max_jobs_per_run (={:}) != 0".format(
        total_jobs, max_jobs_per_run))

num_runs = int(total_jobs // max_jobs_per_run)
print("Total number of runs {:}".format(num_runs))

for run_num in range(num_runs):
    with open("./Tasks/S1_B{:}_greasy_PCA_tasks.txt".format(run_num + 1),
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
