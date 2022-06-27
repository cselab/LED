from itertools import product

hyper_params_dict = {
    "mode": ["all"],
    # "mode": ["test"],
    "system_name": ["cylRe100HR", "cylRe1000HR"],
    # "system_name": ["cylRe1000HR"],
    # "system_name": ["cylRe100HRDt005", "cylRe1000HRDt005"],
    "write_to_log": [1],
    "channels": [2],
    "input_dim": [4],
    "Dx": [1024],
    "Dy": [512],
    "dimred_method": ["pca"],
    "latent_state_dim": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 32],
    # "latent_state_dim": [4],
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

# List of all hyper parameter combinations (list of dictionaries) to run
hyp_dict_list = [
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


max_jobs_per_run = 28
# max_jobs_per_run = 1


# ##################################################
# ##################################################
# ## FOR DEBUGGING PURPOSES
# ##################################################
# ##################################################
# hyp_dict_list = hyp_dict_list[-1:]
# hyp_dict_list[0]["latent_state_dim"]     = 5
# hyp_dict_list[0]["mode"]       = "test"
# # hyp_dict_list[0]["mode"]       = "all"
# # hyp_dict_list[0]["mode"]       = "train"
# hyp_dict_list[0]["truncate_data_batches"]     = 1
# hyp_dict_list[0]["truncate_timesteps"]     = 10
# # hyp_dict_list[0]["mode"]       = "test"
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
model_name = "dimred"



total_jobs = len(hyp_dict_list)
if total_jobs % max_jobs_per_run != 0:
    raise ValueError("total_jobs (={:}) % max_jobs_per_run (={:}) != 0".format(
        total_jobs, max_jobs_per_run))

num_runs = int(total_jobs // max_jobs_per_run)
print("Total number of runs {:}".format(num_runs))

for run_num in range(num_runs):
    with open("./Tasks/A1_B{:}_greasy_PCA_tasks.txt".format(run_num + 1),
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
