import numpy as np
from utils import load_data, sample_data, get_config_dict
import os
import sys


def evaluate_bin(yf, t, tau_true, tau_pred):
    # print("yf: ", yf.shape, np.mean(yf), np.std(yf))
    tau_true = tau_true.flatten()
    tau_pred = tau_pred.flatten()
    print("tau_true: ", tau_true.shape, np.mean(tau_true), np.std(tau_true))
    print("tau_pred: ", tau_pred.shape, np.mean(tau_pred), np.std(tau_pred))
    pehe = np.sqrt(np.mean(np.square(tau_pred - tau_true)))  # PEHE error
    random_pehe = np.sqrt(np.mean(np.square(np.random.uniform(0, 1) - tau_true)))

    ate_pred = np.mean(tau_pred)
    atc_pred = np.mean(tau_pred[(1 - t) > 0])
    att_pred = np.mean(tau_pred[t > 0])

    att = np.mean(tau_true[t > 0])
    ate = np.mean(tau_true)

    bias_att = np.abs(att_pred - att)  # the error of att
    bias_ate = np.abs(ate_pred - ate)  # the error of ate

    return {"E_pehe": pehe, "E_att": bias_att, "E_ate": bias_ate, "ate_pred": ate_pred, "ate_true": ate}


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_eval_result(result_str, result_file):
    with open(result_file, 'a') as f:
        f.write('%s\n' % result_str)


def evaluation(model_args):
    cfg = get_config_dict(model_args)
    if not cfg:
        print("No config files found:", model_args["config"])
        exit(1)
    
    print("log testing ...")
    print("cfg:{}".format(cfg))

    
    print("training dataset loading ...")
    data_train_path = cfg.data_path
    next_util_rate = cfg.next_util_rate
    feature_util_rate = cfg.feature_util_rate
    data_dict = load_data(data_train_path, feature_util_rate, next_util_rate)
    print("training dataset loading ...done.")

    eval_result_txt = "{}/eval_result.txt".format(cfg.pred_output_dir)
    eval_result_summary_txt = "{}/eval_result_summary.txt".format(cfg.pred_output_dir)
    if os.path.exists(eval_result_txt) and os.path.exists(eval_result_summary_txt):
        print(f"Evaluation results already exist for model {cfg.model_name}, skipping evaluation.")
        return

    print("evaluation result loading ...")
    trainset_result = "{}/{}_train_result.test.npz".format(cfg.pred_output_dir, cfg.model_name)
    validset_result = "{}/{}_valid_result.test.npz".format(cfg.pred_output_dir, cfg.model_name)
    testset_result = "{}/{}_test_result.test.npz".format(cfg.pred_output_dir, cfg.model_name)
    if not os.path.exists(testset_result):
        print(f"Original test result file not found: {testset_result}")
    dict_train_result = np.load(trainset_result)
    dict_valid_result = np.load(validset_result)
    dict_test_result = np.load(testset_result)
    print("evaluation result loading ...done.")

    test_eval_result = {"E_pehe": [], "E_att": [], "E_ate": [], "ate_pred": [], "ate_true": [], "key_loss": []}
    train_eval_result = {"E_pehe": [], "E_att": [], "E_ate": [], "ate_pred": [], "ate_true": [], "key_loss": []}
    valid_eval_result = {"E_pehe": [], "E_att": [], "E_ate": [], "ate_pred": [], "ate_true": [], "key_loss": []}

    for i_exp in range(cfg.n_experiments):
        '''split the dataset'''
        num_samples = cfg.num_samples
        train_data_dict, valid_data_dict, test_data_dict = sample_data(
                data_dict,
                train_ratio=cfg.train_rate,
                valid_ratio=cfg.val_rate,
                seed=i_exp,
                num_samples=num_samples,
                fix_testset=cfg.fix_testset
            )
        
        # if i_exp >= dict_test_result["p_tau"].shape[1]:
        #     break
        x_train = train_data_dict["x"]
        x_valid = valid_data_dict["x"]
        x_test = test_data_dict["x"]

        next_x_train = train_data_dict["next_x"]
        next_x_valid = valid_data_dict["next_x"]
        next_x_test = test_data_dict["next_x"]

        yf_train = train_data_dict["yf"]
        yf_valid = valid_data_dict["yf"]
        yf_test = test_data_dict["yf"]

        ycf_train = train_data_dict["ycf"]
        ycf_valid = valid_data_dict["ycf"]
        ycf_test = test_data_dict["ycf"]

        tau_train = train_data_dict["tau"]
        tau_valid = valid_data_dict["tau"]
        tau_test = test_data_dict["tau"]

        t_train = train_data_dict["t"]
        t_valid = valid_data_dict["t"]
        t_test = test_data_dict["t"]

        if "p_tau" in dict_test_result:
            p_tau_test = dict_test_result["p_tau"][:, i_exp, -1]
            p_tau_train = dict_train_result["p_tau"][:, i_exp, -1]
            # p_tau = p_tau.squeeze()
            test_res = evaluate_bin(yf_test, t_test, tau_test, p_tau_test)
            train_res = evaluate_bin(yf_train, t_train, tau_train, p_tau_train)
            
            for k in test_res.keys():
                test_eval_result[k].append(test_res[k])
            for k in train_res.keys():
                train_eval_result[k].append(train_res[k])
        else:
            test_eval_result["ate_pred"] = dict_test_result["ate_pred"]
            test_eval_result["ate_true"] = dict_test_result["ate_true"]
            train_eval_result["ate_pred"] = dict_train_result["ate_pred"]
            train_eval_result["ate_true"] = dict_train_result["ate_true"]
            valid_eval_result["ate_pred"] = dict_valid_result["ate_pred"]
            valid_eval_result["ate_true"] = dict_valid_result["ate_true"]
    
    if "key_loss" in dict_test_result:
        test_eval_result["key_loss"] = dict_test_result["key_loss"]
        train_eval_result["key_loss"] = dict_train_result["key_loss"]
        valid_eval_result["key_loss"] = dict_valid_result["key_loss"]


    result_str = ""
    res = ""
    result_str += "----test set----\n"
    # print("----test set. split line ----")

    if "p_ycf" in dict_test_result:
        try:
            y_cf_var = np.mean(np.var(dict_test_result["p_ycf"].reshape(-1, dict_test_result["p_ycf"].shape[1]), axis=1))
            result_str += "y_cf_var: " + str(round(y_cf_var, 6)) + "\n"
            res += "y_cf_var" + " " + str(round(y_cf_var, 6)) + " "
            y_f_var = np.mean(np.var(dict_test_result["p_yf"].reshape(-1, dict_test_result["p_yf"].shape[1]), axis=1))
            result_str += "y_f_var: " + str(round(y_f_var, 6)) + "\n"
            res += "y_f_var" + " " + str(round(y_f_var, 6)) + " "
        except:
            print("variance error")

    for k in test_eval_result.keys():
        if k == "ate_pred" or k == "ate_true":
            continue
        val = np.mean(test_eval_result[k])
        std = np.std(test_eval_result[k]) / np.sqrt(cfg.n_experiments)
        # print(k + ": %.6f" % val + " +/- %.6f" % std)
        result_str += str(k) + ": " + str(round(val, 6)) + "+/- " + str(round(std, 6)) + "\n"
        if k in ["E_pehe", "E_ate"]:
            res += str(k) + " " + str(round(val, 6)) + " +/- " + str(round(std, 6)) + " "
        
        val = np.mean(train_eval_result[k])
        std = np.std(train_eval_result[k]) / np.sqrt(cfg.n_experiments)
        result_str += str(k) + ": " + str(round(val, 6)) + "+/- " + str(round(std, 6)) + "\n"
        if k in ["E_pehe", "E_ate"]:
            res += str(k + "_in") + " " + str(round(val, 6)) + " +/- " + str(round(std, 6)) + " "

    print("test_eval_result['ate_pred']: ", test_eval_result["ate_pred"])
    print("test_eval_result['ate_true']: ", test_eval_result["ate_true"])
    ate_mse = np.mean(np.square(np.array(test_eval_result["ate_pred"]) - np.array(test_eval_result["ate_true"])))
    ate_mse_std = np.std(np.square(np.array(test_eval_result["ate_pred"]) - np.array(test_eval_result["ate_true"]))) / np.sqrt(cfg.n_experiments)
    ate_square_bias = np.square(
        np.mean(np.array(test_eval_result["ate_pred"]) - np.array(test_eval_result["ate_true"])))
    ate_variance = np.mean(
        np.square(np.array(test_eval_result["ate_pred"]) - np.mean(np.array(test_eval_result["ate_pred"]))))
    
    # print("ate_mse: ", ate_mse, "ate_square_bias: ", ate_square_bias, "ate_variance: ", ate_variance)
    res += " ate_mse: " + str(round(ate_mse, 6)) + " +/- " + str(round(ate_mse_std, 6)) + " ate_square_bias: " + str(
        round(ate_square_bias, 6)) + " ate_variance: " + str(round(ate_variance, 6)) + " key_loss: " + str(
        round(np.mean(test_eval_result["key_loss"]), 6)) if "key_loss" in test_eval_result else ""
    result_str += "ate_mse: " + str(round(ate_mse, 6)) + " +/- " + str(round(ate_mse_std, 6)) + " ate_square_bias: " + str(
        round(ate_square_bias, 6)) + " ate_variance: " + str(round(ate_variance, 6)) + " key_loss: " + str(
        round(np.mean(test_eval_result["key_loss"]), 6)) if "key_loss" in test_eval_result else "" + "\n"
    # model_name = "nokl_" + model_name
    print("res: ", res)
    save_eval_result("{} {}".format(cfg.model_name, result_str), eval_result_txt)

    save_eval_result("{} {}".format(cfg.model_name, res), eval_result_summary_txt)
    print("done.")
