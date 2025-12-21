try:
    import torch
    from torch import nn, optim
    from torch.utils.tensorboard import SummaryWriter
    import torch.nn.functional as F
    torch.autograd.set_detect_anomaly(True)
except:
    print("Import torch error")
import numpy as np
import math
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_predict
from sklearn.exceptions import ConvergenceWarning
import logging
import sys, os, shutil
import time
from pathlib import Path
import traceback, warnings
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import defaultdict

try:
    from model.models import ShareNetwork, PrpsyNetwork, Mu1Network, Mu0Network, TauNetwork, PIPCFR, DRCFR, PrpsyNetworkTrainer
    from model.models import TransformerNetwork, PIPCFR_on_DRCFR
    from model.dataset import ESXDataset
    from model.util import wasserstein_torch, mmd2_torch
except:
    print("Import model or utils error")

from utils import seed_torch, get_logger, get_config_dict, load_data, sample_data, convert_next_x_transformer, get_sample_weight
from utils import convert_next_x_transformer_realworld

# dependencies for Meta Learner 
try:
    from sklearn.linear_model import LogisticRegression, Lasso
    from xgboost import XGBRegressor
    from sklearn.neural_network import MLPRegressor 
    from causalml.inference.meta import BaseXRegressor, BaseDRRegressor, BaseRRegressor
    from causalml.inference.tree import CausalRandomForestRegressor
except:
    print("Import causalml error")

# dependencies for Pairnet
try:
    from catenets.models.jax import TARNet
    from catenets.models.jax import PairNet
    from catenets.datasets.torch_dataset import PairDataset
except Exception as e:
    print(f"Import pairnet error: {e}")
    # traceback.print_exc()

# exit(1)

# dependencies for ESCFR
try:
    import ot
    from ot4tee.escfr import EscfrEstimator
    from ot4tee.data_processor import Escfr_ESXDataset
except Exception as e:
    print(f"Import pairnet error: {e}")
    # traceback.print_exc()

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
WORK_DIR = Path().resolve()

logger = get_logger()

def weighted_ate_error(input, target, weight=1):
    risk = torch.mean(weight * input) - torch.mean(weight * target)
    return risk

def cal_total_loss(inputs, next_inputs, t_labels, y_labels, model, cfg, eff_tau=None, valid_stage=False):
    device = cfg.device
    t1 = time.time()
    # set loss function
    loss_mse = nn.MSELoss()
    loss_with_logit_fn = nn.BCEWithLogitsLoss()

    if  'drcfr' in cfg.teacher_model_name:
        p_prpsy_logit, p_prpsy, p_prpsy_logit_main, p_prpsy_main, p_prpsy_logit_hs, p_prpsy_hs, \
        shared_h, final_gamma, final_delta, final_upsilon, p_mu1_logit, p_mu0_logit, p_mu1, p_mu0, p_h1, p_h0, \
        all_mu1_logit, all_mu0_logit, all_prpsy_nograd_logit, all_prpsy_logit, all_p_prpsy_nograd, all_p_prpsy, \
        hs_mu1_logit, hs_mu0_logit, hs_p_mu1, hs_p_mu0 = model(inputs, next_inputs)
        final_shared_h = final_upsilon
        p_estr, p_escr = None, None
        p_tau_logit = p_mu1_logit - p_mu0_logit
    elif cfg.tarreg:
        p_prpsy_logit, p_estr, p_escr, p_tau_logit, p_mu1_logit, p_mu0_logit, p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h, final_shared_h, \
            all_mu1_logit, all_mu0_logit, hs_mu1_logit, hs_mu0_logit, all_prpsy_nograd_logit, all_prpsy_logit, \
            all_p_mu1, all_p_mu0, hs_p_mu1, hs_p_mu0, all_p_prpsy_nograd, all_p_prpsy, \
            eps, p_mu1_logit_pert, p_mu0_logit_pert, all_eps, all_mu1_logit_pert, all_mu0_logit_pert = model(
            inputs, next_inputs)
    else:
        p_prpsy_logit, p_estr, p_escr, p_tau_logit, p_mu1_logit, p_mu0_logit, p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h, final_shared_h, \
            all_mu1_logit, all_mu0_logit, hs_mu1_logit, hs_mu0_logit, all_prpsy_nograd_logit, all_prpsy_logit, \
            all_p_mu1, all_p_mu0, hs_p_mu1, hs_p_mu0, all_p_prpsy_nograd, all_p_prpsy = model(inputs,
                                                                                                next_inputs)
    logger.info(f"forward: {time.time()-t1}")
    t1 = time.time()

    
    p_t = torch.mean(t_labels).item()
    prpsy_loss = cfg.prpsy_w * loss_with_logit_fn(p_prpsy_logit, t_labels)
    if 'cfrnet' in cfg.model_name:
        tr_loss = cfg.h1_w * loss_mse(p_mu1_logit[t_labels.bool()],
                                    y_labels[t_labels.bool()]) * torch.tensor(1 / (2 * p_t))
        cr_loss = cfg.h0_w * loss_mse(p_mu0_logit[~t_labels.bool()],
                                    y_labels[~t_labels.bool()]) * torch.tensor(1 / (2 * (1 - p_t)))
    else:
        tr_loss = cfg.h1_w * loss_mse(p_mu1_logit[t_labels.bool()],
                                    y_labels[t_labels.bool()])
        cr_loss = cfg.h0_w * loss_mse(p_mu0_logit[~t_labels.bool()],
                                    y_labels[~t_labels.bool()])

    counter_factaul_loss = cfg.cf_w * (
            loss_mse(p_mu1_logit[~t_labels.bool()], all_mu1_logit[~t_labels.bool()].detach()) +
            loss_mse(p_mu0_logit[t_labels.bool()], all_mu0_logit[t_labels.bool()].detach()))
    factaul_loss = cfg.f_w * (
            loss_mse(p_mu1_logit[t_labels.bool()], all_mu1_logit[t_labels.bool()].detach()) +
            loss_mse(p_mu0_logit[~t_labels.bool()], all_mu0_logit[~t_labels.bool()].detach()))
    all_prpsy_loss = cfg.all_prpsy_w * loss_with_logit_fn(all_prpsy_nograd_logit,
                                                            t_labels)
    align_loss = cfg.align_w * p_t * torch.mean((p_mu1_logit[t_labels.bool()] - y_labels[t_labels.bool()]) * (all_mu0_logit[t_labels.bool()].detach() - p_mu0_logit[t_labels.bool()]))
    align_loss += cfg.align_w * (1-p_t) * torch.mean((p_mu0_logit[~t_labels.bool()] - y_labels[~t_labels.bool()]) * (all_mu1_logit[~t_labels.bool()].detach() - p_mu1_logit[~t_labels.bool()]))
    if 'cfrnet' in cfg.teacher_model_name:
        all_y_loss = cfg.all_y_w * (
                    loss_mse(all_mu1_logit[t_labels.bool()], y_labels[t_labels.bool()]) * torch.tensor(
                1 / (2 * p_t)) +
                    loss_mse(all_mu0_logit[~t_labels.bool()], y_labels[~t_labels.bool()]) * torch.tensor(
                1 / (2 * (1 - p_t))))
    elif 'drcfr' in cfg.teacher_model_name:
        w_t = t_labels * (1 + p_t / (1 - p_t)) * ((1 - all_p_prpsy_nograd) / all_p_prpsy_nograd)
        w_c = (1 - t_labels) / (1 + (1 - p_t) / p_t) * (all_p_prpsy_nograd / (1 - all_p_prpsy_nograd))
        sample_weight = w_t + w_c
        loss_w_fn = nn.BCELoss(weight=sample_weight)
        all_y_logit = all_mu1_logit * t_labels + all_mu0_logit * (1 - t_labels)
        all_y_loss = cfg.all_y_w * loss_w_fn(all_y_logit, y_labels)
    else:
        all_y_loss = cfg.all_y_w * (
                    loss_mse(all_mu1_logit[t_labels.bool()], y_labels[t_labels.bool()]) +
                    loss_mse(all_mu0_logit[~t_labels.bool()], y_labels[~t_labels.bool()]))

    hs_kl_loss = cfg.hs_kl_w * (
        torch.mean(
            all_p_prpsy.detach() * (torch.log(all_p_prpsy.detach()) - torch.log(p_prpsy)) +
            (1 - all_p_prpsy.detach()) * (torch.log(1 - all_p_prpsy.detach()) - torch.log(1 - p_prpsy))
        ))

    if 'pipcfr' in cfg.model_name:
        total_loss = prpsy_loss + tr_loss + cr_loss + all_y_loss + hs_kl_loss + all_prpsy_loss \
        + counter_factaul_loss + align_loss
        if 'cfrnet' in cfg.teacher_model_name or 'drcfr' in cfg.teacher_model_name:
            if cfg.hs_imb_dist == "wass":
                # print("wass\n"*1000)
                hs_imb_dist = wasserstein_torch(X=final_shared_h, t=t_labels)
            elif cfg.hs_imb_dist == "mmd":
                hs_imb_dist = mmd2_torch(final_shared_h, t_labels)
            else:
                sys.exit(1)
            total_loss += cfg.hs_imb_dist_w * hs_imb_dist
        elif 'escfr' in cfg.teacher_model_name:
            dist = 0.1 * ot.dist(final_shared_h[~t_labels.flatten().bool()],
                                    final_shared_h[t_labels.flatten().bool()])
            dist_10 = ot.dist(all_p_mu1[~t_labels.flatten().bool()], y_labels[t_labels.flatten().bool()])
            dist_01 = ot.dist(y_labels[~t_labels.flatten().bool()], all_p_mu0[t_labels.flatten().bool()])
            dist += 0.0005 * (dist_10 + dist_01)
            gamma = ot.sinkhorn(
                        torch.ones(len(final_shared_h[~t_labels.flatten().bool()]), device=device) / len(final_shared_h[~t_labels.flatten().bool()]),
                        torch.ones(len(final_shared_h[t_labels.flatten().bool()]), device=device) / len(final_shared_h[t_labels.flatten().bool()]),
                        dist.detach(),
                        reg=1.0,
                        stopThr=1e-4)
            hs_imb_dist = torch.sum(gamma * dist)
            total_loss += cfg.hs_imb_dist_w * hs_imb_dist
    elif 'cfrnet' in cfg.model_name or 'drcfr' in cfg.model_name:
        imb_dist = 0
        if cfg.imb_dist_w > 0:
            if cfg.imb_dist == "wass":
                imb_dist = wasserstein_torch(X=shared_h, t=t_labels)
            elif cfg.imb_dist == "mmd":
                imb_dist = mmd2_torch(shared_h, t_labels)
            else:
                sys.exit(1)
        imb_dist_loss = cfg.imb_dist_w * imb_dist
        if 'cfrnet' in cfg.model_name:
            total_loss = tr_loss + cr_loss + imb_dist_loss
        else:
            w_t = t_labels * (1 + p_t / (1 - p_t)) * ((1 - p_prpsy) / p_prpsy)
            w_c = (1 - t_labels) / (1 + (1 - p_t) / p_t) * (p_prpsy / (1 - p_prpsy))
            sample_weight = w_t + w_c
            loss_w_fn = nn.BCELoss(weight=sample_weight)
            y_logit = p_mu1_logit * t_labels + p_mu0_logit * (1 - t_labels)
            y_loss = loss_w_fn(y_logit, y_labels)
            total_loss = y_loss + imb_dist_loss + prpsy_loss
    elif 'dragonnet' in cfg.model_name:
        total_loss = tr_loss + cr_loss + prpsy_loss
        if cfg.tarreg:
            tr_loss_tar = cfg.tarreg_w * loss_mse(p_mu0_logit_pert[t_labels.bool()],
                                                y_labels[t_labels.bool()])
            cr_loss_tar = cfg.tarreg_w * loss_mse(p_mu1_logit_pert[~t_labels.bool()],
                                                y_labels[~t_labels.bool()])
        total_loss += tr_loss_tar + cr_loss_tar
    elif 'tarnet' in cfg.model_name:
        total_loss = tr_loss + cr_loss
    
    logger.info(f"calculate loss: {time.time()-t1}")
    t1 = time.time()

    if not valid_stage:
        return total_loss
    
    p_h1, p_h0 = p_mu1_logit, p_mu0_logit
    all_p_mu1, all_p_mu0 = all_mu1_logit, all_mu0_logit
    hs_p_mu1, hs_p_mu0 = hs_mu1_logit, hs_mu0_logit
    if cfg.tarreg:
        p_h1, p_h0 = p_mu1_logit_pert, p_mu0_logit_pert

    p_tau = p_h1 - p_h0
    p_yf = p_h1 * t_labels + p_h0 * (1 - t_labels)
    p_ycf = p_h0 * t_labels + p_h1 * (1 - t_labels)
    p_tau_all = all_p_mu1 - all_p_mu0
    print(cfg.use_y_scaler)
    if cfg.use_y_scaler:
        p_tau_np = p_tau.cpu().detach().numpy()
        p_yf_np = p_yf.cpu().detach().numpy()
        p_ycf_np = p_ycf.cpu().detach().numpy()
        p_h1_np = p_h1.cpu().detach().numpy()
        p_h0_np = p_h0.cpu().detach().numpy()
        # print("p_tau_np:", p_tau_np.shape, p_tau_np[:10])
        # print("p_yf_np:", p_yf_np[:10])
        # print("p_ycf_np:", p_ycf_np[:10])

        print(model.y_scaler)
        # 反归一化预测结果
        p_yf_original = torch.from_numpy(model.y_scaler.inverse_transform(p_yf_np)).float().to(model.device)
        p_ycf_original = torch.from_numpy(model.y_scaler.inverse_transform(p_ycf_np)).float().to(model.device)
        p_h1_original = torch.from_numpy(model.y_scaler.inverse_transform(p_h1_np)).float().to(model.device)
        p_h0_original = torch.from_numpy(model.y_scaler.inverse_transform(p_h0_np)).float().to(model.device)

        # 重新计算tau
        p_tau_original = p_h1_original - p_h0_original

        # 用于计算metrics的变量使用反归一化后的结果
        p_tau_for_metrics = p_tau_original
        p_yf_for_metrics = p_yf_original
        p_ycf_for_metrics = p_ycf_original

        print("p_tau_for_metrics: ", p_tau_for_metrics[:10])
        # print("p_yf_for_metrics: ", p_yf_for_metrics[:10])
        # print("p_ycf_for_metrics: ", p_ycf_for_metrics[:10])
        print("eff_tau: ", eff_tau[:10])

        # 反归一化 all_p_mu1 和 all_p_mu0
        all_p_mu1_np = all_p_mu1.cpu().detach().numpy()
        all_p_mu0_np = all_p_mu0.cpu().detach().numpy()
        all_p_mu1_original = torch.from_numpy(model.y_scaler.inverse_transform(all_p_mu1_np)).float().to(model.device)
        all_p_mu0_original = torch.from_numpy(model.y_scaler.inverse_transform(all_p_mu0_np)).float().to(model.device)
        all_p_tau_original = all_p_mu1_original - all_p_mu0_original
        all_p_tau_for_metrics = all_p_tau_original
    else:
        p_tau_for_metrics = p_tau
        p_yf_for_metrics = p_yf
        p_ycf_for_metrics = p_ycf
        all_p_tau_for_metrics = p_tau_all

    pehe, ate_error = None, None
    if eff_tau is not None:
        pehe = torch.sqrt(torch.mean((p_tau_for_metrics - eff_tau) ** 2)).item()
        ate_error = weighted_ate_error(input=p_tau_for_metrics, target=eff_tau)
    dict_result = {
        "loss": [total_loss.cpu().detach().numpy()],
        "p_tau": (
            p_tau_for_metrics if hasattr(cfg, 'use_y_scaler') and cfg.use_y_scaler else p_tau).cpu().detach().numpy(),
        "p_yf": (
            p_yf_for_metrics if hasattr(cfg, 'use_y_scaler') and cfg.use_y_scaler else p_yf).cpu().detach().numpy(),
        "p_ycf": (
            p_ycf_for_metrics if hasattr(cfg, 'use_y_scaler') and cfg.use_y_scaler else p_ycf).cpu().detach().numpy(),
        "p_prpsy": p_prpsy.cpu().detach().numpy(),
        "pehe": pehe,
        "ate_error": ate_error,
        "key_loss": np.sum([loss.cpu().detach().numpy() if isinstance(loss, torch.Tensor) else loss for loss in [
            counter_factaul_loss, factaul_loss, all_y_loss,]])
    }
    return dict_result


def evalWithData(group_name, model, writer, cfg, x, next_x, yf, t, eff_tau=None):
    logging.info("group_name:{}, evalWithData... -----------------------------------".format(group_name))
    t1 = time.time()
    x = torch.from_numpy(x).float().to(model.device)
    next_x = torch.from_numpy(next_x).float().to(model.device)
    yf = torch.from_numpy(yf).float().reshape((-1, 1)).to(model.device)
    t = torch.from_numpy(t).float().reshape((-1, 1)).to(model.device)
    if eff_tau is not None:
        eff_tau = torch.from_numpy(eff_tau).float().reshape((-1, 1)).to(model.device)
    writer_flag = not writer is None
    dict_result = cal_total_loss(x, next_x, t, yf, model, cfg, eff_tau=eff_tau, valid_stage=True)
    
    return dict_result


def train(data_dict, device, cfg):
    share_dim = cfg.share_dim
    base_dim = cfg.base_dim
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    log_step = cfg.log_step
    eval_step = cfg.eval_step
    num_samples = cfg.num_samples
    early_stop_patience = cfg.early_stop_patience
    early_stop_mindelta = cfg.early_stop_mindelta
    pred_output_dir = cfg.pred_output_dir
    if not os.path.exists(pred_output_dir):
        os.makedirs(pred_output_dir)
    eval_result_txt = "{}/eval_result.txt".format(pred_output_dir)
    eval_result_summary_txt = "{}/eval_result_summary.txt".format(pred_output_dir)
    if os.path.exists(eval_result_txt) and os.path.exists(eval_result_summary_txt):
        print(f"Evaluation results already exist for model {cfg.model_name}, skipping evaluation.")
        return

    y_scaler = data_dict.get('y_scaler', None) if cfg.use_y_scaler else None
    print(y_scaler)
    use_transformer = cfg.use_transformer
    prpsy_indep = cfg.prpsy_indep
    if 'pipcfr' in cfg.model_name:
        if 'dragonnet' in cfg.teacher_model_name:
            prpsy_indep = False
        else:
            prpsy_indep = cfg.prpsy_indep
    cfg.tarreg = cfg.tarreg if 'dragonnet' in cfg.model_name else False

    result_dict = {}
    for group in ["train", "valid", "test"]:
        result_dict[group] = defaultdict(list)

    if cfg.summary_base_dir:
        summary_root = '{}/{}'.format(cfg.summary_base_dir, cfg.model_name)
        if not os.path.exists(summary_root):
            logging.info(" os.mkdir({}) ...".format(summary_root))
            os.makedirs(summary_root)
    else:
        summary_root = None

    if cfg.verbose > 0:
        group_list = ["train", "test", "valid"]
    else:
        group_list = ["test", "valid"]
    
    for i_exp in range(cfg.n_experiments):
        print("i_exp:", i_exp)
        if summary_root:
            '''init summary'''
            summary_path = os.path.join(summary_root, 'exp_{}'.format(i_exp))
            if os.path.exists(summary_path):
                logging.info(" shutil.rmtree({}) ...".format(summary_path))
                shutil.rmtree(summary_path)
                time.sleep(0.5)
            else:
                ''' create summary folder'''
                logging.info(" os.mkdir({}) ...".format(summary_path))
                os.mkdir(summary_path)
            writer = SummaryWriter(summary_path)
        else:
            writer = None

        '''split the dataset'''
        train_data_dict, valid_data_dict, test_data_dict = sample_data(
                data_dict,
                train_ratio=cfg.train_rate,
                valid_ratio=cfg.val_rate,
                seed=i_exp,
                num_samples=num_samples,
                fix_testset=cfg.fix_testset
            )
        
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

        if "tau_original" in train_data_dict:
            tau_train_original = train_data_dict["tau_original"]
            tau_valid_original = valid_data_dict["tau_original"]
            tau_test_original = test_data_dict["tau_original"]
        else:
            tau_train_original = tau_train
            tau_valid_original = tau_valid
            tau_test_original = tau_test

        t_train = train_data_dict["t"]
        t_valid = valid_data_dict["t"]
        t_test = test_data_dict["t"]

        dim = train_data_dict['dim']
        next_x_dim = next_x_train.shape[1]

        x_groups = {"train": x_train, "valid": x_valid, "test": x_test}
        t_groups = {"train": t_train, "valid": t_valid, "test": t_test}
        yf_groups = {"train": yf_train, "valid": yf_valid, "test": yf_test}
        ycf_groups = {"train": ycf_train, "valid": ycf_valid, "test": ycf_test}
        tau_groups = {"train": tau_train, "valid": tau_valid, "test": tau_test}

        ''' Set up for saving variables for each repeat experiment'''
        iexp_p_prpsy = {"train": [None], "valid": [None], "test": [None]}
        iexp_p_yf = {"train": [None], "valid": [None], "test": [None]}
        iexp_p_ycf = {"train": [None], "valid": [None], "test": [None]}
        iexp_p_tau = {"train": [None], "valid": [None], "test": [None]}
        iexp_losses = {"train": [None], "valid": [None], "test": [None]}
        iexp_key_loss = {"train": [None], "valid": [None], "test": [None]}

        if i_exp == 0:
            # for training set
            logging.info("exp_{}, Train. x.shape : {}".format(i_exp, x_train.shape))
            logging.info("exp_{}, Train. next_x.shape : {}".format(i_exp, next_x_train.shape))
            logging.info("exp_{}, Train. mean(t) : {}".format(i_exp, np.mean(t_train)))
            logging.info("exp_{}, Train. mean(yf): {}".format(i_exp, np.mean(yf_train)))
            logging.info("exp_{}, Train. mean(yf) when t=1: {}".format(i_exp, np.mean(yf_train[t_train.astype(bool)])))
            logging.info(
                "exp_{}, Train. mean(yf) when t=0: {}".format(i_exp, np.mean(yf_train[(1 - t_train).astype(bool)])))

            # for validation set
            logging.info("exp_{}, Valid. x.shape : {}".format(i_exp, x_valid.shape))
            logging.info("exp_{}, Valid. next_x.shape : {}".format(i_exp, next_x_valid.shape))
            logging.info("exp_{}, Valid. mean(t) : {}".format(i_exp, np.mean(t_valid)))
            logging.info("exp_{}, Valid. mean(yf): {}".format(i_exp, np.mean(yf_valid)))
            logging.info("exp_{}, Valid. mean(yf) when t=1: {}".format(i_exp, np.mean(yf_valid[t_valid.astype(bool)])))
            logging.info(
                "exp_{}, Valid. mean(yf) when t=0: {}".format(i_exp, np.mean(yf_valid[(1 - t_valid).astype(bool)])))

            # for test set
            logging.info("exp_{}, Test. x.shape : {}".format(i_exp, x_test.shape))
            logging.info("exp_{}, Test. next_x.shape : {}".format(i_exp, next_x_test.shape))
            logging.info("exp_{}, Test. mean(t) : {}".format(i_exp, np.mean(t_test)))
            logging.info("exp_{}, Test. mean(yf): {}".format(i_exp, np.mean(yf_test)))
            logging.info("exp_{}, Test. mean(yf) when t=1: {}".format(i_exp, np.mean(yf_test[t_test.astype(bool)])))
            logging.info(
                "exp_{}, Test. mean(yf) when t=0: {}".format(i_exp, np.mean(yf_test[(1 - t_test).astype(bool)])))

        if 'crf' in cfg.model_name:
            
            params = dict(
                criterion=cfg.crf_criterion,  # 损失函数类型, 'causal_mse' 'standard_mse' or 't_test'
                control_name=0,  # 控制组名称
                min_samples_leaf=cfg.min_samples_leaf,  # 叶节点最小样本数
                groups_penalty=cfg.groups_penalty,  # 组间差异惩罚系数,用于平衡组间差异
                groups_cnt=cfg.groups_cnt,  # 是否计算组数,用于评估组间差异
                max_depth=cfg.max_depth  # 树的最大深度
            )
            cforest = CausalRandomForestRegressor(**params)
            cforest.fit(X=x_train, 
                       treatment=t_train.squeeze(),
                       y=yf_train.squeeze())
            
            for group in group_list:
                p_tau = cforest.predict(x_groups[group], with_outcomes=False)[:,np.newaxis]
                p_tau[np.isnan(p_tau)] = np.nanmean(p_tau)  # Fill NaN values with the mean of non-NaN values
                p_tau = (y_scaler.inverse_transform(p_tau) - y_scaler.data_min_) if hasattr(cfg, 'use_y_scaler') and cfg.use_y_scaler else p_tau
                result_dict[group]["p_tau"].append([p_tau])
                tau_test_o = (y_scaler.inverse_transform(tau_groups[group].reshape(-1, 1)) - y_scaler.data_min_) if hasattr(cfg, 'use_y_scaler') and cfg.use_y_scaler else tau_groups[group].reshape(-1, 1)
                print('ate_pred:', np.mean(p_tau), 'ate_true:', np.mean(tau_test_o))
                print('pehe:', np.sqrt(np.mean(np.square(p_tau - tau_test_o))))
                # print(np.array(result_dict["test"]["p_tau"]).shape)

                all_p_tau = np.array(np.swapaxes(np.swapaxes(result_dict[group]["p_tau"], 0, 2), 1, 2))

                logging.info("saving predict result as a file...")
                npz_file_path = "{}/{}_{}_result.test".format(pred_output_dir, cfg.model_name, group)
                np.savez(npz_file_path, p_tau=all_p_tau)
                logging.info("saving predict result as a file: {}...done".format(npz_file_path))

            continue

        if 'xlearner' in cfg.model_name or 'rlearner' in cfg.model_name or 'DRlearner' in cfg.model_name:
            for name in ['xlearner', 'rlearner', 'DRlearner']:
                if name in cfg.model_name:
                    meta_model_name = name
                    break
            if 'mlp' in cfg.model_name:
                base_learner = MLPRegressor(hidden_layer_sizes=(10, 10),
                                            learning_rate_init=cfg.meta_learning_rate_init,
                                            early_stopping=cfg.meta_early_stopping,
                                            max_iter=cfg.meta_epochs)
            else:
                base_learner = XGBRegressor(random_state=cfg.seed)
            propensity_learner = LogisticRegression(penalty='l1', solver='liblinear')
            propensity_learner.fit(x_train, t_train.flatten())
            clip_v = cfg.meta_clip_v
            print("Start training Propensity Score Model...")
            e_hat_train = cross_val_predict(propensity_learner, x_train, t_train.flatten(), method='predict_proba', cv=cfg.k_folds)[:, 1]
            e_hat_train = (e_hat_train + clip_v) / (1 + clip_v * 2)
            e_hat_test = propensity_learner.predict_proba(x_test)[:, 1]
            e_hat_test = (e_hat_test + clip_v) / (1 + clip_v * 2)
            e_hat_valid = propensity_learner.predict_proba(x_valid)[:, 1]
            e_hat_valid = (e_hat_valid + clip_v) / (1 + clip_v * 2)
            e_hat_groups = {"train": e_hat_train, "valid": e_hat_valid, "test": e_hat_test}
            print("End training!")

            print("Start training XLearner Model...")
            if meta_model_name == 'xlearner':
                meta_learner = BaseXRegressor(learner=base_learner)
            elif meta_model_name == 'rlearner':
                meta_learner = BaseRRegressor(learner=base_learner, n_fold=cfg.k_folds, effect_learner=XGBRegressor(random_state=42))
            elif meta_model_name == 'DRlearner':
                meta_learner = BaseDRRegressor(learner=base_learner)
            meta_learner.fit(X=x_train, treatment=t_train.flatten(), y=yf_train.flatten(), p=e_hat_train)
            print("End training!")

            for group in group_list:
                if meta_model_name == 'rlearner':
                    p_tau = meta_learner.predict(X=x_groups[group], p=e_hat_groups[group]).reshape(-1, 1)
                else:
                    p_tau = meta_learner.predict(X=x_groups[group], treatment=t_groups[group], p=e_hat_groups[group]).reshape(-1, 1)
                p_tau = (y_scaler.inverse_transform(p_tau) - y_scaler.data_min_) if hasattr(cfg, 'use_y_scaler') and cfg.use_y_scaler else p_tau
                result_dict[group]["p_tau"].append([p_tau])

                all_p_tau = np.array(np.swapaxes(np.swapaxes(result_dict[group]["p_tau"], 0, 2), 1, 2))
                logging.info("saving predict result as a file...")
                npz_file_path = "{}/{}_{}_result.test".format(pred_output_dir, cfg.model_name, group)
                np.savez(npz_file_path, p_tau=all_p_tau)
                logging.info("saving predict result as a file: {}...done".format(npz_file_path))

            del meta_learner
            del base_learner
            import gc
            gc.collect()

            continue

        if 'drcfr' in cfg.model_name:
            print("Start training DRCFR...")
            # 创建DRCFR模型
            model = DRCFR(input_dim=data_dict['dim'], 
                        share_dim=cfg.share_dim,
                        base_dim=cfg.base_dim,
                        cfg=cfg,
                        device=device)
            
            # 如果使用y_scaler,将其添加到模型中
            if hasattr(cfg, 'use_y_scaler') and cfg.use_y_scaler: 
                model.y_scaler = data_dict['y_scaler']

            if cfg.optim == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2)
            else:
                optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2)
            
            # 存储每个实验的预测结果
            iexp_p_prpsy = {"train": [None], "valid": [None], "test": [None]}
            iexp_p_yf = {"train": [None], "valid": [None], "test": [None]}
            iexp_p_ycf = {"train": [None], "valid": [None], "test": [None]}
            iexp_p_tau = {"train": [None], "valid": [None], "test": [None]}
            iexp_losses = {"train": [None], "valid": [None], "test": [None]}

            dataset = ESXDataset(x_train, next_x_train, yf_train, t_train)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
            train_step = 0
            for epoch in range(cfg.epochs):
                model.train()
                for i, (inputs, _, t_labels, y_labels) in enumerate(train_loader):
                    inputs = inputs.to(device)
                    t_labels = torch.unsqueeze(t_labels.to(device), 1)
                    y_labels = torch.unsqueeze(y_labels.to(device), 1)
                    
                    ''' Compute sample reweighting '''
                    p_t = torch.mean(t_labels).item()
                    if cfg.reweight_sample:
                        w_t = t_labels / (
                                2 * p_t)
                        w_c = (1 - t_labels) / (2 * (1 - p_t))
                        sample_weight = w_t + w_c
                    else:
                        sample_weight = torch.ones_like(t_labels)
                        p_t = 0.5

                    # set loss functions
                    loss_w_fn = nn.BCELoss(weight=sample_weight)
                    loss_fn = nn.BCELoss()
                    loss_mse = nn.MSELoss()
                    loss_with_logit_fn = nn.BCEWithLogitsLoss()  # for logit
                    loss_w_with_logit_fn = nn.BCEWithLogitsLoss(
                        pos_weight=torch.tensor(1 / (2 * p_t)))  # for propensity loss
                    # loss_w_mse = nn.MSELoss(weight=torch.tensor(1 / (2 * p_t)))

                    optimizer.zero_grad()
                    
                    p_prpsy_logit, p_prpsy, p_mu1, p_mu0, p_tau, p_yf, p_ycf, gamma, delta, upsilon = model(inputs)
                    
                    # 计算损失
                    prpsy_loss = cfg.prpsy_w * loss_w_with_logit_fn(p_prpsy_logit, t_labels)
                    h1_loss = cfg.h1_w * loss_mse(p_mu1[t_labels.bool()], y_labels[t_labels.bool()])
                    h0_loss = cfg.h0_w * loss_mse(p_mu0[~t_labels.bool()], y_labels[~t_labels.bool()])
                    imb_loss = cfg.imb_dist_w * mmd2_torch(upsilon, t_labels) if cfg.imb_dist == "mmd" else cfg.imb_dist_w * wasserstein_torch(X=upsilon, t=t_labels)
                    
                    total_loss = prpsy_loss + h1_loss + h0_loss + imb_loss
                    total_loss.backward()
                    optimizer.step()

                    train_step += 1
                    
                    # 按照train_step进行评估
                    if (train_step + 1) % cfg.eval_step == 0:
                        model.eval()
                        with torch.no_grad():
                            # 验证集评估
                            def eval_and_store_predictions(data, data_name, tau_original=None):
                                """Evaluate model and store predictions"""
                                p_prpsy_logit, p_prpsy, p_mu1, p_mu0, p_tau, p_yf, p_ycf, _, _, _ = model(torch.from_numpy(data).float().to(device))
                                iexp_losses[data_name][0] = torch.ones_like(p_tau).cpu().detach().numpy()
                                
                                if hasattr(cfg, 'use_y_scaler') and cfg.use_y_scaler and y_scaler is not None:
                                    p_tau_np = p_tau.cpu().detach().numpy()
                                    p_yf_np = p_yf.cpu().detach().numpy()
                                    p_ycf_np = p_ycf.cpu().detach().numpy()
                                    p_mu1_np = p_mu1.cpu().detach().numpy()
                                    p_mu0_np = p_mu0.cpu().detach().numpy()
                                    
                                    p_yf_original = torch.from_numpy(y_scaler.inverse_transform(p_yf_np)).float().to(device)
                                    p_ycf_original = torch.from_numpy(y_scaler.inverse_transform(p_ycf_np)).float().to(device)
                                    p_mu1_original = torch.from_numpy(y_scaler.inverse_transform(p_mu1_np)).float().to(device)
                                    p_mu0_original = torch.from_numpy(y_scaler.inverse_transform(p_mu0_np)).float().to(device)
                                    p_tau_original = p_mu1_original - p_mu0_original
                                    
                                    iexp_p_tau[data_name][0] = p_tau_original.cpu().detach().numpy()
                                    iexp_p_yf[data_name][0] = p_yf_original.cpu().detach().numpy()
                                    iexp_p_ycf[data_name][0] = p_ycf_original.cpu().detach().numpy()
                                    
                                    if tau_original is not None and data_name == "test":
                                        pehe = np.sqrt(np.mean(np.square(p_tau_original.cpu().detach().numpy().flatten() - tau_original.flatten())))
                                        print(f"{data_name}_pehe: {pehe}")
                                        if writer is not None:
                                            writer.add_scalar(f"{data_name}_pred_result/PEHE", pehe, train_step)
                                else:
                                    iexp_p_tau[data_name][0] = p_tau.cpu().detach().numpy()
                                    iexp_p_yf[data_name][0] = p_yf.cpu().detach().numpy()
                                    iexp_p_ycf[data_name][0] = p_ycf.cpu().detach().numpy()
                                    
                                    if tau_original is not None and data_name == "test":
                                        pehe = np.sqrt(np.mean(np.square(p_tau.cpu().detach().numpy().flatten() - tau_original.flatten())))
                                        print(f"{data_name}_pehe: {pehe}")
                                        if writer is not None:
                                            writer.add_scalar(f"{data_name}_pred_result/PEHE", pehe, train_step)
                                
                                iexp_p_prpsy[data_name][0] = p_prpsy.cpu().detach().numpy()
                            
                            # 验证集评估
                            eval_and_store_predictions(x_valid, "valid")
                            
                            # 测试集评估
                            eval_and_store_predictions(x_test, "test", tau_test_original if cfg.use_y_scaler else tau_test)
                            
                            # 训练集评估（如果需要）
                            if cfg.verbose > 0:
                                eval_and_store_predictions(x_train, "train", tau_train_original if cfg.use_y_scaler else tau_train)

            # 保存每个实验的结果
            if cfg.verbose > 0:
                group_list = ["train", "test", "valid"]
            else:
                group_list = ["test", "valid"]

            for group in group_list:
                result_dict[group]["p_prpsy"].append(iexp_p_prpsy[group])
                result_dict[group]["p_yf"].append(iexp_p_yf[group])
                result_dict[group]["p_ycf"].append(iexp_p_ycf[group])
                result_dict[group]["p_tau"].append(iexp_p_tau[group])
                result_dict[group]["loss"].append(iexp_losses[group])

                
                # print("p_prpsy", result_dict[group]["p_prpsy"][:10])
                # print("p_yf", result_dict[group]["p_yf"][:10])
                # print("p_ycf", result_dict[group]["p_ycf"][:10])
                # print("p_tau", result_dict[group]["p_tau"][:10])
                # print("loss", result_dict[group]["loss"][:10])
            
            for group in group_list:
                '''units, exp_i, outputs'''
                # for result_type in result_dict[group]:
                #     print(f"result_dict[{group}][{result_type}]: {type(result_dict[group][result_type])}")
                print("result_dict[group]['p_prpsy']:",group, np.array(result_dict[group]["p_prpsy"]).shape)
                # all_p_prpsy = np.array(np.swapaxes(np.swapaxes(result_dict[group]["p_prpsy"], 0, 2), 1, 2))
                all_p_yf = np.array(np.swapaxes(np.swapaxes(result_dict[group]["p_yf"], 0, 2), 1, 2))
                all_p_ycf = np.array(np.swapaxes(np.swapaxes(result_dict[group]["p_ycf"], 0, 2), 1, 2))
                all_p_tau = np.array(np.swapaxes(np.swapaxes(result_dict[group]["p_tau"], 0, 2), 1, 2))

                all_losses = [
                    [[loss.cpu().detach().numpy() if isinstance(loss, torch.Tensor) else loss for loss in sublist] for sublist in
                        exp] for exp in result_dict[group]["loss"]]
                all_losses = np.swapaxes(np.swapaxes(all_losses, 0, 1), 1, 2)

                logging.info("saving predict result as a file...")
                npz_file_path = "{}/{}_{}_result.test".format(cfg.pred_output_dir, cfg.model_name, group)
                np.savez(npz_file_path, p_yf=all_p_yf,
                            p_ycf=all_p_ycf, p_tau=all_p_tau, loss=all_losses)
                logging.info("saving predict result as a file: {}...done".format(npz_file_path))

            continue

        if 'escfr' in cfg.model_name:
            train_dataset = Escfr_ESXDataset(x_train, next_x_train, yf_train, t_train, ycf_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
            eval_dataset = Escfr_ESXDataset(x_valid, next_x_valid, yf_valid, t_valid, ycf_valid)
            eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.batch_size, shuffle=True)
            test_dataset = Escfr_ESXDataset(x_test, next_x_test, yf_test, t_test, ycf_test)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)

            input_dim = dim
            escfr = EscfrEstimator(board=writer, hparams=cfg.__dict__, train_loader=train_loader, traineval_loader=eval_loader, eval_loader=eval_loader, test_loader=test_loader, input_dim=input_dim)
            escfr.fit()
            for group in group_list:
                if group == "test":
                    pred_0, pred_1, _, _, _ = escfr.predict(test_loader)
                elif group == "train":
                    pred_0, pred_1, _, _, _ = escfr.predict(train_loader)
                elif group == "valid":
                    pred_0, pred_1, _, _, _ = escfr.predict(eval_loader)
                norm_pred_0 = y_scaler.inverse_transform(pred_0.reshape(-1, 1)) if hasattr(cfg, 'use_y_scaler') and cfg.use_y_scaler else pred_0.reshape(-1, 1)
                norm_pred_1 = y_scaler.inverse_transform(pred_1.reshape(-1, 1)) if hasattr(cfg, 'use_y_scaler') and cfg.use_y_scaler else pred_1.reshape(-1, 1)
                p_tau = (norm_pred_1 - norm_pred_0).reshape(-1, 1)

                result_dict[group]["p_tau"].append([p_tau])
                tau_test_o = (y_scaler.inverse_transform(tau_groups[group].reshape(-1, 1)) - y_scaler.data_min_) if hasattr(cfg, 'use_y_scaler') and cfg.use_y_scaler else tau_groups[group].reshape(-1, 1)
                logging.info(f'ate_pred: {np.mean(p_tau)}, ate_true: {np.mean(tau_test_o)}')

                all_p_tau = np.array(np.swapaxes(np.swapaxes(result_dict[group]["p_tau"], 0, 2), 1, 2))

                logging.info("saving predict result as a file...")
                npz_file_path = "{}/{}_{}_result.test".format(pred_output_dir, cfg.model_name, group)
                np.savez(npz_file_path, p_tau=all_p_tau)
                logging.info("saving predict result as a file: {}...done".format(npz_file_path))

            continue

        if 'pairnet' in cfg.model_name:
            tarnet = TARNet()
            print('start training TARNet')
            if hasattr(cfg, "input_S") and cfg.input_S:
                tar_x_train = np.concatenate((x_train, next_x_train), axis=1)
                tar_x_test = np.concatenate((x_test, next_x_test), axis=1)
            else:
                tar_x_train = x_train
                tar_x_test = x_test
            tarnet.fit(X=tar_x_train, y=yf_train, w=t_train)
            print('finish training TARNet')
            cate_pred_in, mu0_tr, mu1_tr= tarnet.predict(
                tar_x_train, return_po=True
            )
            cate_pred_out, mu0_te, mu1_te = tarnet.predict(
                tar_x_test, return_po=True
            )
            trn_reps = np.concatenate([tarnet.getrepr(tar_x_train), mu0_tr, mu1_tr], axis=1)
            tst_reps = np.concatenate([tarnet.getrepr(tar_x_test), mu0_te, mu1_te], axis=1)


            tar_path = Path(
                f"{cfg.emb_dir}/TARNet"
            )
            # Save representations
            print('start saving TARNet emb')
            if not os.path.exists(tar_path):
                logging.info(" os.mkdir({}) ...".format(tar_path))
                os.makedirs(tar_path)
            np.save(
                tar_path / "trn.npy",
                trn_reps,
            )
            np.save(
                tar_path / "tst.npy",
                tst_reps,
            )
            print('finish saving TARNet emb')

            print('start loading TARNet emb')
            tar_train = np.load(tar_path / "trn.npy")
            tar_test = np.load(tar_path / "tst.npy")
            print(f"Loaded Embeddings from {str(tar_path)}")

            tar_train_emb = tar_train[:, :-2]
            tar_test_emb = tar_test[:, :-2]

            pair_data_args = {
                "det": False,
                "num_cfz": 3,
                "sm_temp": 1.0,
                "dist": "euc",  # cos/euc
                "pcs_dist": True,  # Process distances
                "drop_frac": 0.1,  # distance threshold
                "arbitrary_pairs": False,
                "OT": False,
            }
            
            ads_train = PairDataset(
                X=x_train,
                beta=t_train,
                y=yf_train,
                xemb=tar_train_emb,
                **pair_data_args,
            )

            pairnet = PairNet(batch_size=cfg.batch_size, seed=cfg.seed)
            print('start training PairNet')
            pairnet.agree_fit(ads_train)
            print('finish training PairNet')
            for group in group_list:
                p_tau = pairnet.predict(X=x_groups[group])
                p_tau = p_tau.reshape(-1, 1)
                p_tau = (y_scaler.inverse_transform(p_tau) - y_scaler.data_min_) if hasattr(cfg, 'use_y_scaler') and cfg.use_y_scaler else p_tau
                result_dict[group]["p_tau"].append([p_tau])
                tau_test_o = (y_scaler.inverse_transform(tau_groups[group].reshape(-1, 1)) - y_scaler.data_min_) if hasattr(cfg, 'use_y_scaler') and cfg.use_y_scaler else tau_groups[group].reshape(-1, 1)
                print('pehe:', np.sqrt(np.mean(np.square(p_tau - tau_test_o))))
                logging.info(f'ate_pred: {np.mean(p_tau)}, ate_true: {np.mean(tau_test_o)}')

                all_p_tau = np.array(np.swapaxes(np.swapaxes(result_dict[group]["p_tau"], 0, 2), 1, 2))

                logging.info("saving predict result as a file...")
                npz_file_path = "{}/{}_{}_result.test".format(pred_output_dir, cfg.model_name, group)
                np.savez(npz_file_path, p_tau=all_p_tau)
                logging.info("saving predict result as a file: {}...done".format(npz_file_path))

            continue

        if 'perfectmatch' in cfg.model_name:
            if hasattr(cfg, "input_S") and cfg.input_S:
                prpsynetwork = PrpsyNetworkTrainer(dim+next_x_dim, cfg=cfg, device=cfg.device)
                print('start training psnet')
                x_train_tensor = torch.from_numpy(x_train).float().to(device)
                next_x_train_tensor = torch.from_numpy(next_x_train).float().to(device)
                prpsynetwork.fit(np.concatenate([x_train, next_x_train], axis=1), t_train)
                print('finish training psnet')
                t_pred = prpsynetwork.predict(np.concatenate([x_train, next_x_train], axis=1))
            else:
                prpsynetwork = PrpsyNetworkTrainer(dim, cfg=cfg, device=cfg.device)
                print('start training psnet')
                prpsynetwork.fit(x_train, t_train)
                print('finish training psnet')
                t_pred = prpsynetwork.predict(x_train)
            print(t_pred)

            def create_matched_dataset(x_train, t_train, t_pred, yf_train):
                """
                创建匹配后的新数据集
                Args:
                    x_train: 原始特征数据 (numpy array)
                    t_train: 原始处理分配 (numpy array, 0/1)
                    t_pred: 预测的倾向得分 (numpy array)
                    yf_train: 实际观察结果 (numpy array)
                Returns:
                    x_new, t_new, yf_new: 匹配后的新数据集
                """
                # 将数据转换为numpy数组（如果输入是torch.Tensor）
                t_train = np.asarray(t_train)
                t_pred = np.asarray(t_pred)
                
                # 分离处理组和对照组
                mask_t0 = (t_train == 0)
                mask_t1 = (t_train == 1)
                
                # 获取各组的索引和倾向得分
                indices_t0 = np.where(mask_t0)[0]
                indices_t1 = np.where(mask_t1)[0]
                tp_t0 = t_pred[indices_t0]
                tp_t1 = t_pred[indices_t1]
                
                # 准备存储新数据集
                x_new = []
                t_new = []
                yf_new = []
                
                # 遍历所有样本
                for i in range(len(x_train)):
                    current_t = t_train[i]
                    current_tp = t_pred[i]
                    
                    # 根据当前样本的t值选择搜索范围
                    if current_t == 0:
                        search_tp = tp_t1
                        search_indices = indices_t1
                    else:
                        search_tp = tp_t0
                        search_indices = indices_t0
                    
                    # 找到最接近的倾向得分索引
                    distances = np.abs(search_tp - current_tp)
                    closest_idx = np.argmin(distances)
                    j = search_indices[closest_idx]
                    
                    # 添加当前样本i和匹配样本j
                    x_new.append(x_train[i])
                    t_new.append(current_t)
                    yf_new.append(yf_train[i])
                    
                    x_new.append(x_train[j])
                    t_new.append(t_train[j])
                    yf_new.append(yf_train[j])
                
                # 转换为numpy数组
                return np.array(x_new), np.array(t_new), np.array(yf_new)

            x_train, t_train, yf_train = create_matched_dataset(
                x_train=x_train.numpy() if isinstance(x_train, torch.Tensor) else x_train,
                t_train=t_train.numpy() if isinstance(t_train, torch.Tensor) else t_train,
                t_pred=t_pred,
                yf_train=yf_train.numpy() if isinstance(yf_train, torch.Tensor) else yf_train
            )

            tarnet = TARNet()
            print('start training TARNet')
            tarnet.fit(X=x_train, y=yf_train, w=t_train)
            print('finish training TARNet')
            
            for group in group_list:
                # cate_pred_in, mu0_tr, mu1_tr= tarnet.predict(
                #     x_train, return_po=True
                # )
                cate_pred_out, mu1_te, mu1_te = tarnet.predict(
                    x_groups[group], return_po=True
                )

                p_tau = mu1_te - mu1_te
                p_tau = p_tau.reshape(-1, 1)
                p_tau = (y_scaler.inverse_transform(p_tau) - y_scaler.data_min_) if hasattr(cfg, 'use_y_scaler') and cfg.use_y_scaler else p_tau
                result_dict[group]["p_tau"].append([p_tau])
                tau_test_o = (y_scaler.inverse_transform(tau_groups[group].reshape(-1, 1)) - y_scaler.data_min_) if hasattr(cfg, 'use_y_scaler') and cfg.use_y_scaler else tau_groups[group].reshape(-1, 1)
                print('pehe:', np.sqrt(np.mean(np.square(p_tau - tau_test_o))))
                logging.info(f'ate_pred: {np.mean(p_tau)}, ate_true: {np.mean(tau_test_o)}')

                all_p_tau = np.array(np.swapaxes(np.swapaxes(result_dict[group]["p_tau"], 0, 2), 1, 2))

                logging.info("saving predict result as a file...")
                npz_file_path = "{}/{}_{}_result.test".format(pred_output_dir, cfg.model_name, group)
                np.savez(npz_file_path, p_tau=all_p_tau)
                logging.info("saving predict result as a file: {}...done".format(npz_file_path))

            continue

        if use_transformer:
            if 'real_world' in cfg.data_path:
                next_x_train = convert_next_x_transformer_realworld(next_x_train, data_dict)
                next_x_valid = convert_next_x_transformer_realworld(next_x_valid, data_dict)
                next_x_test = convert_next_x_transformer_realworld(next_x_test, data_dict)
            else:
                next_x_train = convert_next_x_transformer(next_x_train, data_dict)
                next_x_valid = convert_next_x_transformer(next_x_valid, data_dict)
                next_x_test = convert_next_x_transformer(next_x_test, data_dict)

        ''' create graph '''
        if 'drcfr' in cfg.model_name:
            model = DRCFR(input_dim=data_dict['dim'], 
                        share_dim=cfg.share_dim,
                        base_dim=cfg.base_dim,
                        next_dim=data_dict['next_x'].shape[1],
                        cfg=cfg,
                        device=device)
        else:
            shareNetwork = ShareNetwork(input_dim=dim, share_dim=share_dim, base_dim=base_dim, cfg=cfg, device=device)
            notshareNetwork = ShareNetwork(input_dim=dim, share_dim=share_dim, base_dim=base_dim, cfg=cfg, device=device)
            if prpsy_indep:
                prpsy_shareNetwork = ShareNetwork(input_dim=dim, share_dim=share_dim, base_dim=base_dim, cfg=cfg, device=device)
            else:
                prpsy_shareNetwork = None
            prpsy_network = PrpsyNetwork(base_dim, cfg=cfg)
            mu1_network = Mu1Network(base_dim, cfg=cfg)
            mu0_network = Mu0Network(base_dim, cfg=cfg)
            tau_network = TauNetwork(base_dim, cfg=cfg)

            # teacher network
            if use_transformer and data_dict['num_timesteps'] > 0:
                m_v = data_dict['m_v']
                m_m = data_dict['m_m']
                m_a = data_dict['m_a']
                next_shareNetwork = TransformerNetwork(input_dim=m_v + m_m + m_a, num_heads=cfg.num_heads,
                                                    num_layers=cfg.num_layers, hidden_dim=cfg.hidden_dim,
                                                    output_dim=base_dim, dropout=cfg.do_rate, device=device)
            elif use_transformer and data_dict['num_timesteps'] == 0:
                # 当num_timesteps=0时，使用一个简单的线性层
                m_v = data_dict['m_v']
                m_m = data_dict['m_m']
                m_a = data_dict['m_a']
                next_shareNetwork = ShareNetwork(input_dim=m_v + m_m + m_a, share_dim=share_dim, base_dim=base_dim, cfg=cfg,
                                                device=device)
            else:
                next_shareNetwork = ShareNetwork(input_dim=next_x_dim, share_dim=share_dim, base_dim=base_dim, cfg=cfg,
                                                device=device)

            hs_mu1_network = Mu1Network(base_dim, cfg=cfg)  # 只包含next_x信息
            hs_mu0_network = Mu0Network(base_dim, cfg=cfg)  # 只包含next_x信息

            all_mu1_network = Mu1Network(2 * base_dim, cfg=cfg)  # 包含x和next_x信息
            all_mu0_network = Mu0Network(2 * base_dim, cfg=cfg)  # 包含x和next_x信息
            # all_mu1_network = Mu1Network(base_dim, cfg=cfg)
            # all_mu0_network = Mu0Network(base_dim, cfg=cfg)
            all_prpsy_network = PrpsyNetwork(2 * base_dim, cfg=cfg)  # 包含x和next_x信息
            freeze_all_prpsy_network = PrpsyNetwork(2 * base_dim, cfg=cfg)

            if 'drcfr' in cfg.teacher_model_name:
                drcfr_model = DRCFR(input_dim = data_dict['dim'], 
                                    share_dim=cfg.share_dim,
                                    base_dim=cfg.base_dim,
                                    cfg=cfg,
                                    device=device,
                                    next_dim=cfg.base_dim)
                model = PIPCFR_on_DRCFR(prpsy_network, mu1_network, mu0_network, shareNetwork, drcfr_model,
                                        next_shareNetwork, all_prpsy_network, hs_mu1_network, hs_mu0_network, 
                                        device, freeze_all_prpsy_network=freeze_all_prpsy_network, prpsy_shareNetwork=prpsy_shareNetwork, y_scaler=y_scaler)
            else:
                model = PIPCFR(prpsy_network, mu1_network, mu0_network, tau_network, shareNetwork, notshareNetwork,
                                    next_shareNetwork, hs_mu1_network, hs_mu0_network, all_mu1_network, all_mu0_network,
                                    all_prpsy_network,
                                    cfg, device, tarreg=cfg.tarreg,
                                    freeze_all_prpsy_network=freeze_all_prpsy_network, share_rep=cfg.share_rep, prpsy_shareNetwork=prpsy_shareNetwork, y_scaler=y_scaler)
            model = model.to(device)

        ''' create optimizer '''
        if cfg.optim == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2)
        else:
            optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2)

        min_metrics = float('inf')

        ''' Build dataloader '''
        dataset = ESXDataset(x_train, next_x_train, yf_train, t_train)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # 打印训练数据集的大小
        logger.info(f"number of training sumaples: {len(dataset)}")
        logger.info(f"batch size: {batch_size}")
        logger.info(f"number of batches: {len(train_loader)}")

        """ start fitting """
        model.train()
        if (cfg.verbose):
            logging.info("exp_{} start trainning ...".format(i_exp))

        train_step = 0
        patience = 0
        for epoch in range(epochs):
            if ((epoch + 1) % log_step == 0):
                logging.info("exp_i:{},  epoch:{} ...".format(i_exp, epoch))

            for i, (inputs, next_inputs, t_labels, y_labels) in enumerate(train_loader):
                t1 = time.time()
                model.train()
                inputs = inputs.to(device)
                next_inputs = next_inputs.to(device)
                t_labels = torch.unsqueeze(t_labels.to(device), 1)
                y_labels = torch.unsqueeze(y_labels.to(device), 1)

                try:
                    total_loss = cal_total_loss(inputs, next_inputs, t_labels, y_labels, model, cfg)
                    t1 = time.time()

                    # Backpropagation
                    total_loss.backward()
                    # total_loss.backward(retain_graph=True)
                    # Update parameters
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()

                    if 'pipcfr' in cfg.model_name:
                        model.freeze_all_prpsy_network.load_state_dict(model.all_prpsy_network.state_dict())

                except Exception as e:
                    logging.info("error message:{}".format(e))
                    logging.info('traceback.format_exc():\n%s' % traceback.format_exc())
                    logging.error("there something wrong when calculating loss.")

                    logger.info(f"update gradient: {time.time()-t1}")
                    t1 = time.time()

                if (train_step + 1) % eval_step == 0:
                    valid_dict_result = evalWithData("valid_pred_result", model, writer, cfg, x_valid,
                                                        next_x_valid, yf_valid, t_valid, tau_valid_original if cfg.use_y_scaler else tau_valid)
                    logger.info(f"valid time for valid: {time.time() - t1}")
                    t1 = time.time()

                    if cfg.valid_metrics == 'valid_loss':
                        valid_metrics_val = valid_dict_result['loss'][-1]
                    elif cfg.valid_metrics == 'total_loss':
                        valid_metrics_val = valid_dict_result['loss'][0]
                    elif cfg.valid_metrics == 'pehe':
                        valid_metrics_val = valid_dict_result['pehe']
                    elif cfg.valid_metrics == 'key_loss':
                        valid_metrics_val = valid_dict_result['key_loss']
                    else:
                        valid_metrics_val = valid_dict_result['loss'][-1]

                    key_loss = valid_dict_result['key_loss']

                    test_dict_result = evalWithData("test_pred_result", model, writer, cfg, x_test,
                                                    next_x_test, yf_test, t_test, tau_test_original if cfg.use_y_scaler else tau_test)
                    if cfg.verbose > 0:
                        train_dict_result = evalWithData("train_pred_result", model, writer, cfg,
                                                            x_train, next_x_train, yf_train, t_train, tau_train_original if cfg.use_y_scaler else tau_train)

                    # 使用PEHE选择最佳模型
                    if valid_metrics_val < min_metrics:
                        min_metrics = valid_metrics_val
                        logging.info(
                            f'change output value ... i_exp:{i_exp},epochs:{epoch}, train_step:{train_step}, min_metrics:{min_metrics}, key_loss:{key_loss}')

                        # 立即转换为numpy并释放GPU内存
                        iexp_p_prpsy["valid"][0] = valid_dict_result["p_prpsy"]
                        iexp_p_yf["valid"][0] = valid_dict_result["p_yf"]
                        iexp_p_ycf["valid"][0] = valid_dict_result["p_ycf"]
                        iexp_p_tau["valid"][0] = valid_dict_result["p_tau"]
                        iexp_losses["valid"][0] = valid_dict_result["loss"]
                        iexp_key_loss["valid"][0] = valid_dict_result["key_loss"]

                        # 清理valid_dict_result
                        del valid_dict_result
                        torch.cuda.empty_cache()

                        iexp_p_prpsy["test"][0] = test_dict_result["p_prpsy"]
                        iexp_p_yf["test"][0] = test_dict_result["p_yf"]
                        iexp_p_ycf["test"][0] = test_dict_result["p_ycf"]
                        iexp_p_tau["test"][0] = test_dict_result["p_tau"]
                        iexp_losses["test"][0] = test_dict_result["loss"]
                        iexp_key_loss["test"][0] = test_dict_result["key_loss"]

                        if cfg.verbose > 0:
                            iexp_p_prpsy["train"][0] = train_dict_result["p_prpsy"]
                            iexp_p_yf["train"][0] = train_dict_result["p_yf"]
                            iexp_p_ycf["train"][0] = train_dict_result["p_ycf"]
                            iexp_p_tau["train"][0] = train_dict_result["p_tau"]
                            iexp_losses["train"][0] = train_dict_result["loss"]
                            iexp_key_loss["train"][0] = train_dict_result["key_loss"]

                        logger.info(f"valid time for test or training: {time.time() - t1}")
                        t1 = time.time()
                    if valid_metrics_val > min_metrics + early_stop_mindelta:
                        patience += 1
                        if patience >= early_stop_patience:
                            logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                            break
                    else:
                        patience = 0

                train_step += 1
                logger.info(f'update step:{train_step}')
            
            if patience >= early_stop_patience:
                break

        ''' save preidctions '''
        for group in group_list:
            result_dict[group]["p_prpsy"].append(iexp_p_prpsy[group])
            result_dict[group]["p_yf"].append(iexp_p_yf[group])
            result_dict[group]["p_ycf"].append(iexp_p_ycf[group])
            result_dict[group]["p_tau"].append(iexp_p_tau[group])
            result_dict[group]["loss"].append(iexp_losses[group])
            result_dict[group]["key_loss"].append(iexp_key_loss[group])
        ''' Format the prediction results and loss of ["train", "valid", "test"] data set and save them locally'''
        for group in group_list:
            '''units, exp_i, outputs'''
            # for result_type in result_dict[group]:
            #     print(f"result_dict[{group}][{result_type}]: {type(result_dict[group][result_type])}")
            all_p_prpsy = np.array(np.swapaxes(np.swapaxes(result_dict[group]["p_prpsy"], 0, 2), 1, 2))
            all_p_yf = np.array(np.swapaxes(np.swapaxes(result_dict[group]["p_yf"], 0, 2), 1, 2))
            all_p_ycf = np.array(np.swapaxes(np.swapaxes(result_dict[group]["p_ycf"], 0, 2), 1, 2))
            all_p_tau = np.array(np.swapaxes(np.swapaxes(result_dict[group]["p_tau"], 0, 2), 1, 2))
            all_key_loss = np.array(result_dict[group]["key_loss"])


            all_losses = [
                [[loss.cpu().detach().numpy() if isinstance(loss, torch.Tensor) else loss for loss in sublist] for
                 sublist in
                 exp] for exp in result_dict[group]["loss"]]
            all_losses = np.swapaxes(np.swapaxes(all_losses, 0, 1), 1, 2)

            logging.info("saving predict result as a file...")
            npz_file_path = "{}/{}_{}_result.test".format(pred_output_dir, cfg.model_name, group)
            np.savez(npz_file_path, p_prpsy=all_p_prpsy, p_yf=all_p_yf,
                     p_ycf=all_p_ycf, p_tau=all_p_tau, loss=all_losses, key_loss=all_key_loss)
            logging.info("saving predict result as a file: {}...done".format(npz_file_path))

        # 清理显存
        if writer is not None:
            writer.close()
        del model
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

def effect_estimate(model_args):
    try:
        seed_torch(2)
    except:
        pass

    cfg = get_config_dict(model_args)
    if not cfg:
        logging.info("No config files found:", model_args["config"])
        exit(1)
    
    logging.info("log testing ...")
    logging.info("cfg:{}".format(cfg))
    logging.debug("cfg:{}".format(cfg))
    
    logging.info("training dataset loading ...")
    data_train_path = cfg.data_path
    next_util_rate = cfg.next_util_rate
    feature_util_rate = cfg.feature_util_rate
    data_dict = load_data(data_train_path, feature_util_rate, next_util_rate)
    logging.info("training dataset loading ...done.")
    
    device = torch.device(cfg.device)
    logging.info(f"Use {cfg.device}")

    train(data_dict, device, cfg)
