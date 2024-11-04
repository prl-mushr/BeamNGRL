import torch
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarNetworkDyn import SimpleCarNetworkDyn
from BeamNGRL.control.UW_mppi.Dynamics.SimpleCarDynamicsCUDA import SimpleCarDynamics
from BeamNGRL.dynamics.utils.exp_utils import get_dataloaders
import yaml
import os
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import sys
from scipy.stats import mannwhitneyu, t as student_t

def conf(data):
    # Sample size
    n = len(data)
    s = np.std(data, ddof=1)  # Use ddof=1 to get the sample standard deviation
    # Confidence level
    C = 0.95  # 95%
    # Significance level, α
    alpha = 1 - C
    # Number of tails
    tails = 2
    # Quantile (the cumulative probability)
    q = 1 - (alpha / tails)
    # Degrees of freedom
    dof = n - 1
    # Critical t-statistic, calculated using the percent-point function (aka the
    # quantile function) of the t-distribution
    t_star = student_t.ppf(q, dof)
    # Confidence interval
    return t_star * s / np.sqrt(n)


## the job of this script is to take ground-truth data for controls and states, run the controls through the dynamics model and compare the predicted states to the ground-truth states

def get_dynamics(model, Config):
    Dynamics_config = Config["Dynamics_config"]
    MPPI_config = Config["MPPI_config"]
    Map_config = Config["Map_config"]
    print("loading: ", model)
    if model == 'baseline' or model=='GT':
        Dynamics_config["network"] = Dynamics_config["network_baseline"]
        Dynamics_config["model_weights"] = Dynamics_config["model_weights_baseline_spnorm"]
        model_weights_path = str(Path(os.getcwd()).parent.absolute()) + "/logs/baseline_spnorm/" + Dynamics_config["model_weights"]
        Dynamics_config["network"]["net_kwargs"]["spectral_norm"] = True
        dynamics = SimpleCarNetworkDyn(Dynamics_config, Map_config, MPPI_config, model_weights_path=model_weights_path)
    elif model == 'slip3d' or model == "gt_rollout":
        Dynamics_config["type"] = "slip3d" ## just making sure 
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
    elif model == 'noslip3d' or model == "no_change" or model=="zeros":
        # temporarily change the dynamics type to noslip3d
        Dynamics_config["type"] = "noslip3d"
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
        Dynamics_config["type"] = "slip3d"
    elif model == 'slip3d_bad_sys':
        # temporarily change the dynamics type to noslip3d
        Dynamics_config["type"] = "slip3d"
        temp_D = Dynamics_config["D"]
        Dynamics_config["D"] = 1.2 ## 50 % of the original D
        dynamics = SimpleCarDynamics(Dynamics_config, Map_config, MPPI_config)
        Dynamics_config["D"] = temp_D ## change it back
    else:
        print("bruh moment")
        raise ValueError('Unknown model type')
    return dynamics

def evaluator(
        data_loader,
        config,
        tn_args,
        ):
        Dynamics_config = config["Dynamics_config"]
        MPPI_config = config["MPPI_config"]
        dt = Dynamics_config["dt"]
        dataset_dt = 0.02
        skip = int(dt/dataset_dt) ## please keep the dt a multiple of the dataset_dt
        TIMESTEPS = MPPI_config["TIMESTEPS"]
        np.set_printoptions(threshold=sys.maxsize)

        if 0:
            learned_model = get_dynamics('baseline', config)
            for model in ['noslip3d','slip3d_bad_sys' , 'slip3d']:
                dynamics = get_dynamics(model, config)
                predict_states_list = []
                for i, (states_tn, controls_tn, ctx_tn_dict) in enumerate(tqdm(data_loader)):
                    states_tn = states_tn.to(**tn_args)[:,::skip,:]
                    controls_tn = controls_tn.to(**tn_args)[:,::skip,:]
                    ctx_tn_dict = {k: tn.to(**tn_args) for k, tn in ctx_tn_dict.items()}

                    BEV_heght = ctx_tn_dict["bev_elev"].squeeze(0).squeeze(0)
                    BEV_normal = ctx_tn_dict["bev_normal"].squeeze(0).squeeze(0)

                    states = torch.zeros(17).to(**tn_args)
                    states[:15] = states_tn[0,0,:].clone()
                    states = states.repeat(dynamics.M, dynamics.K, dynamics.T, 1)
                    gt_states = states_tn.repeat(dynamics.K, 1, 1).clone()

                    controls = controls_tn.repeat((dynamics.K, 1, 1)).clone()
                    learned_bev_hgt = BEV_heght.repeat((dynamics.K, 1, 1, 1))
                    learned_bev_nor = BEV_normal.repeat((dynamics.K, 1, 1, 1))

                    dynamics.set_BEV(BEV_heght, BEV_normal)
                    predict_states = dynamics.forward(states, controls)
                    predict_states_list.append(predict_states.cpu().numpy())

                predict_states_list = np.array(predict_states_list)
                dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/States/" + model
                if(not os.path.isdir(dir_name)):
                    os.makedirs(dir_name)
                data_name = "/{}.npy".format(config["dataset"]["name"])
                filename = dir_name + data_name
                np.save(filename, predict_states_list)

            for model in ['noslip3d','slip3d_bad_sys' , 'slip3d', 'GT']:
                predict_states_list = np.load(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/States/" + model + "/{}.npy".format(config["dataset"]["name"]))
                predict_states_list = torch.from_numpy(predict_states_list).to(**tn_args)
                disturbed_vectors = []
                for i, (states_tn, controls_tn, ctx_tn_dict) in enumerate(tqdm(data_loader)):
                    states_tn = states_tn.to(**tn_args)[:,::skip,:]
                    controls_tn = controls_tn.to(**tn_args)[:,::skip,:]
                    ctx_tn_dict = {k: tn.to(**tn_args) for k, tn in ctx_tn_dict.items()}

                    BEV_heght = ctx_tn_dict["bev_elev"].squeeze(0).squeeze(0)
                    BEV_normal = ctx_tn_dict["bev_normal"].squeeze(0).squeeze(0)

                    states = torch.zeros(17).to(**tn_args)
                    states[:15] = states_tn[0,0,:].clone()
                    states = states.repeat(dynamics.M, dynamics.K, dynamics.T, 1)
                    gt_states = states_tn.repeat(dynamics.K, 1, 1).clone()

                    controls = controls_tn.repeat((dynamics.K, 1, 1)).clone()
                    learned_bev_hgt = BEV_heght.repeat((dynamics.K, 1, 1, 1))
                    learned_bev_nor = BEV_normal.repeat((dynamics.K, 1, 1, 1))

                    if model == 'GT':
                        disturbed_vector = gt_states # learned_model.dyn_model._forward(gt_states, controls, ctx_data={'bev_elev':learned_bev_hgt, 'bev_normal':learned_bev_nor}, dt =dt)
                    else:
                        predict_states = predict_states_list[i]
                        disturbed_vector = learned_model.dyn_model._forward(predict_states.squeeze(0), controls.squeeze(0), ctx_data={'bev_elev':learned_bev_hgt, 'bev_normal':learned_bev_nor}, dt =dt)
                    disturbed_vectors.append(disturbed_vector.cpu().numpy())

                disturbed_vectors = np.array(disturbed_vectors)[:,0,:,:]
                dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/DV/" + model
                if(not os.path.isdir(dir_name)):
                    os.makedirs(dir_name)
                data_name = "/{}.npy".format(config["dataset"]["name"])
                filename = dir_name + data_name
                np.save(filename, disturbed_vectors)
            

        plt.figure().set_size_inches(6, 3)
        plt.subplots_adjust(left=0.112, right=0.98, top=0.952, bottom=0.188)  # Adjust the values as needed

        dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/DV/GT"
        data_name = "/{}.npy".format(config["dataset"]["name"])
        filename = dir_name + data_name
        GT_vector = np.load(filename)

        mean = np.mean(GT_vector, axis=(0,1))
        std = np.max(np.abs(GT_vector - mean), axis=(0,1))
        GT_v = (GT_vector[..., 6:15])/std[6:15]
        norm_GT_v = np.linalg.norm(GT_v, axis=2)
        #lipschitz:
        L = np.max(np.linalg.norm(GT_v, axis=2))
        # plot lipschitz:
        plt.plot(np.arange(TIMESTEPS)*config["Dynamics_config"]["dt"], L*np.ones(TIMESTEPS), label="Lipschitz", color = 'black', linestyle='--')

        color_palette = 'plasma'
        original_colors = plt.get_cmap(color_palette)(range(256))
        skips = 256 //3 #*len(rolls))
        colors = original_colors[::skips]
        count = 0

        for model in ['noslip3d','slip3d_bad_sys', 'slip3d']:
            dir_name = str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/DV/" + model
            if(not os.path.isdir(dir_name)):
                os.makedirs(dir_name)
            data_name = "/{}.npy".format(config["dataset"]["name"])
            filename = dir_name + data_name
            disturbed_vector = np.load(filename)
            if model == 'baseline':
                model = "Learned-baseline"

            dist_v = (disturbed_vector[..., 6:15])/std[6:15]
            norm_dist_v = np.linalg.norm(dist_v, axis=2)
            # cosine similarity between dist_v and GT_v
            # cosine_similarity = np.sum(dist_v * GT_v, axis=2) / (norm_dist_v * norm_GT_v)
            # mean_cosine = np.mean(cosine_similarity, axis=0)
            # conf_cosine = conf(cosine_similarity)
            # plt.plot(np.arange(TIMESTEPS)*config["Dynamics_config"]["dt"], mean_cosine, label=model, color = colors[count])
            # plt.fill_between(np.arange(TIMESTEPS)*config["Dynamics_config"]["dt"], mean_cosine - conf_cosine, mean_cosine + conf_cosine, alpha=0.2, color=colors[count])
            mean_v = np.mean(norm_dist_v, axis=0)
            conf_v = conf(norm_dist_v)
            plt.plot(np.arange(TIMESTEPS)*config["Dynamics_config"]["dt"], mean_v, label=model, color = colors[count])
            plt.fill_between(np.arange(TIMESTEPS)*config["Dynamics_config"]["dt"], mean_v - conf_v, mean_v + conf_v, alpha=0.2, color=colors[count])
            count += 1
        plt.ylim(0.5, 5.0)
        plt.xlabel("Time(s)")
        plt.ylabel("State derivative norm")
        plt.legend()
        # plt.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/States/Cosine_similarity.png")
        plt.savefig(str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Results/States/Lipschitz.png")
        plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="Eval_long.yaml", help='config file for training model')
    parser.add_argument('--shuffle', type=bool, required=False, default=False, help='shuffle data')
    parser.add_argument('--batchsize', type=int, required=False, default=1, help='training batch size')

    args = parser.parse_args()

    # Set torch params
    torch.manual_seed(0)
    torch.set_num_threads(1)
    
    tensor_args = {'device': torch.device('cuda'), 'dtype': torch.float32}

    # Load experiment config
    config = yaml.load(open( str(Path(os.getcwd()).parent.absolute()) + "/Experiments/Configs/" + args.config).read(), Loader=yaml.SafeLoader)
    # Dataloaders
    train_loader, valid_loader, stats, data_cfg = get_dataloaders(args, config)
    with torch.no_grad():
        evaluator(valid_loader, config, tensor_args)