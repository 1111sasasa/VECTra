import sys
sys.path.append("..")

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch

from evluation_utils import (
    get_road,
    fair_sampling,
    get_seq_emb_from_traj_withRouteOnly,
    prepare_data,
)
from task import sim_srh


torch.set_num_threads(5)

dev_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(dev_id)
torch.cuda.set_device(dev_id)


def evaluation(city, exp_path, model_name, start_time, detour_rate=0.3, fold=10):
    route_min_len, route_max_len, gps_min_len, gps_max_len = 10, 100, 10, 256
    model_path = os.path.join(exp_path, 'model', model_name)

    feature_df = pd.read_csv("/home/shzheng2025/data/{}/edge_features.csv".format(city))
    num_nodes = len(feature_df)
    print("num_nodes:", num_nodes)

    test_node_data = pickle.load(
        open('/home/shzheng2025/data/{}/{}_1101_1115_data_sample10w.pkl'.format(city, city), 'rb'))
    road_list = get_road(test_node_data)
    print('number of road obervased in test data: {}'.format(len(road_list)))

    num_samples = 'all'
    if isinstance(num_samples, int):
        test_node_data = fair_sampling(test_node_data, num_samples)

    road_list = get_road(test_node_data)
    print('number of road obervased after sampling: {}'.format(len(road_list)))

    seq_model = torch.load(model_path, map_location="cuda:{}".format(dev_id))['model']
    seq_model.eval()

    print('start time : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))))
    print("\n=== sim_srh Evaluation ===")

    test_seq_data = pickle.load(
        open('/home/shzheng2025/data/{}/{}_1101_1115_data_seq_evaluation.pkl'.format(city, city), 'rb'))
    test_seq_data = test_seq_data.sample(50000, random_state=0)

    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset = prepare_data(
        test_seq_data, route_min_len, route_max_len, gps_min_len, gps_max_len)
    test_data = (route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset)
    seq_embedding = get_seq_emb_from_traj_withRouteOnly(seq_model, test_data, batch_size=1024)

    geometry_df = pd.read_csv("/home/shzheng2025/data/{}/edge_geometry.csv".format(city))
    trans_mat = np.load('/home/shzheng2025/data/{}/transition_prob_mat.npy'.format(city))
    trans_mat = torch.tensor(trans_mat)

    sim_srh.evaluation3(
        seq_embedding,
        None,
        seq_model,
        test_seq_data,
        num_nodes,
        trans_mat,
        feature_df,
        geometry_df,
        detour_rate=detour_rate,
        fold=fold,
    )

    end_time = time.time()
    print("cost time : {:.2f} s".format(end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='chengdu')
    parser.add_argument('--exp_path', type=str, default='/home/shzheng2025/bias/exp/JTMR_chengdu_260513140154')
    parser.add_argument('--model_name', type=str, default='JTMR_chengdu_v1_hier_image_20_100000_260513140154_19.pt')
    parser.add_argument('--detour_rate', type=float, default=0.3)
    parser.add_argument('--fold', type=int, default=10)
    args = parser.parse_args()

    start_time = time.time()
    evaluation(args.city, args.exp_path, args.model_name, start_time, detour_rate=args.detour_rate, fold=args.fold)
