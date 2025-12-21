import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import shutil

seed=42
np.random.seed(seed)
T = 60
noise_scale = 1.0
data_size = 5000
normalize = True  # 开启归一化
x_t0 = False
surrogate_dim = 10

if not x_t0:
    T += 1

# if T==60 and noise_scale==3.0 and data_size==400:
#     seed=41

base_path = 'figures'
if os.path.exists(base_path):
    shutil.rmtree(base_path)
os.mkdir(base_path)

cluster_coeffs = (np.array([-0.6, -0.15, 0, 0.15, 0.3, 0.45, 0.6]) + 0.4) * 2
x_offset_scale = 3.0 
kmeans_cf = True
x_offset_scale = 20.0
kmeans_offset = False

def save_data(data, file_path):
    # if not os.path.exists(file_path):
    #     os.makedirs(file_path)
    np.savez(file_path, **data)

def generate_stable_matrix(m, radius_thred=0.9):
    A = np.random.rand(m, m) * 2 - 1
    eigenvalues = np.linalg.eigvals(A)
    spectral_radius = max(abs(eigenvalues))
    if spectral_radius >= radius_thred:
        A = A * (radius_thred / spectral_radius)
    return A

def generate_dataset():
    num_samples = 5000
    num_features = 3477
    k = 150

    X_all = []
    next_X_all = []
    next_X_cf_all = []
    Y_ts_all = []
    T_all = []
    Y_all = []
    Y_cf_all = []
    tau_all = []

    mix_matrix = generate_stable_matrix(surrogate_dim)
    scale_matrix = np.mat(np.random.rand(surrogate_dim)).reshape(1, surrogate_dim)


    for i in tqdm(range(1,2)):
        TY = np.loadtxt('csv/topic_doc_mean_n5000_k3477_seed_' + str(i) + '.csv.y', delimiter=',')
        # treatment = TY[:,0]
        # treated = np.where(treatment > 0)[0]
        # controlled = np.where(treatment < 1)[0]
        y = TY[:, 1]
        yc = TY[:, 2]

        matrix = np.zeros((num_samples, num_features))

        with open('csv/topic_doc_mean_n5000_k3477_seed_' + str(i) + '.csv.x', 'r') as fin:
            for line in fin.readlines():
                line = line.strip().split(',')
                matrix[int(line[0]) - 1, int(line[1]) - 1] = int(line[2])

        # run LDA
        no_topics = 50
        #z = np.random.randn(num_samples, 50) * 0.001
        z = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online',
                                      learning_offset=50., random_state=0).fit_transform(matrix)
        z1 = z[np.random.randint(num_samples, size=1), :]
        z0 = np.mean(z, axis=0)

        prps = np.exp(k * np.matmul(z, z1.reshape(-1))) / (np.exp(k * np.matmul(z, z0.reshape(-1))) + np.exp(k * np.matmul(z, z1.reshape(-1))))

        print(((0.3< prps)  * (prps < 0.45)).sum())

        sns.kdeplot(prps.flatten())
        plt.title(f'{np.mean(prps)}')
        plt.savefig(f'{base_path}/prps_{i}.png')
        plt.close()

        treatment = np.random.binomial(1, prps, size=num_samples)

        Y = np.zeros((5000, T, surrogate_dim))
        YC = np.zeros((5000, T, surrogate_dim))

        for i in range(surrogate_dim):
            Y[:,0, i]=y
            YC[:,0, i]=yc
        for t in range(1, T):
            noise = np.random.laplace(0, noise_scale, surrogate_dim)
            offset_item_f = np.dot(
                (np.matmul(z, z0.reshape(-1)) + treatment * np.matmul(z, z1.reshape(-1))).reshape(-1, 1),
                scale_matrix)
            Y[:, t] = offset_item_f * x_offset_scale + np.dot(Y[:, t - 1], mix_matrix) + noise
            offset_item_cf = np.dot((
                        np.matmul(z, z0.reshape(-1)) + (1 - treatment) * np.matmul(z, z1.reshape(-1))).reshape(-1, 1), scale_matrix)
            YC[:, t] = offset_item_cf * x_offset_scale + np.dot(YC[:, t - 1], mix_matrix) + noise

        treatment=np.reshape(treatment,(num_samples,1))
        # data=np.concatenate((treatment,Y),axis=1)
        #
        # y_treated = np.concatenate((YC[controlled], Y[treated]), axis=0)
        # y_controlled = np.concatenate((Y[controlled], YC[treated]), axis=0)
        # #causal_effects = np.mean(y_treated - y_controlled, axis=0)
        # causal_effects = np.array(y_treated - y_controlled)[:, -1]
        # print("treatment.shape: ", treatment.shape, '\n\n')
        # print("matrix.shape: ", matrix.shape, '\n\n')
        # print("Y.shape: ", Y.shape, '\n\n')
        # print("data.shape: ", data.shape, '\n\n')
        # print("y_treated.shape: ", y_treated.shape, '\n\n')
        # print("y_controlled.shape: ", y_controlled.shape, '\n\n')
        # print("causal_effects.shape: ", causal_effects.shape, '\n\n')


        # np.savetxt('data/Series_groundtruth_'+str(i)+'.txt', causal_effects, delimiter=',', fmt='%.2f')
        # np.savetxt('data/Series_y_'+str(i)+'.txt', data, delimiter=',', fmt='%.2f')

        LT_Y = np.mean(Y[:, -3:], axis=(1, 2)).reshape(num_samples, 1)
        LT_YCF = np.mean(YC[:, -3:], axis=(1, 2)).reshape(num_samples, 1)

        X_all.append(matrix)
        next_X_all.append(Y[:, :-1].reshape(num_samples, -1))
        next_X_cf_all.append(YC[:, :-1].reshape(num_samples, -1))
        Y_ts_all.append(Y)
        causal_effects = (LT_Y[:,0] - LT_YCF[:,0]) * (treatment[:, -1] - 0.5) * 2
        T_all.append(treatment)
        Y_all.append(LT_Y)
        Y_cf_all.append(LT_YCF)
        tau_all.append(causal_effects.reshape(num_samples, 1))

    X_all = np.concatenate(X_all, axis=0)
    next_X_all = np.concatenate(next_X_all, axis=0)
    next_X_cf_all = np.concatenate(next_X_cf_all, axis=0)
    Y_ts_all = np.concatenate(Y_ts_all, axis=0)
    T_all = np.concatenate(T_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)
    Y_cf_all = np.concatenate(Y_cf_all, axis=0)
    tau_all = np.concatenate(tau_all, axis=0)
    e_all = np.zeros((tau_all.shape[0], 1))

    # kmeans cluster
    if kmeans_offset:
        if kmeans_cf:
            kmeans_inputs = next_X_all * T_all + next_X_cf_all * (1 - T_all)
        else:
            kmeans_inputs = next_X_all
        kmeans = KMeans(n_clusters=len(cluster_coeffs), random_state=seed)
        clusters = kmeans.fit_predict(kmeans_inputs)
        Y_all += Y_all * cluster_coeffs[clusters].reshape(-1, 1) * T_all
        Y_cf_all += Y_cf_all * cluster_coeffs[clusters].reshape(-1, 1) * (1 - T_all)
    tau_all = (Y_all - Y_cf_all) * (T_all - 0.5) * 2

    print("X_all.shape:", X_all.shape)
    print("next_X_all.shape:", next_X_all.shape)
    print("T_all.shape:", T_all.shape)
    print("Y_all.shape:", Y_all.shape)
    print("Y_cf_all.shape:", Y_cf_all.shape)
    print("tau_all.shape:", tau_all.shape)
    print("e_all.shape:", e_all.shape)

    if normalize:
        x_scaler = StandardScaler()
        next_X_all_normalized = x_scaler.fit_transform(next_X_all)
        next_X_all_normalized = next_X_all.reshape(next_X_all.shape[0], -1)
        

    sns.kdeplot(Y_all.flatten())
    plt.title(f'{np.mean(Y_all)}')
    plt.savefig(f'figures/train_raw_Yf_dist.png')
    plt.close()

    sns.kdeplot(Y_all[T_all[:, 0].astype(bool)].flatten())
    plt.title(f'{np.mean(Y_all[T_all[:, 0].astype(bool)])}')
    plt.savefig(f'figures/train_raw_Yft_dist.png')
    plt.close()

    sns.kdeplot(Y_all[(1 - T_all[:, 0]).astype(bool)].flatten())
    plt.title(f'{np.mean(Y_all[(1 - T_all[:, 0]).astype(bool)])}')
    plt.savefig(f'figures/train_raw_Yfc_dist.png')
    plt.close()

    sns.kdeplot(Y_cf_all.flatten())
    plt.title(f'{np.mean(Y_cf_all)}')
    plt.savefig(f'figures/train_raw_Ycf_dist.png')
    plt.close()

    sns.kdeplot(Y_cf_all[T_all[:, 0].astype(bool)].flatten())
    plt.title(f'{np.mean(Y_cf_all[T_all[:, 0].astype(bool)])}')
    plt.savefig(f'figures/train_raw_Ycfc_dist.png')
    plt.close()

    sns.kdeplot(Y_cf_all[(1 - T_all[:, 0]).astype(bool)].flatten())
    plt.title(f'{np.mean(Y_cf_all[(1 - T_all[:, 0]).astype(bool)])}')
    plt.savefig(f'figures/train_raw_Ycft_dist.png')
    plt.close()

    sns.kdeplot(tau_all.flatten())
    plt.savefig(f'figures/train_raw_Tau_dist.png')
    plt.close()

    plt.plot(np.arange(T), Y_ts_all.mean(axis=(0, 2)))
    plt.savefig(f'figures/train_raw_ts.png')
    plt.close()

    return {
        'x': X_all[:, :, np.newaxis],
        'next_x': next_X_all_normalized[:, :, np.newaxis],
        'next_x_not_normalized': next_X_all[:, :, np.newaxis],
        't': T_all,
        'yf': Y_all,
        'ycf': Y_cf_all,
        'tau': tau_all,
        'e': e_all,
        'm_x': X_all.shape[1],
        'm_v': surrogate_dim,
        'm_m': 0,
        'm_a': 0,
        'num_timesteps': T - 1
    }

def create_experiments(train_data, num_exp=5, exp_ratio=0.9):
    num_samples = train_data['x'].shape[0]
    exp_size = int(num_samples * exp_ratio)
    experiments = []

    for i in range(num_exp):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        exp_indices = indices[:exp_size]

        exp_data = {key: value[exp_indices] for key, value in train_data.items()}
        experiments.append(exp_data)

    # Combine the experiments along the last dimension
    combined_experiments = {}
    for key in train_data.keys():
        combined_experiments[key] = np.stack([exp[key] for exp in experiments], axis=-1).squeeze()

    return combined_experiments

full_data = generate_dataset()
for i in full_data:
    try:
        print(full_data[i].shape)
    except:
        pass

num_exp = 1
if data_size > 0:
    split_num = data_size * 3
elif data_size == 0:
    split_num = int(len(full_data['x']))
else:
    split_num = int(0.8 * len(full_data['x']))
split_index = np.arange(len(full_data['x']))
np.random.shuffle(split_index)
# print(split_index)
train_data = {key: value[split_index[:split_num]] if not isinstance(value, int) else value for key, value in full_data.items()}
test_data = {key: value[split_index[split_num:]] if not isinstance(value, int) else value for key, value in full_data.items()}

final_train_data = train_data # create_experiments(train_data, num_exp=num_exp, exp_ratio=1.0)
#print(final_train_data)
for k, v in final_train_data.items():
    if not isinstance(v, int):
        print(k, v.shape)
    else:
        print(k, v)

for k, v in test_data.items():
    if not isinstance(v, int):
        test_data[k] = np.repeat(v, num_exp, axis=-1)
for k, v in test_data.items():
    if not isinstance(v, int):
        print(k, v.shape)
    else:
        print(k, v)

# train_data['e'] = np.zeros(len(train_data['x']))[:, np.newaxis]
# test_data['e'] = np.zeros(len(test_data['x']))[:, np.newaxis]
# print("train_data['e'].shape:", train_data['e'].shape)
# print("test_data['e'].shape:", test_data['e'].shape)

path = f'_{num_exp}'
data_path = f'data'
suffix = ''
if normalize:
    suffix += '_normalize'
if not os.path.exists(data_path):
    os.makedirs(data_path)
train_save_path = f'{data_path}/train_T{T-1}_noise{noise_scale}_size{data_size}_data{path}{suffix}.npz'
test_save_path = f'{data_path}/test_T{T-1}_noise{noise_scale}_size{data_size}_data{path}{suffix}.npz'
print('save to ', train_save_path, test_save_path)
save_data(final_train_data, train_save_path)
save_data(test_data, test_save_path)

print("Data generation and saving complete.")