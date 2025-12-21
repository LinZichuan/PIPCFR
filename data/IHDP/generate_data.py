import numpy as np
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

n = 60
generate_num_exp = False
normalize = True
noise_scale = 1.0
not_fix_testset = True
true_n = n + 1
surrogate_dim = 10

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
    N=747
    values_t=[0,1,2,3,4]
    values_c=[-2,-1, 0,1,2]
    probabilities_t=[0.5,0.2,0.15,0.1,0.05]
    probabilities_c=[0.2,0.2,0.2,0.2,0.2]
    '''Data generation follow Bayesian NonParameteric modeling'''

    X_all = []
    next_X_all = []
    next_X_all_not_normalized = []
    T_all = []
    Y_all = []
    Y_all_not_normalized = []
    Y_cf_all = []
    Y_cf_all_not_normalized = []
    tau_all = []
    tau_all_not_normalized = []

    mix_matrix = generate_stable_matrix(surrogate_dim)
    scale_matrix = np.mat(np.random.rand(surrogate_dim)).reshape(1, surrogate_dim)

    for i in range(1,2):
        TY=np.loadtxt('csv/ihdp_npci_'+str(i)+'.csv',delimiter=',')
        treatment=TY[:,0]
        treated = np.where(treatment > 0)[0]
        N_treated=treated.shape[0]
        controlled = np.where(treatment < 1)[0]
        N_control=controlled.shape[0]
        y=TY[:,1]
        yc=TY[:,2]

        matrix = TY[:,5:]

        Y=np.zeros((N, true_n, surrogate_dim))
        YC=np.zeros((N, true_n, surrogate_dim))

        for i in range(surrogate_dim):
            Y[:, 0, i] = y
            YC[:, 0, i] = yc
        for t in range(1, true_n):
            noise = np.random.laplace(0, noise_scale, surrogate_dim)
            beta_t=np.random.choice(values_t, (25, surrogate_dim), p=probabilities_t)
            beta_c=np.random.choice(values_c, (25, surrogate_dim), p=probabilities_c)
            # print("debug:", matrix[controlled,].shape, np.dot(matrix[controlled,],beta).shape)
            # print(Y[controlled,t].shape, np.random.laplace(np.dot(matrix[controlled,],beta),1).shape, np.mean(Y[controlled,0:t-1],axis=1).shape)
            # exit(0)
            # Y[controlled,t]=np.random.laplace(np.dot(matrix[controlled,],beta_c),1)+0.02*np.mean(Y[controlled,0:t],axis=1)
            # Y[treated, t] = np.random.laplace(np.dot(matrix[treated,], beta_c)+4, 1) + 0.02*np.mean(Y[treated, 0:t], axis=1)
            # YC[controlled,t]=np.random.laplace(np.dot(matrix[controlled,],beta_c)+4,1)+0.02*np.mean(YC[controlled,0:t],axis=1)
            # YC[treated, t] = np.random.laplace(np.dot(matrix[treated,], beta_c), 1) + 0.02*np.mean(YC[treated, 0:t], axis=1)

            Y[controlled,t]=0.02*np.random.laplace(np.dot(matrix[controlled,],beta_c),1)+Y[controlled,t-1]
            Y[treated, t] = 0.02*np.random.laplace(np.dot(matrix[treated,], beta_t), 1) + Y[treated, t-1]
            YC[controlled,t]=0.02*np.random.laplace(np.dot(matrix[controlled,],beta_t),1)+YC[controlled,t-1]
            YC[treated, t] = 0.02*np.random.laplace(np.dot(matrix[treated,], beta_c), 1) + YC[treated, t-1]

        # 先保存未归一化的版本
        LT_Y = np.mean(Y[:, -3:], axis=(1,2)).reshape(N, 1)
        LT_YCF = np.mean(YC[:, -3:], axis=(1,2)).reshape(N, 1)

        next_X_all_not_normalized.append(Y[:, :-3].reshape(N, -1))
        Y_all_not_normalized.append(LT_Y)
        Y_cf_all_not_normalized.append(LT_YCF)
        
        # 计算未归一化的tau
        print(LT_Y.shape, LT_YCF.shape)
        tau_not_normalized = np.where(treatment.flatten() == 1, 
                                    (LT_Y - LT_YCF).flatten(), 
                                    (LT_YCF - LT_Y).flatten())
        tau_all_not_normalized.append(tau_not_normalized.reshape(N, 1))
        
        if normalize:
            # 将所有outcome（所有时间步的Y和YC）一起归一化
            y_scaler = MinMaxScaler()
            # y_scaler = StandardScaler()
            all_outcomes = np.concatenate([
                Y.reshape(-1, 1),      # 所有时间步的factual outcomes
                YC.reshape(-1, 1)      # 所有时间步的counterfactual outcomes
            ], axis=0)
            all_outcomes_normalized = y_scaler.fit_transform(all_outcomes)
            
            # 分别获取归一化后的值并重塑
            Y = all_outcomes_normalized[:Y.size].reshape(Y.shape)
            YC = all_outcomes_normalized[Y.size:].reshape(YC.shape)

        treatment=np.reshape(treatment,(N,1))
        # data=np.concatenate((treatment,Y),axis=1)

        # 计算归一化后的tau
        LT_Y = np.mean(Y[:, -3:], axis=(1,2)).reshape(N, 1)
        LT_YCF = np.mean(YC[:, -3:], axis=(1,2)).reshape(N, 1)

        tau = np.where(treatment.flatten() == 1, 
                    (LT_Y - LT_YCF).flatten(), 
                    (LT_YCF - LT_Y).flatten())

        X_all.append(matrix)
        next_X_all.append(Y[:, :-3].reshape(N, -1))
        T_all.append(treatment)
        Y_all.append(LT_Y)
        Y_cf_all.append(LT_YCF)
        tau_all.append(tau.reshape(N, 1))

    X_all = np.vstack(X_all)
    next_X_all = np.vstack(next_X_all)
    next_X_all_not_normalized = np.vstack(next_X_all_not_normalized)
    T_all = np.vstack(T_all)
    Y_all = np.vstack(Y_all)
    Y_all_not_normalized = np.vstack(Y_all_not_normalized)
    Y_cf_all = np.vstack(Y_cf_all)
    Y_cf_all_not_normalized = np.vstack(Y_cf_all_not_normalized)
    tau_all = np.vstack(tau_all)
    tau_all_not_normalized = np.vstack(tau_all_not_normalized)
    e_all = np.zeros((tau_all.shape[0], 1))

    # 生成随机排列索引
    np.random.seed(42)
    shuffle_indices = np.random.permutation(X_all.shape[0])
    
    # 使用相同的随机索引打乱所有数据
    X_all = X_all[shuffle_indices]
    next_X_all = next_X_all[shuffle_indices]
    next_X_all_not_normalized = next_X_all_not_normalized[shuffle_indices]
    T_all = T_all[shuffle_indices]
    Y_all = Y_all[shuffle_indices]
    Y_all_not_normalized = Y_all_not_normalized[shuffle_indices]
    Y_cf_all = Y_cf_all[shuffle_indices]
    Y_cf_all_not_normalized = Y_cf_all_not_normalized[shuffle_indices]
    tau_all = tau_all[shuffle_indices]
    tau_all_not_normalized = tau_all_not_normalized[shuffle_indices]
    e_all = e_all[shuffle_indices]

    print("tau_all:", tau_all.mean(), tau_all.std())
    print("tau_all_not_normalized:", tau_all_not_normalized.mean(), tau_all_not_normalized.std())
    print("Y_all:", Y_all.mean(), Y_all.std())
    print("Y_all[t]:", Y_all[T_all.flatten().astype(np.int32)].mean(), Y_all[T_all.flatten().astype(np.int32)].std())
    print("Y_all[c]:", Y_all[1-T_all.flatten().astype(np.int32)].mean(), Y_all[1-T_all.flatten().astype(np.int32)].std())
    print("Y_all_not_normalized:", Y_all_not_normalized.mean(), Y_all_not_normalized.std())
    print("X_all.shape:", X_all.shape)
    print("next_X_all.shape:", next_X_all.shape)
    print("T_all.shape:", T_all.shape)
    print("Y_all.shape:", Y_all.shape)
    print("Y_cf_all.shape:", Y_cf_all.shape)
    print("tau_all.shape:", tau_all.shape)
    print("e_all.shape:", e_all.shape)

    return {
        'x': X_all[:, :, np.newaxis],
        'next_x_not_normalized': next_X_all_not_normalized[:, :, np.newaxis],
        'next_x': next_X_all[:, :, np.newaxis],
        't': T_all,
        'yf': Y_all_not_normalized,
        'ycf': Y_cf_all_not_normalized,
        'tau': tau_all_not_normalized,
        'e': e_all,
        'yf_original': Y_all_not_normalized,
        'ycf_original': Y_cf_all_not_normalized,
        'tau_original': tau_all_not_normalized,
        'y_scaler': y_scaler if normalize else None
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
    if i == 'y_scaler':
        continue
    print(full_data[i].shape)

split_index = int(0.8 * len(full_data['x']))
train_data = {key: value[:split_index] for key, value in full_data.items() if key != 'y_scaler'}
test_data = {key: value[split_index:] for key, value in full_data.items() if key != 'y_scaler'}
metadata = {'m_x': 1, 'm_v': 1, 'm_m': 0, 'm_a': 0, 'num_timesteps': n, 'y_scaler': full_data['y_scaler']}
train_data.update(metadata)
test_data.update(metadata)
full_data.update(metadata)


path = f'_n{n}'
# if normalize:
#     path += "_normalize"
if not_fix_testset:
    path += "_notfixtestset"
path += "_Ynotnormalize"

if not os.path.exists('data'):
    os.mkdir('data')

path += "_loop1"
if not_fix_testset:
    # 如果not_fix_testset为True，将所有数据保存在一个文件中 
    #final_data = {**final_train_data, **test_data}
    save_data(full_data, f'data/train_data{path}.npz')
    print("\nData saved with path (not_fix_testset=True):", path)
else:
    # 如果not_fix_testset为False，按原来的方式分割保存训练集和测试集
    save_data(train_data, f'data/train_data{path}.npz') 
    save_data(test_data, f'data/test_data{path}.npz')
    print("\nData saved with path (not_fix_testset=False):", path)

print("Data generation and saving complete.")
