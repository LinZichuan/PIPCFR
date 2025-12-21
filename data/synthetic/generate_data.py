import sys
import numpy as np
import scipy.special
import csv
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from scipy.stats import t as t_dist

generate_num_exp = False
#not_fix_testset = True  # 设置是否固定测试集
not_fix_testset = True

num_exp = 100
# Parameters
m_x = 10
m_t = 1
m_v = 10
m_m = 10
m_a = 10
n = int(sys.argv[1])
seed = 42
dep = False
A_noise_scale = 1.0
outcomenoise = 1.0
noise_scale = float(sys.argv[2])
A_noise_scale = outcomenoise = noise_scale
binary = False
normalize = True  # 默认开启归一化
normalize_exceptY = False
onlyVMA = True
XdependA = True
entangle = False
nl_fn = 'blank'
normaloutcome = False
timevaring_noise = False
nonlinear_outcome = False
gamma = 0.9
virtual_num_steps = 5
n += virtual_num_steps
yplus1 = False
xscale = 1
sanjieju = False
normal_outcome_coef = 0
for i in range(n):
    normal_outcome_coef += gamma ** i
julei = False
save_scaler = True  # 添加save_scaler参数
last_rate = 0.05  # 添加last_rate参数

act_fn = lambda x: x
if nl_fn == 'sigmoid':
    def act_fn(x):
        return 1 / (1 + np.exp(-x))
elif nl_fn == 'sin':
    def act_fn(x):
        return np.sin(x)
elif nl_fn == 'tanh':
    def act_fn(x):
        return np.tanh(x)
elif nl_fn == 'blank':
    def act_fn(x):
        return x

def generate_stable_matrix(m, radius_thred=0.95):
    A = np.random.rand(m, m) * 2 - 1
    eigenvalues = np.linalg.eigvals(A)
    spectral_radius = max(abs(eigenvalues))
    if spectral_radius >= radius_thred:
        A = A * (radius_thred / spectral_radius)
    return A

def get_initial_latent(m, num_samples, seed, dep):
    np.random.seed(seed)
    if dep:
        mu = np.random.normal(size=m) / 10.
        temp = np.random.uniform(size=(m, m))
        temp = .5 * (np.transpose(temp) + temp)
        sig = (temp + m * np.eye(m)) / 100.
    else:
        mu = np.zeros(m)
        sig = np.eye(m)
    return np.array([np.random.multivariate_normal(mean=mu, cov=sig) for _ in range(num_samples)])

def generate_treatment_batch(x_batch, coefs_xt, noise_scale=0.1, seed=None):
    np.random.seed(seed + 1)
    z = np.dot(x_batch, coefs_xt)
    pi = scipy.special.expit(z + np.random.normal(scale=noise_scale, size=z.shape))
    return np.random.binomial(1, pi)

def get_initial_latent(m, num_samples, seed, dep):
    np.random.seed(seed)
    if dep:
        mu = np.random.normal(size=m) / 10.
        temp = np.random.uniform(size=(m, m))
        temp = .5 * (np.transpose(temp) + temp)
        sig = (temp + m * np.eye(m)) / 100.
    else:
        mu = np.zeros(m)
        sig = np.eye(m)
    return np.array([np.random.multivariate_normal(mean=mu, cov=sig) for _ in range(num_samples)])

def generate_treatment_batch(x_batch, coefs_xt, noise_scale=0.1, seed=None):
    np.random.seed(seed + 1)
    z = np.dot(x_batch, coefs_xt)
    pi = scipy.special.expit(z + np.random.normal(scale=noise_scale, size=z.shape))
    return np.random.binomial(1, pi)

def update_latent_batch(x_batch, t_batch, coefs_tx, coefs_xx, noise_scale=0.1, seed=None):
    np.random.seed(seed + 2)
    # return act_fn(np.dot(x_batch, coefs_xx) + np.dot(t_batch, coefs_tx) + np.random.laplace(scale=noise_scale, size=x_batch.shape))
    return x_batch + np.dot(t_batch, coefs_tx) + np.random.laplace(scale=noise_scale, size=x_batch.shape)

def generate_V_batch(x_batch, t_batch, coefs_xv, coefs_tv, noise_scale=0.1, seed=None):
    np.random.seed(seed + 3)
    V = np.dot(x_batch, coefs_xv) + np.dot(t_batch, coefs_tv) + np.random.laplace(scale=noise_scale, size=(x_batch.shape[0], coefs_xv.shape[1]))
    return act_fn(V)

def generate_M_batch(x_batch, t_batch, coefs_xm, coefs_tm, noise_scale=0.1, seed=None):
    np.random.seed(seed + 4)
    M = np.dot(x_batch, coefs_xm) + np.dot(t_batch, coefs_tm) + np.random.laplace(scale=noise_scale, size=(x_batch.shape[0], coefs_xm.shape[1]))
    return act_fn(M)

def generate_A_batch(x_batch, coefs_xa, noise_scale=0.1, seed=None):
    np.random.seed(seed + 5)
    if XdependA:
        A = np.dot(x_batch, coefs_xa) + np.random.laplace(scale=noise_scale, size=(x_batch.shape[0], coefs_xa.shape[1]))
    else:
        A = np.random.laplace(scale=noise_scale, size=(x_batch.shape[0], coefs_xa.shape[1]))
    return act_fn(A)

def generate_outcome_batch(X_all, T, M_all, A_all, coefs_xy, coefs_ty, coefs_my, coefs_ay, 
                         T_true, noise_scale=0.1, binary=False, gamma=0.99, 
                         cluster_labels=None, cluster_coeffs=None, kmeans_model=None,
                         seed=None):
    np.random.seed(seed)
    batch_size = X_all.shape[0]
    n = X_all.shape[1]

    T = T.reshape(-1, 1) if len(T.shape) == 1 else T

    outcome = np.zeros(batch_size)
    outcome_1_1 = np.zeros(batch_size)
    outcome_1_2 = np.zeros(batch_size)
    outcome_1_3 = np.zeros(batch_size)
    
    # 计算开始时间步
    start_t = virtual_num_steps if virtual_num_steps > 0 else 0
    # 根据last_rate计算实际使用的时间步范围
    actual_start_t = max(start_t, int(n - n * last_rate))
    
    if nonlinear_outcome:
        mask = (T == 1).flatten()
        for t in range(actual_start_t, n):  # 从actual_start_t开始遍历到最后
            decay = gamma ** (n - t - 1)
            # 对T=1的样本使用非线性计算
            decay = 1
            if np.any(mask):
                outcome[mask] += decay * (np.dot(X_all[mask, t] ** 2, coefs_xy) + 
                                       np.dot(M_all[mask, t] ** 2, coefs_my) + 
                                       np.dot(A_all[mask, t] ** 2, coefs_ay)).flatten()
            # 对T=0的样本使用线性计算
            if np.any(~mask):
                outcome[~mask] += decay * (np.dot(X_all[~mask, t], coefs_xy) + 
                                        np.dot(M_all[~mask, t], coefs_my) + 
                                        np.dot(A_all[~mask, t], coefs_ay)).flatten()
    else:
        for t in range(actual_start_t, n):  # 从actual_start_t开始遍历到最后
            decay = gamma ** (n - t - 1)
            if normaloutcome:
                outcome += decay * (np.dot(X_all[:, t], coefs_xy) + 
                                  np.dot(M_all[:, t], coefs_my) + 
                                  np.dot(A_all[:, t], coefs_ay)).flatten() / normal_outcome_coef
            else:
                decay = 1
                outcome += decay * (np.dot(X_all[:, t], coefs_xy) + 
                                  np.dot(M_all[:, t], coefs_my) * (1/xscale) + 
                                  np.dot(A_all[:, t], coefs_ay) * (1/xscale)                                 
                                  ).flatten() * (1 + np.random.laplace(scale=outcomenoise, size=(X_all.shape[0])))
            
            outcome_1_1 += decay * (np.dot(X_all[:, t], coefs_xy)).flatten()
            outcome_1_2 += decay * (np.dot(M_all[:, t], coefs_my) * (1/xscale)).flatten() 
            outcome_1_3 += decay * (np.dot(A_all[:, t], coefs_ay) * (1/xscale)).flatten()

    if normaloutcome:
        outcome_1_1 /= normal_outcome_coef
        outcome_1_2 /= normal_outcome_coef
        outcome_1_3 /= normal_outcome_coef
    
    if n * last_rate < 1:
        outcome = outcome * (n * last_rate)
    
    outcome_1 = outcome.copy()
    outcome_2 = np.zeros_like(outcome) if nonlinear_outcome else np.dot(T, coefs_ty).flatten() * (1/xscale)
    outcome += outcome_2
    
    outcome_3 = np.zeros_like(outcome)
    if timevaring_noise:
        outcome_3 += np.random.normal(scale=noise_scale * (n * 0.1), size=batch_size)
        outcome += outcome_3
    if sanjieju:
        # 增: 计算V和M的波动性作为S
        VM_combined = np.concatenate([M_all, A_all], axis=2)  # 合并V和M
        VM_reshaped = np.mean(VM_combined, axis=2)  # 对最后一维取平均
        mean_S = np.power(
            np.abs(VM_reshaped - np.mean(VM_reshaped, axis=1, keepdims=True)), 
            3
        )
        mean_S = np.mean(mean_S, axis=1)
        offset_Y = np.exp(mean_S)
        outcome_3 = offset_Y * T_true.flatten()
        outcome += outcome_3
        
        print("outcome3",outcome_3[:10],outcome_3.shape, np.mean(outcome_3), np.std(outcome_3))
        print("offset_Y", offset_Y[:10],offset_Y.shape, np.mean(offset_Y), np.std(offset_Y))
        print("T_true_outcome", T_true[:10], T_true.shape, np.mean(T_true), np.std(T_true))

    if julei and cluster_labels is not None and cluster_coeffs is not None:
        # 获取每个样本对应的系数
        sample_coeffs = cluster_coeffs[cluster_labels]
        # 只对T_true = 1的样本应用系数调整
        outcome_3 = np.zeros_like(outcome)
        mask = (T_true.flatten() == 1)
        outcome_3[mask] = outcome[mask] * (1 + sample_coeffs[mask])
        
        outcome += outcome_3
        print("\nCluster Statistics for T=1 samples:")
        for i in range(len(cluster_coeffs)):
            cluster_mask = (cluster_labels == i) & mask
            cluster_size = np.sum(cluster_mask)
            if cluster_size > 0:  # 只打印有T=1样本的类别统计信息
                print(f"Cluster {i} (coeff {cluster_coeffs[i]}): {cluster_size} T=1 samples")
                print(f"Mean outcome in cluster {i}: {np.mean(outcome[cluster_mask])}")
    
    if binary:
        outcome = np.random.binomial(1, scipy.special.expit(outcome))
    
    return act_fn(outcome), outcome_1, outcome_2, outcome_3, outcome_1_1, outcome_1_2, outcome_1_3

def generate_dataset(m_x, m_t, m_v, m_m, m_a, n, seed, dep, noise_scale, A_noise_scale, num_samples, binary=False, normalize=True):
    # 生成系数
    np.random.seed(seed - 1)
    coefs_xx = generate_stable_matrix(m_x)
    coefs_xt = np.random.normal(size=(m_x, m_t))
    coefs_xv = np.random.normal(size=(m_x, m_v))
    coefs_tv = np.random.normal(size=(m_t, m_v))
    coefs_xm = np.random.normal(size=(m_x, m_m))
    coefs_tm = np.random.normal(size=(m_t, m_m))
    coefs_xa = np.random.normal(size=(m_x, m_a))
    coefs_xy = np.random.normal(size=(m_x, 1))
    coefs_ty = np.random.normal(size=(m_t, 1))
    coefs_my = np.random.normal(size=(m_m, 1))
    coefs_ay = np.random.normal(size=(m_a, 1))
    coefs_tx = np.random.normal(size=(m_t, m_x))
    coefs_en = np.random.normal(size=(m_a+m_m+m_v, m_a+m_m+m_v))

    # 初始化数据存储
    X = get_initial_latent(m_x, num_samples, seed, dep)
    X_all = np.zeros((num_samples, n, m_x))
    T_all = np.zeros((num_samples, n))
    V_all = np.zeros((num_samples, n, m_v))
    M_all = np.zeros((num_samples, n, m_m))
    A_all = np.zeros((num_samples, n, m_a))
    
    # 第一步：生成所有X和T
    print("Step 1: Generating X and T sequences...")
    X_current = X.copy()
    for i in tqdm(range(n)):
        new_seed = seed * n * 10 + i
        X_all[:, i] = X_current
        T = generate_treatment_batch(X_current, coefs_xt, noise_scale, new_seed)
        T_all[:, i] = T.flatten()
        X_current = update_latent_batch(X_current, T, coefs_tx, coefs_xx, noise_scale, new_seed)
    
    # 第二步：对所有X进行归一化
    print("Step 2: Normalizing X...")
    if normalize:
        x_scaler = StandardScaler()
        X_all_reshaped = X_all.reshape(-1, m_x)
        X_all_normalized = x_scaler.fit_transform(X_all_reshaped)
        X_all = X_all_normalized.reshape(num_samples, n, m_x)
    

    # 第三步：使用归一化后的X生成VMA
    print("Step 3: Generating V, M, A sequences...")
    for i in tqdm(range(n)):
        new_seed = seed * n * 10 + i
        V = generate_V_batch(X_all[:, i], T_all[:, i:i+1], coefs_xv, coefs_tv, noise_scale, new_seed)
        M = generate_M_batch(X_all[:, i], T_all[:, i:i+1], coefs_xm, coefs_tm, noise_scale, new_seed)
        A = generate_A_batch(X_all[:, i], coefs_xa, A_noise_scale, new_seed)
        
        V_all[:, i] = V
        M_all[:, i] = M
        A_all[:, i] = A
    
    # 第四步：对VMA进行整体归一化
    print("Step 4: Normalizing V, M, A...")
    if normalize:
        v_scaler = StandardScaler()
        m_scaler = StandardScaler()
        a_scaler = StandardScaler()
        
        V_all_reshaped = V_all.reshape(-1, m_v)
        M_all_reshaped = M_all.reshape(-1, m_m)
        A_all_reshaped = A_all.reshape(-1, m_a)
        
        V_all = v_scaler.fit_transform(V_all_reshaped).reshape(num_samples, n, m_v)
        M_all = m_scaler.fit_transform(M_all_reshaped).reshape(num_samples, n, m_m)
        A_all = a_scaler.fit_transform(A_all_reshaped).reshape(num_samples, n, m_a)
    
    # 第五步：生成outcome
    print("Step 5: Generating outcomes...")
    
    cluster_labels = None
    cluster_coeffs = None
    kmeans = None
    
    if julei:
        # 在生成outcome之前，对S进行聚类
        print("Clustering surrogate variables...")
        # 修改：合并M和A作为surrogate，而不是V和M
        S_combined = np.concatenate([M_all, A_all], axis=2)  # shape: [num_samples, n, m_m+m_a]
        S_mean = np.mean(S_combined, axis=1)  # 对时间维度取平均，shape: [num_samples, m_m+m_a]
        
        # 使用KMeans进行聚类
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
        cluster_labels = kmeans.fit_predict(S_mean)
        
        # 定义每个类别的系数
        cluster_coeffs = np.array([-0.3, -0.15, 0, 0.15, 0.3])
        
    
    Y_all, outcome_1, outcome_2, outcome_3, outcome_1_1, outcome_1_2, outcome_1_3 = generate_outcome_batch(
        X_all, T_all[:, -1], M_all, A_all, coefs_xy, coefs_ty, coefs_my, coefs_ay, 
        T_all[:, virtual_num_steps], noise_scale, binary, gamma, 
        cluster_labels=cluster_labels, cluster_coeffs=cluster_coeffs,
        kmeans_model=kmeans,  # 传入模型用于反事实计算
        seed=seed
    )
    
    # 打印outcome统计信息
    print("\nOutcome Components Statistics:")
    print(f"outcome_1 {outcome_1.shape}", np.mean(outcome_1), np.std(outcome_1))
    print(f"outcome_2 {outcome_2.shape}", np.mean(outcome_2), np.std(outcome_2))
    print(f"outcome_3 {outcome_3.shape}", np.mean(outcome_3), np.std(outcome_3))
    print(f"outcome_1_1 {outcome_1_1.shape}", np.mean(outcome_1_1), np.std(outcome_1_1))
    print(f"outcome_1_2 {outcome_1_2.shape}", np.mean(outcome_1_2), np.std(outcome_1_2))
    print(f"outcome_1_3 {outcome_1_3.shape}", np.mean(outcome_1_3), np.std(outcome_1_3))
    
    # 第六步：生成反事实结果
    print("\nStep 6: Generating counterfactuals...")
    def generate_counterfactual_with_scalers(X_initial, T_initial, kmeans_model=None, cluster_coeffs=None):
        X = X_initial
        X_cf_all = np.zeros((num_samples, n, m_x))
        M_cf_all = np.zeros((num_samples, n, m_m))
        A_cf_all = np.zeros((num_samples, n, m_a))
        T_cf_all = np.zeros((num_samples, n))
        
        # 生成反事实序列
        for i in range(n):
            new_seed = seed * n * 10 + i
            if i == virtual_num_steps:  # 在virtual_num_steps位置进行干预
                T = 1 - T_initial
            else:
                T = generate_treatment_batch(X, coefs_xt, noise_scale, new_seed)
            
            X_cf_all[:, i] = X
            T_cf_all[:, i] = T.flatten()
            X = update_latent_batch(X, T, coefs_tx, coefs_xx, noise_scale, new_seed)
        
        # 使用相同的scaler归一化X
        if normalize:
            X_cf_all_reshaped = X_cf_all.reshape(-1, m_x)
            X_cf_all = x_scaler.transform(X_cf_all_reshaped).reshape(num_samples, n, m_x)
        
        # 生成并归一化VMA
        for i in range(n):
            new_seed = seed * n * 10 + i
            M = generate_M_batch(X_cf_all[:, i], T_cf_all[:, i:i+1], coefs_xm, coefs_tm, noise_scale, new_seed)
            A = generate_A_batch(X_cf_all[:, i], coefs_xa, A_noise_scale, new_seed)
            
            if normalize:
                M = m_scaler.transform(M.reshape(-1, m_m)).reshape(num_samples, -1)
                A = a_scaler.transform(A.reshape(-1, m_a)).reshape(num_samples, -1)
            
            M_cf_all[:, i] = M
            A_cf_all[:, i] = A
        
        cluster_labels_cf = None
        if julei and kmeans_model is not None:
            # 修改：对反事实的S进行聚类预测，使用M和A而不是V和M
            S_cf_combined = np.concatenate([M_cf_all, A_cf_all], axis=2)
            S_cf_mean = np.mean(S_cf_combined, axis=1)
            cluster_labels_cf = kmeans_model.predict(S_cf_mean)
        
        return generate_outcome_batch(X_cf_all, T_cf_all[:, -1], M_cf_all, A_cf_all,
                                   coefs_xy, coefs_ty, coefs_my, coefs_ay, 1 - T_initial, 
                                   noise_scale, binary, gamma, 
                                   cluster_labels=cluster_labels_cf,
                                   cluster_coeffs=cluster_coeffs,
                                   kmeans_model=kmeans_model,
                                   seed=seed)
    #print("T_initial", T_all[:, virtual_num_steps][:10], T_all[:, virtual_num_steps].shape, np.mean(T_all[:, virtual_num_steps]), np.std(T_all[:, virtual_num_steps]))
    print("T_all", T_all[:, virtual_num_steps:virtual_num_steps+1][:10], T_all[:, virtual_num_steps:virtual_num_steps+1].shape, np.mean(T_all[:, virtual_num_steps:virtual_num_steps+1]), np.std(T_all[:, virtual_num_steps:virtual_num_steps+1]))
    Y_cf_all, outcome_cf1, outcome_cf2, outcome_cf3, outcome_cf1_1, outcome_cf1_2, outcome_cf1_3 = generate_counterfactual_with_scalers(
        X_all[:, virtual_num_steps], T_all[:, virtual_num_steps:virtual_num_steps+1], 
        kmeans if julei else None, 
        cluster_coeffs if julei else None
    )
    
    # 打印反事实outcome统计信息
    print("\nCounterfactual Outcome Components Statistics:")
    print(f"outcome_cf1 {outcome_cf1.shape}", np.mean(outcome_cf1), np.std(outcome_cf1))
    print(f"outcome_cf2 {outcome_cf2.shape}", np.mean(outcome_cf2), np.std(outcome_cf2))
    print(f"outcome_cf3 {outcome_cf3.shape}", np.mean(outcome_cf3), np.std(outcome_cf3))
    print(f"outcome_cf1_1 {outcome_cf1_1.shape}", np.mean(outcome_cf1_1), np.std(outcome_cf1_1))
    print(f"outcome_cf1_2 {outcome_cf1_2.shape}", np.mean(outcome_cf1_2), np.std(outcome_cf1_2))
    print(f"outcome_cf1_3 {outcome_cf1_3.shape}", np.mean(outcome_cf1_3), np.std(outcome_cf1_3))
    
    # 对结果进行归一化
    Y_all_original = Y_all.copy()
    Y_cf_all_original = Y_cf_all.copy()
    
    
    if normalize:
        Y_scaler = MinMaxScaler()
        Y_combined = np.concatenate((Y_all.reshape(-1, 1), Y_cf_all.reshape(-1, 1)), axis=0)
        Y_combined = Y_scaler.fit_transform(Y_combined).reshape(-1)
        Y_all = Y_combined[:Y_all.size]
        Y_cf_all = Y_combined[Y_all.size:]
        if yplus1:
            Y_all[T_all[:, virtual_num_steps] == 1] += 0.1
            Y_cf_all[T_all[:, virtual_num_steps] == 0] += 0.1
    
    tau_all = np.where(T_all[:, virtual_num_steps] == 1, Y_all - Y_cf_all, Y_cf_all - Y_all)
    tau_all_original = np.where(T_all[:, virtual_num_steps] == 1, Y_all_original - Y_cf_all_original, Y_cf_all_original - Y_all_original)
    e_all = np.zeros((tau_all.shape[0], 1))

    print("\nOriginal Outcome Statistics:")
    print("Y_all_original[T_all[:, 0] == 1]:", np.mean(Y_all_original[T_all[:, virtual_num_steps] == 1]), np.std(Y_all_original[T_all[:, virtual_num_steps] == 1]))
    print("Y_all_original[T_all[:, 0] == 0]:", np.mean(Y_all_original[T_all[:, virtual_num_steps] == 0]), np.std(Y_all_original[T_all[:, virtual_num_steps] == 0]))
    print("Y_all_original:", np.mean(Y_all_original), np.std(Y_all_original))
    print("Y_cf_all_original[T_all[:, 0] == 1]:", np.mean(Y_cf_all_original[T_all[:, virtual_num_steps] == 1]), np.std(Y_cf_all_original[T_all[:, virtual_num_steps] == 1]))
    print("Y_cf_all_original[T_all[:, 0] == 0]:", np.mean(Y_cf_all_original[T_all[:, virtual_num_steps] == 0]), np.std(Y_cf_all_original[T_all[:, virtual_num_steps] == 0]))
    print("Y_cf_all_original:", np.mean(Y_cf_all_original), np.std(Y_cf_all_original))
    print("tau_all_original:", np.mean(tau_all_original), np.std(tau_all_original))
    
    print("\nNormalized Outcome Statistics:")
    print("Y_all[T_all[:, 0] == 1]:", np.mean(Y_all[T_all[:, virtual_num_steps] == 1]), np.std(Y_all[T_all[:, virtual_num_steps] == 1]))
    print("Y_all[T_all[:, 0] == 0]:", np.mean(Y_all[T_all[:, virtual_num_steps] == 0]), np.std(Y_all[T_all[:, virtual_num_steps] == 0]))
    print("Y_all:", np.mean(Y_all), np.std(Y_all))
    print("Y_cf_all[T_all[:, 0] == 1]:", np.mean(Y_cf_all[T_all[:, virtual_num_steps] == 1]), np.std(Y_cf_all[T_all[:, virtual_num_steps] == 1]))
    print("Y_cf_all[T_all[:, 0] == 0]:", np.mean(Y_cf_all[T_all[:, virtual_num_steps] == 0]), np.std(Y_cf_all[T_all[:, virtual_num_steps] == 0]))
    print("Y_cf_all:", np.mean(Y_cf_all), np.std(Y_cf_all))
    print("tau_all:", np.mean(tau_all), np.std(tau_all))

    # 生成S_flat
    if onlyVMA:
        S_flat = np.concatenate([
            V_all[:, virtual_num_steps:, :].reshape(num_samples, -1),
            M_all[:, virtual_num_steps:, :].reshape(num_samples, -1),
            A_all[:, virtual_num_steps:, :].reshape(num_samples, -1)
        ], axis=1)
    else:
        S_flat = np.concatenate([
            V_all.reshape(num_samples, -1),
            M_all.reshape(num_samples, -1),
            A_all.reshape(num_samples, -1),
            X_all[:,1:,:].reshape(num_samples, -1)
        ], axis=1)   
    
    
    # 绘制分布图
    plot_tau_density = True
    if plot_tau_density:
        density = gaussian_kde(tau_all)
        x = np.linspace(min(tau_all), max(tau_all), 1000)
        density_values = density(x)
        
        plt.figure(figsize=(8, 6))
        plt.plot(x, density_values, label='Probability Density')
        plt.fill_between(x, density_values, alpha=0.5)
        plt.title('Probability Density of tau_all, n=' + str(n))
        plt.xlabel('tau_all')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        if not os.path.exists('figures'):
            os.mkdir('figures')
        plt.savefig('figures/tau_n{}_noise{}_outcomenoise{}_normalize{}_vsteps{}_v8.png'.format(
            n-virtual_num_steps, noise_scale, outcomenoise, normalize, virtual_num_steps))
        plt.close()
    
    plot_y_density = True
    if plot_y_density:
        density = gaussian_kde(Y_all)
        x = np.linspace(min(Y_all), max(Y_all), 1000)
        density_values = density(x)
        
        plt.figure(figsize=(8, 6))
        plt.plot(x, density_values, label='Probability Density')
        plt.fill_between(x, density_values, alpha=0.5)
        plt.title('Probability Density of Y_all, n=' + str(n))
        plt.xlabel('Y_all')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        if not os.path.exists('figures'):
            os.mkdir('figures')
        plt.savefig('figures/Y_n{}_noise{}_normalize{}_vsteps{}_v8.png'.format(
            n-virtual_num_steps, noise_scale, normalize, virtual_num_steps))
        plt.close()
    
    X_all = X_all[:, virtual_num_steps, :].squeeze()
    # 打印统计信息
    print_stats = True
    if print_stats:
        print("\nData Statistics:")
        print("X_all shape:", X_all.shape)
        print("S_flat shape:", S_flat.shape)
        print("T_all shape:", T_all.shape)
        print("Y_all shape:", Y_all.shape)
        print("Y_cf_all shape:", Y_cf_all.shape)
        print("tau_all shape:", tau_all.shape)
        print("V_all shape:", V_all.shape)
        print("M_all shape:", M_all.shape)
        print("A_all shape:", A_all.shape)
        print("e_all shape:", e_all.shape)
        
        print("\nOutcome Statistics:")
        print("Y_all[T_all[:, 0] == 1]:", np.mean(Y_all[T_all[:, virtual_num_steps] == 1]), np.std(Y_all[T_all[:, virtual_num_steps] == 1]))
        print("Y_all[T_all[:, 0] == 0]:", np.mean(Y_all[T_all[:, virtual_num_steps] == 0]), np.std(Y_all[T_all[:, virtual_num_steps] == 0]))
        print("Y_all:", np.mean(Y_all), np.std(Y_all))
        print("Y_cf_all[T_all[:, 0] == 1]:", np.mean(Y_cf_all[T_all[:, virtual_num_steps] == 1]), np.std(Y_cf_all[T_all[:, virtual_num_steps] == 1]))
        print("Y_cf_all[T_all[:, 0] == 0]:", np.mean(Y_cf_all[T_all[:, virtual_num_steps] == 0]), np.std(Y_cf_all[T_all[:, virtual_num_steps] == 0]))
        print("Y_cf_all:", np.mean(Y_cf_all), np.std(Y_cf_all))
        print("tau_all:", np.mean(tau_all), np.std(tau_all))
    
    result = {
        'x': X_all[:, :, np.newaxis],
        'next_x': S_flat[:, :, np.newaxis],
        't': T_all[:, virtual_num_steps, np.newaxis],
        'yf': Y_all[:, np.newaxis],
        'ycf': Y_cf_all[:, np.newaxis],
        'tau': tau_all[:, np.newaxis],
        'e': e_all,
        'yf_original': Y_all_original[:, np.newaxis],
        'ycf_original': Y_cf_all_original[:, np.newaxis],
        'tau_original': tau_all_original[:, np.newaxis],
    }
    if save_scaler:
        result['y_scaler'] = Y_scaler
    return result

def create_experiments(train_data, num_exp=5, exp_ratio=0.9, seed=None):
    num_samples = train_data['x'].shape[0]
    exp_size = int(num_samples * exp_ratio)
    experiments = []
    np.random.seed(seed)
    
    for i in range(num_exp):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        exp_indices = indices[:exp_size]
        exp_data = {key: value[exp_indices] for key, value in train_data.items()}
        experiments.append(exp_data)
    
    combined_experiments = {}
    for key in train_data.keys():
        combined_experiments[key] = np.stack([exp[key] for exp in experiments], axis=-1).squeeze()
    
    return combined_experiments

if __name__ == "__main__":
    # 生成数据
    full_data = generate_dataset(m_x, m_t, m_v, m_m, m_a, n, seed, dep, noise_scale, A_noise_scale, 
                               num_samples=10000, binary=binary, normalize=normalize)
    
    if generate_num_exp:
        #创建实验数据
        final_data = create_experiments(full_data, num_exp=num_exp, seed=seed)
    else:
        final_data = full_data
    
    # 添加元数据
    metadata = {'m_x': m_x, 'm_v': m_v, 'm_m': m_m, 'm_a': m_a, 'num_timesteps': n - virtual_num_steps}
    final_data.update(metadata)
    
    # 构建保存路径
    path = 'binary_' if binary else ''
    path += "v8_numexp{}_dep{}_n{}_x{}_v{}_m{}_a{}_noise_scale{}_A_noise_scale{}_vsteps{}".format(
        num_exp, dep, n - virtual_num_steps, m_x, m_v, m_m, m_a, noise_scale, A_noise_scale, virtual_num_steps
    )
    if normalize:
        path += "_normalize"
    if onlyVMA:
        path += "_onlyVMA"
    if XdependA:
        path += "_XdependA"
    if normaloutcome:
        path += "_normaloutcome"
    if timevaring_noise:
        path += "_timevaring_noise"
    if yplus1:
        path += "_yplus1"
    if xscale != 1:
        path += "_xscale{}".format(xscale)
    if sanjieju:
        path += "_sanjieju"
    if julei:
        path += "_julei"
    if not_fix_testset:
        path += "_notfixtestset"
    if normalize_exceptY:
        path += "_normalize_exceptY"
    if save_scaler:
        path += "_save_scaler"
    if last_rate != 1.0:
        path += "_lastrate{}".format(last_rate)
    #if outcomenoise:
    path += "_outcomenoise{}".format(outcomenoise)
        
    # 保存数据
    if not os.path.exists('data/'):
        os.mkdir('data/')
        
    
    
    if not_fix_testset:
        # 如果not_fix_testset为True，将所有数据保存在train文件中
        np.savez(f'data/train_data_{path}.npz', **final_data)
        print("\nData saved with path (not_fix_testset=True):", path)
    else:
        # 如果not_fix_testset为False，按原来的方式分割并保存训练集和测试集
        split_index = int(0.8 * len(final_data['x']))
        train_data = {key: value[:split_index] for key, value in final_data.items() if isinstance(value, np.ndarray)}
        test_data = {key: value[split_index:] for key, value in final_data.items() if isinstance(value, np.ndarray)}
        
        # 添加元数据到训练集和测试集
        for key, value in final_data.items():
            if not isinstance(value, np.ndarray):
                train_data[key] = value
                test_data[key] = value
        
        np.savez(f'data/train_data_{path}.npz', **train_data)
        np.savez(f'data/test_data_{path}.npz', **test_data)
        print("\nData saved with path (not_fix_testset=False):", path) 
