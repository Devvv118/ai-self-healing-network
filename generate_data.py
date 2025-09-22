import numpy as np
import pandas as pd
import pandapower as pp
# from copy import deepcopy

from network import create_network, calculate_pi_qi_ui, calculate_Ii_INi

def perturb_loads(nodes_df, pd_col='PD', qd_col='QD', pd_factor=(0.8,1.2), qd_factor=(0.8,1.2)):
    df = nodes_df.copy()
    if pd_col in df.columns:
        df[pd_col] = df[pd_col] * np.random.uniform(*pd_factor, size=len(df))
    if qd_col in df.columns:
        df[qd_col] = df[qd_col] * np.random.uniform(*qd_factor, size=len(df))
    return df

def generate_corrupted_sample(perfect_result, noise_level=0.05, drop_prob=0.1):
    noisy = perfect_result.copy()
    # Add Gaussian noise
    for col in perfect_result.columns:
        noise = np.random.normal(0, noise_level, size=len(noisy))
        noisy[col] += noise
    # Randomly drop values
    mask = np.random.rand(*noisy.shape) < drop_prob
    noisy[mask] = np.nan
    return noisy

def simulate_and_export(lines_df, nodes_df, iterations=500, corrupted_copies=20):
    results = []
    for i in range(iterations):
        perturbed_nodes = perturb_loads(nodes_df)
        net, _ = create_network(lines_df, perturbed_nodes)
        Pi, Qi, Ui, Ri, si, net_res_df = calculate_pi_qi_ui(net)
        Ii, INi = calculate_Ii_INi(net)
        perfect_result = pd.DataFrame({'Pi': Pi, 'Qi': Qi, 'Ui': Ui, 'Ii': Ii})
        for j in range(corrupted_copies):
            corrupted = generate_corrupted_sample(perfect_result)
            merged = pd.concat(
                [corrupted.add_suffix("_noisy"), perfect_result.add_suffix("_clean")],
                axis=1
            )
            results.append(merged)
        print(f"Iteration {i+1}/{iterations} done.")
    big_df = pd.concat(results, ignore_index=True)
    big_df.to_csv(f'training_data_{iterations}x{corrupted_copies}.csv', index=False)
    print("Synthetic training set created.")

if __name__ == "__main__":
    lines_df = pd.read_csv('Lines_33.csv')
    nodes_df = pd.read_csv('Nodes_33.csv')
    simulate_and_export(lines_df, nodes_df, iterations = 500, corrupted_copies=20)