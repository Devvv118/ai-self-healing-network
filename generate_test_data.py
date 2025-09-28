import numpy as np
import pandas as pd
import pandapower as pp

from network import create_network, calculate_pi_qi_ui, calculate_Ii_INi

def perturb_loads(nodes_df, lines_df ,pd_factor=(0.5, 1.5), qd_factor=(0.5, 1.5), r_factor=(0.7, 1.3), x_factor=(0.7, 1.3)):
    n_df = nodes_df.copy()
    l_df = lines_df.copy()
    if 'PD' in n_df.columns:
        n_df['PD'] = n_df['PD'] * np.random.uniform(*pd_factor, size=len(n_df))
    if 'QD' in n_df.columns:
        n_df['QD'] = n_df['QD'] * np.random.uniform(*qd_factor, size=len(n_df))
    if 'R' in l_df.columns:
        l_df['R'] = l_df['R'] * np.random.uniform(*r_factor, size=len(l_df))
    if 'X' in l_df.columns:
        l_df['X'] = l_df['X'] * np.random.uniform(*x_factor, size=len(l_df))
    return n_df, l_df

def perturb_loads_extreme(nodes_df, lines_df,
                  pd_factors=((0.2, 0.5), (1.5, 2.0)),
                  qd_factors=((0.2, 0.5), (1.5, 2.0)),
                  r_factors=((0.7, 0.9), (1.1, 1.3)),
                  x_factors=((0.7, 0.9), (1.1, 1.3))):
    n_df = nodes_df.copy()
    l_df = lines_df.copy()
    
    def sample_disjoint(size, low_range, high_range):
        # Randomly choose which range each element comes from
        choices = np.random.choice([0, 1], size=size)
        result = np.empty(size)
        # sample from the lower range
        result[choices == 0] = np.random.uniform(*low_range, size=np.sum(choices == 0))
        # sample from the higher range
        result[choices == 1] = np.random.uniform(*high_range, size=np.sum(choices == 1))
        return result

    if 'PD' in n_df.columns:
        n_df['PD'] = n_df['PD'] * sample_disjoint(len(n_df), *pd_factors)
    if 'QD' in n_df.columns:
        n_df['QD'] = n_df['QD'] * sample_disjoint(len(n_df), *qd_factors)
    if 'R' in l_df.columns:
        l_df['R'] = l_df['R'] * sample_disjoint(len(l_df), *r_factors)
    if 'X' in l_df.columns:
        l_df['X'] = l_df['X'] * sample_disjoint(len(l_df), *x_factors)

    return n_df, l_df


def introduce_noise(perfect_result, noise_level=0.05):
    noisy = perfect_result.copy()
    for col in perfect_result.columns:
        col_std = perfect_result[col].std()
        noise = np.random.normal(0, noise_level * col_std, size=len(noisy))
        noisy[col] += noise
    return noisy

def introduce_missing_data(df):
    corrupted = df.copy()
    num_cols = min(4, corrupted.shape[1])
    for i in range(len(corrupted)):
        col_to_nan = i % num_cols
        corrupted.iat[i, col_to_nan] = np.nan
    return corrupted

def simulate_and_export(lines_df, nodes_df, iterations=5):
    results = []
    perturbed_nodes, perturbed_lines = perturb_loads(nodes_df, lines_df)
    net, _ = create_network(perturbed_lines, perturbed_nodes)
    Pi, Qi, Ui, Ri, Xi, si, net_res_df = calculate_pi_qi_ui(net)
    Ii, INi = calculate_Ii_INi(net)
    perfect_result = pd.DataFrame({'Pi': Pi, 'Qi': Qi, 'Ui': Ui, 'Ii': Ii})
    extras = pd.DataFrame({'Ri': Ri, 'Xi': Xi, 'si': si})
    corrupted = introduce_noise(perfect_result)
    merged = pd.concat(
        [corrupted.add_suffix("_noisy"), perfect_result.add_suffix("_clean"), extras],
        axis=1
    )
    results.append(merged)
    test_df = pd.concat(results, ignore_index=True)
    test_df.to_csv(f'generated_data/normal_testing_data.csv', index=False)


    results = []
    perturbed_nodes, perturbed_lines = perturb_loads_extreme(nodes_df, lines_df)
    net, _ = create_network(perturbed_lines, perturbed_nodes)
    Pi, Qi, Ui, Ri, Xi, si, net_res_df = calculate_pi_qi_ui(net)
    Ii, INi = calculate_Ii_INi(net)
    perfect_result = pd.DataFrame({'Pi': Pi, 'Qi': Qi, 'Ui': Ui, 'Ii': Ii})
    extras = pd.DataFrame({'Ri': Ri, 'Xi': Xi, 'si': si})
    corrupted = introduce_noise(perfect_result)
    merged = pd.concat(
        [corrupted.add_suffix("_noisy"), perfect_result.add_suffix("_clean"), extras],
        axis=1
    )
    results.append(merged)
    test_df = pd.concat(results, ignore_index=True)
    test_df.to_csv(f'generated_data/extreme_testing_data.csv', index=False)


    results = []
    perturbed_nodes, perturbed_lines = perturb_loads(nodes_df, lines_df)
    net, _ = create_network(perturbed_lines, perturbed_nodes)
    Pi, Qi, Ui, Ri, Xi, si, net_res_df = calculate_pi_qi_ui(net)
    Ii, INi = calculate_Ii_INi(net)
    perfect_result = pd.DataFrame({'Pi': Pi, 'Qi': Qi, 'Ui': Ui, 'Ii': Ii})
    extras = pd.DataFrame({'Ri': Ri, 'Xi': Xi, 'si': si})
    corrupted = introduce_noise(perfect_result)
    corrupted = introduce_missing_data(corrupted)
    merged = pd.concat(
        [corrupted.add_suffix("_noisy"), perfect_result.add_suffix("_clean"), extras],
        axis=1
    )
    results.append(merged)
    test_df = pd.concat(results, ignore_index=True)
    test_df.to_csv(f'generated_data/missing_testing_data.csv', index=False)

    print("Synthetic testing set created.")

if __name__ == "__main__":
    lines_df = pd.read_csv('Lines_33.csv')
    nodes_df = pd.read_csv('Nodes_33.csv')
    simulate_and_export(lines_df, nodes_df, iterations=5)