from tensorflow import keras
import pandas as pd
import numpy as np

from functions import calculate_minf1, calculate_minf2

def compare(model_path="my_regression_model2500x2.keras",csv_path="generated_data/training_data_2500x2.csv"):

    model = keras.models.load_model(model_path)
    df = pd.read_csv(csv_path)
    samples = df.sample(5)

    noisy_df = samples.drop(columns=["Pi_clean","Qi_clean","Ui_clean","Ii_clean"], axis=1)
    clean_df = samples.drop(columns=["Pi_noisy","Qi_noisy","Ui_noisy","Ii_noisy"], axis=1)
    X_samples = noisy_df.drop(columns=["Ri","Xi","si"], axis=1)

    predicted_clean_samples = model.predict(X_samples)

    predicted_clean_df = pd.DataFrame(predicted_clean_samples, columns=["Pi_predicted", "Qi_predicted", "Ui_predicted", "Ii_predicted"])

    predicted_clean_df['Ri'] = noisy_df['Ri'].values
    predicted_clean_df['Xi'] = noisy_df['Xi'].values
    predicted_clean_df['si'] = noisy_df['si'].values

    def calculate_metrics(df, prefix):
        df_filled = df.fillna(0)
        Pi = df_filled[f'Pi_{prefix}'].values
        Qi = df_filled[f'Qi_{prefix}'].values
        Ui = df_filled[f'Ui_{prefix}'].values
        Ri = df_filled['Ri'].values
        si = df_filled['si'].values
        Ii = df_filled[f'Ii_{prefix}'].values

        minf1 = calculate_minf1(Pi, Qi, Ui, Ri, si)

        INi = np.ones_like(Ii)
        minf2 = calculate_minf2(Ii, INi)

        return minf1, minf2

    minf1_clean, minf2_clean = calculate_metrics(clean_df, 'clean')
    minf1_noisy, minf2_noisy = calculate_metrics(noisy_df, 'noisy')
    minf1_predicted, minf2_predicted = calculate_metrics(predicted_clean_df, 'predicted')

    print("----- Clean Data Sample -----")
    print(clean_df.head())
    print("----- Noisy Data Sample -----")
    print(noisy_df.head())
    print("----- Predicted Data Sample -----")
    print(predicted_clean_df.head())

    print("-" * 30)
    print("--- Comparison of minf1 and minf2 ---")
    print(f"minf1 (Clean):     {minf1_clean:.6f}")
    print(f"minf1 (Noisy):     {minf1_noisy:.6f}")
    print(f"minf1 (Predicted): {minf1_predicted:.6f}")
    print("-" * 30)
    print(f"minf2 (Clean):     {minf2_clean:.6f}")
    print(f"minf2 (Noisy):     {minf2_noisy:.6f}")
    print(f"minf2 (Predicted): {minf2_predicted:.6f}")

compare()
# compare(csv_path="generated_data/missing_testing_data.csv")
