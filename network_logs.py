import numpy as np
import pandas as pd
import pandapower as pp
import time

def log(msg):
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}")

def create_ieee33_network_from_dataframes(lines_df, nodes_df):
    log("Initializing empty network ...")
    net = pp.create_empty_network(name="IEEE 33-bus")
    base_voltage_kv = 12.66
    bus_mapping = {}
    for _, row in nodes_df.iterrows():
        node_num = int(row['NODES'])
        bus_idx = pp.create_bus(net, vn_kv=base_voltage_kv, name=f"Bus_{node_num}")
        bus_mapping[node_num] = bus_idx

    log("Adding external grid at slack bus ...")
    slack_bus = bus_mapping[1]
    pp.create_ext_grid(net, slack_bus, vm_pu=1.0, va_degree=0.0)

    log("Creating loads ...")
    for _, row in nodes_df.iterrows():
        node_num = int(row['NODES'])
        if node_num != 1:
            pd_mw = row['PD'] / 1000.0
            qd_mvar = row['QD'] / 1000.0
            if pd_mw > 0 or qd_mvar > 0:
                pp.create_load(net, bus_mapping[node_num],
                               p_mw=pd_mw, q_mvar=qd_mvar,
                               name=f"Load_{node_num}")

    log("Building lines between buses ...")
    for _, row in lines_df.iterrows():
        if row['STATUS'] == 1:
            from_bus = bus_mapping[int(row['FROM'])]
            to_bus = bus_mapping[int(row['TO'])]
            r_ohm_per_km = row['R']
            x_ohm_per_km = row['X']
            c_nf_per_km = row['B'] * 1000 if row['B'] > 0 else 0.1
            length_km = 1.0
            pp.create_line_from_parameters(
                net, from_bus, to_bus, length_km=length_km,
                r_ohm_per_km=r_ohm_per_km, x_ohm_per_km=x_ohm_per_km,
                c_nf_per_km=c_nf_per_km, max_i_ka=0.4,
                name=f"Line_{row['FROM']}_{row['TO']}"
            )
    log("Network construction complete.")
    return net, bus_mapping

def calculate_pi_qi_ui(net):
    log("Running power flow ...")
    start = time.time()
    try:
        pp.runpp(net, verbose=False)
        log("Power flow converged successfully!")
    except Exception as e:
        log(f"Power flow did not converge. {e}")
        return None, None, None, None, None, None
    log(f"Power flow finished in {time.time() - start:.2f} seconds.")

    Pi = net.res_line.p_from_mw.values
    Qi = net.res_line.q_from_mvar.values
    to_buses = net.line.to_bus.values
    Ui = net.res_bus.vm_pu.values[to_buses]
    Ri = net.line.r_ohm_per_km.values * net.line.length_km.values
    si = np.ones(len(Pi))
    minf1_terms = (Pi ** 2 + Qi ** 2) / (Ui ** 2) * si * Ri
    minf1_total = np.sum(minf1_terms)

    log("Computation results:")
    print(f"Minimum loss objective function (minf1): {minf1_total:.6f}")
    print(f"Total system losses: {np.sum(net.res_line.pl_mw):.6f} MW")
    print(f"Minimum bus voltage: {np.min(net.res_bus.vm_pu):.6f} p.u.")
    print(f"Number of lines analyzed: {len(Pi)}")

    log("Printing detailed per-line results ...")
    df = pd.DataFrame({
        'Line_From': net.line.from_bus.values + 1,
        'Line_To': net.line.to_bus.values + 1,
        'Pi_MW': Pi,
        'Qi_MVAr': Qi,
        'Ui_pu': Ui,
        'Ri_ohm': Ri,
        'Minf1_Term': minf1_terms
    })
    print(df.round(6).to_string(index=False))
    return Pi, Qi, Ui, Ri, si, minf1_total

def calculate_minf2(net, line_capacity_ka=None):
    """
    Calculate load balancing function minf2:
    minf2 = max(actual line current) - min(actual line current)
    or if branch ratings are known:
    minf2 = max(Ii/INi) - min(Ii/INi)
    If not provided, assumes all line capacities are equal.
    """
    log("Calculating branch currents for minf2 ...")
    Ii = net.res_line.i_from_ka    # Branch current magnitude (from bus side) in kA

    if line_capacity_ka is None:    # If not provided, set all to 1 (per-unit)
        line_capacity_ka = np.ones_like(Ii)

    Ii_pu = Ii / line_capacity_ka

    minf2_raw = np.max(Ii) - np.min(Ii)
    minf2_pu = np.max(Ii_pu) - np.min(Ii_pu)

    log(f"minf2 (raw current difference): {minf2_raw:.6f} kA")
    log(f"minf2 (per-unit current difference): {minf2_pu:.6f}")
    print("Full branch currents (kA):", Ii)
    print("Full branch currents (per unit):", Ii_pu)

    return Ii, minf2_raw, minf2_pu

# Main execution
if __name__ == "__main__":
    log("Loading data ...")
    lines_df = pd.read_csv('Lines_33.csv')
    nodes_df = pd.read_csv('Nodes_33.csv')

    log(f"Lines DataFrame columns: {lines_df.columns.tolist()}")
    log(f"Nodes DataFrame columns: {nodes_df.columns.tolist()}")
    log(f"Lines DataFrame shape: {lines_df.shape}")
    log(f"Nodes DataFrame shape: {nodes_df.shape}")

    net, bus_mapping = create_ieee33_network_from_dataframes(lines_df, nodes_df)
    Pi, Qi, Ui, Ri, si, minf1_total = calculate_pi_qi_ui(net)
    Ii, minf2_raw, minf2_pu = calculate_minf2(net)
