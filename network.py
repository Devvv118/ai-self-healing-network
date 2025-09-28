# import necessary libraries
import numpy as np
import pandas as pd
import pandapower as pp

from functions import calculate_minf1, calculate_minf2

def create_network(lines_df, nodes_df):
    net = pp.create_empty_network(name="IEEE 33-bus")
    base_voltage_kv = 12.66
    bus_mapping = {}
    for _, row in nodes_df.iterrows():
        node_num = int(row['NODES'])
        bus_idx = pp.create_bus(net, vn_kv=base_voltage_kv, name=f"Bus_{node_num}")
        bus_mapping[node_num] = bus_idx

    # Adding external grid at slack bus ...
    slack_bus = bus_mapping[1]
    pp.create_ext_grid(net, slack_bus, vm_pu=1.0, va_degree=0.0)

    # Creating loads ...
    for _, row in nodes_df.iterrows():
        node_num = int(row['NODES'])
        if node_num != 1:
            pd_mw = row['PD'] / 1000.0
            qd_mvar = row['QD'] / 1000.0
            if pd_mw > 0 or qd_mvar > 0:
                pp.create_load(net, bus_mapping[node_num],
                               p_mw=pd_mw, q_mvar=qd_mvar,
                               name=f"Load_{node_num}")

    # Building lines between buses ...
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

    return net, bus_mapping

def calculate_pi_qi_ui(net):
    try:
        pp.runpp(net, verbose=False)
    except Exception as e:
        return None, None, None, None, None, None

    Pi = net.res_line.p_from_mw.values
    Qi = net.res_line.q_from_mvar.values
    to_buses = net.line.to_bus.values
    Ui = net.res_bus.vm_pu.values[to_buses]
    Ri = net.line.r_ohm_per_km.values * net.line.length_km.values
    Xi = net.line.x_ohm_per_km.values * net.line.length_km.values
    si = np.ones(len(Pi))

    net_res_df = pd.DataFrame({
        'Line_From': net.line.from_bus.values + 1,
        'Line_To': net.line.to_bus.values + 1,
        'Pi_MW': Pi,
        'Qi_MVAr': Qi,
        'Ui_pu': Ui,
        'Ri_ohm': Ri,
        'Xi_ohm': Xi
    })
    return Pi, Qi, Ui, Ri, Xi, si, net_res_df

def calculate_Ii_INi(net, line_capacity_ka=None):
    """
    Consider INi = 1
    """
    Ii = net.res_line.i_from_ka    # Branch current magnitude (from bus side) in kA

    if line_capacity_ka is None:    # If not provided, set all to 1 (per-unit)
        INi = np.ones_like(Ii)
    else:
        INi = line_capacity_ka

    return Ii, INi


if __name__ == "__main__":
    lines_df = pd.read_csv('Lines_33.csv')
    nodes_df = pd.read_csv('Nodes_33.csv')
    net, bus_mapping = create_network(lines_df, nodes_df)
    Pi, Qi, Ui, Ri, Xi, si, net_res_df = calculate_pi_qi_ui(net)
    minf1 = calculate_minf1(Pi, Qi, Ui, Ri, si)
    Ii, INi = calculate_Ii_INi(net, line_capacity_ka=None)
    minf2 = calculate_minf2(Ii, INi)
    print("-" * 30)
    print("minf1:", minf1)
    print("minf2:", minf2)
    print("-" * 30)