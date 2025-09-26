import pandas as pd
import networkx as nx
import pandapower.plotting as plot
import pandapower as pp

from network import create_network  # your existing function

# -----------------------------
# Load your CSVs
# -----------------------------
nodes_df = pd.read_csv("Nodes_33.csv")
lines_df = pd.read_csv("Lines_33.csv")

# -----------------------------
# Create network using the original function
# -----------------------------
net, bus_mapping = create_network(lines_df, nodes_df)

# -----------------------------
# Run power flow
# -----------------------------
pp.runpp(net)

# -----------------------------
# Generate bus coordinates automatically
# -----------------------------
G = nx.Graph()
for _, row in lines_df.iterrows():
    if row['STATUS'] == 1:
        G.add_edge(int(row['FROM']), int(row['TO']))

# Use spring_layout for radial-like layout
pos = nx.spring_layout(G, seed=42)

# Assign coordinates to bus_geodata
net.bus_geodata = pd.DataFrame(index=net.bus.index, columns=["x","y"])
for node_num, bus_idx in bus_mapping.items():
    x, y = pos[node_num]
    net.bus_geodata.at[bus_idx, "x"] = x * 10  # scale for better plotting
    net.bus_geodata.at[bus_idx, "y"] = y * 10

print(net.bus.geo)
# print(bus_mapping)

# # -----------------------------
# # Color buses by voltage and lines by loading
# # -----------------------------
# bus_colors = ['red' if vm < 0.95 else 'blue' if vm < 1.05 else 'green' 
#               for vm in net.res_bus.vm_pu]

# line_loading = net.res_line.loading_percent.values  # % loading
# line_colors = ['green' if l < 80 else 'orange' if l < 100 else 'red' 
#                for l in line_loading]

# # -----------------------------
# # Plot network
# # -----------------------------
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(figsize=(15, 10))

# bus_collection = plot.create_bus_collection(net, buses=net.bus.index, size=100, color=bus_colors)
# line_collection = plot.create_line_collection(net, lines=net.line.index, color=line_colors, linewidths=2)

# ax.add_collection(bus_collection)
# ax.add_collection(line_collection)

# ax.autoscale()
# ax.margins(0.1)
# ax.set_aspect('equal')
# plt.show()
# # -----------------------------