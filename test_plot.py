import pandapower as pp
import pandapower.plotting as plot

net = pp.create_empty_network()
b1 = pp.create_bus(net, vn_kv=12.66)
b2 = pp.create_bus(net, vn_kv=12.66)
pp.create_line_from_parameters(net, b1, b2, length_km=1,
                               r_ohm_per_km=0.1, x_ohm_per_km=0.2,
                               c_nf_per_km=0.1, max_i_ka=0.2)
plot.simple_plot(net, show_plot=True)
