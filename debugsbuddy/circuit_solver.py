# Kirschoff's current law (KCL) is the method used to solve for all the voltages in a given circuit. This is easier to
# set up programmatically compared to KVL as we only need a list of nodes versus graphically identifying the closed
# loops within a circuit's set of connections.

# In each nodal equation all currents are assumed to flow out from the given node, and each equation is rearranged to
# be zero on one side with one term per node voltage on the other. With this arrangement we can solve the circuit as a
# set of linear equations using a linear algebra circuit solver in the form Av = B.

# For AC analysis, we simply make this a complex-valued matrix equation. For linear circuits the output frequency will
# always be the same as the input frequency and so the frequency is a given; we only need to solve for amplitude and
# phase at each node.

# This method of solving does require the coefficient matrix A which contains all the impedance terms to be invertible.
# For solving the circuit normally this is not an issue, however the defect modelling method results in numerous shorts
# and opens which can lead to non-invertible matrices A (i.e. more than one solution is possible). To overcome this, the
# shorts and opens are instead represented by extremely low and high resistances to force the circuit to be 'standard'
# from a typical electrical engineering perspective, ensuring the matrix is always invertible at the cost of a small
# error in the resulting solution (smaller than the measurement uncertainty!). This does lead to one additional
# challenge, however, as the circuit ends up covering many orders of magnitude, increasing the risk of instability in
# our probabilistic programming techniques.

import torch as tc

pu = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
zero = tc.tensor(0.0, device=pu)
shrt = tc.tensor(1e4, device=pu)
open = tc.tensor(1e-4, device=pu)


def solve_circuit_complex(source_vals, frequency, nodes, edge_states, prms, shrt_res=shrt, open_res=open):
    # The vector 'b' indicates whether any nodes in the circuit have fixed voltages (e.g. source nodes)
    # The corresponding row in the matrix 'a' will be all zeros except 1 for the fixed voltage node coefficient
    batch_dims = source_vals.shape[:-1]
    row_dims = (batch_dims, len(nodes)) if type(batch_dims) == int else (*batch_dims, len(nodes))

    # Set up the non-linear op amp saturation required data structures
    sat = {}
    opamp_list = []
    for i, node in enumerate(nodes):
        # Ensure we only iterate over the output of each op amp once, and only those with power rails modelled
        if node.name == 'o' and 'op' in node.prnt_comp and node.lims is not None and node.prnt_comp not in opamp_list:
            opamp_list.append(node.prnt_comp)
            sat[node.prnt_comp] =\
                {'n': i, 'min': node.lims[0], 'max': node.lims[1],
                 'is_sat': tc.zeros(batch_dims, device=pu), 'v_fixed': tc.zeros(batch_dims, device=pu)}

    # Set up the linear circuit equations first
    b = tc.zeros(row_dims, dtype=tc.cfloat, device=pu)
    a_list = []
    # Node order is fixed, and the source value order must match that in which the source nodes appear
    s = 0
    for i, n1 in enumerate(nodes):
        a_row = tc.zeros(row_dims, dtype=tc.cfloat, device=pu)
        if n1.name == 'vin':
            # Set the input values across all batch dimensions then increment the source counter
            b[..., i] = source_vals[..., s]
            s += 1
            # Set the row coefficients to 0 except for 1 in the position corresponding to the node itself
            a_row[..., i] = 1
        elif n1.name == 'gnd':
            b[..., i] = tc.zeros(batch_dims, device=pu)
            a_row[..., i] = 1
        else:
            # Construct the node KCL equation
            for j, n2 in enumerate(nodes):
                if i != j:
                    a_row[..., j] -= tc.where(edge_states[..., i, j] == 1, shrt_res, open_res)
                    a_row[..., i] += tc.where(edge_states[..., i, j] == 1, shrt_res, open_res)
                    if n1.prnt_comp == n2.prnt_comp:
                        a_row[..., i] += n1.calc_coeff(n1.name, n1.name, n2.name, prms[n1.prnt_comp], frequency)
                        a_row[..., j] -= n1.calc_coeff(n1.name, n2.name, n1.name, prms[n1.prnt_comp], frequency)
        a_list.append(a_row)
    a = tc.stack(a_list, -2)

    # Solve the system of equations, check for op amp saturation in linear solution, then fix voltages if needed
    # We solve again and keep fixing any new supply violations until all circuits within the batch are within limits
    more_sat = True
    while more_sat:
        for i, n in enumerate(nodes):
            if n.name == 'o' and n.prnt_comp in sat.keys() and tc.count_nonzero(sat[n.prnt_comp]['is_sat'] > 0):
                b[..., i] = tc.where(sat[n.prnt_comp]['is_sat'] == 1, sat[n.prnt_comp]['v_fixed'], zero)
                a[..., i, :] = tc.where(sat[n.prnt_comp]['is_sat'].unsqueeze(-1) == 1, zero, a[..., i, :])
                a[..., i, i] = tc.where(sat[n.prnt_comp]['is_sat'] == 1, tc.tensor(1.0, device=pu), a[..., i, i])

        # Create the matrix a by stacking all the rows of coefficients from the KCL equations then solve
        v = tc.linalg.solve(a, b)

        # Check for op amp saturation in the computed values
        more_sat = False
        for o in sat.values():
            new_sat = tc.where(v[..., o['n']].abs() < o['min'], 1.0, o['is_sat'])
            new_sat = tc.where(v[..., o['n']].abs() > o['max'], 1.0, new_sat)
            # If any of the batch circuits now have saturation we must fix voltages for that circuit and recalculate
            if not tc.allclose(o['is_sat'], new_sat):
                more_sat = True
                o['is_sat'] = new_sat
                o['v_fixed'] = tc.where(
                    v[..., o['n']].abs() < o['min'],
                    tc.complex(o['min'] * tc.cos(v[..., o['n']].angle()), o['min'] * tc.sin(v[..., o['n']].angle())),
                    o['v_fixed'])
                o['v_fixed'] = tc.where(
                    v[..., o['n']].abs() > o['max'],
                    tc.complex(o['max'] * tc.cos(v[..., o['n']].angle()), o['max'] * tc.sin(v[..., o['n']].angle())),
                    o['v_fixed'])

    return v



def solve_circuit(source_vals, nodes, edge_states, prms, shrt_res=shrt, open_res=open):
    # The vector 'b' indicates whether any nodes in the circuit have fixed voltages (e.g. source nodes)
    # The corresponding row in the matrix 'a' will be all zeros except 1 for the fixed voltage node coefficient
    batch_dims = source_vals.shape[:-1]
    row_dims = (batch_dims, len(nodes)) if type(batch_dims) == int else (*batch_dims, len(nodes))

    # Set up the non-linear op amp saturation required data structures
    sat = {}
    opamp_list = []
    for i, node in enumerate(nodes):
        # Ensure we only iterate over the output of each op amp once, and only those with power rails modelled
        if node.name == 'o' and 'op' in node.prnt_comp and node.lims is not None and node.prnt_comp not in opamp_list:
            opamp_list.append(node.prnt_comp)
            sat[node.prnt_comp] =\
                {'n': i, 'min': node.lims[0], 'max': node.lims[1],
                 'is_sat': tc.zeros(batch_dims, device=pu), 'v_fixed': tc.zeros(batch_dims, device=pu)}

    # Set up the linear circuit equations first
    b = tc.zeros(row_dims, dtype=tc.float, device=pu)
    a_list = []
    # Node order is fixed, and the source value order must match that in which the source nodes appear
    s = 0
    for i, n1 in enumerate(nodes):
        a_row = tc.zeros(row_dims, dtype=tc.float, device=pu)
        if n1.name == 'vin':
            # Set the input values across all batch dimensions then increment the source counter
            b[..., i] = source_vals[..., s]
            s += 1
            # Set the row coefficients to 0 except for 1 in the position corresponding to the node itself
            a_row[..., i] = 1
        elif n1.name == 'gnd':
            b[..., i] = tc.zeros(batch_dims, device=pu)
            a_row[..., i] = 1
        else:
            # Construct the node KCL equation
            for j, n2 in enumerate(nodes):
                if i != j:
                    a_row[..., j] -= tc.where(edge_states[..., i, j] == 1, shrt_res, open_res)
                    a_row[..., i] += tc.where(edge_states[..., i, j] == 1, shrt_res, open_res)
                    if n1.prnt_comp == n2.prnt_comp:
                        a_row[..., i] += n1.calc_coeff(n1.name, n1.name, n2.name, prms[n1.prnt_comp], None)
                        a_row[..., j] -= n1.calc_coeff(n1.name, n2.name, n1.name, prms[n1.prnt_comp], None)
        a_list.append(a_row)
    a = tc.stack(a_list, -2)

    # Solve the system of equations, check for op amp saturation in linear solution, then fix voltages if needed
    # We solve again and keep fixing any new supply violations until all circuits within the batch are within limits
    more_sat = True
    while more_sat:
        for i, n in enumerate(nodes):
            if n.name == 'o' and n.prnt_comp in sat.keys() and tc.count_nonzero(sat[n.prnt_comp]['is_sat'] > 0):
                b[..., i] = tc.where(sat[n.prnt_comp]['is_sat'] == 1, sat[n.prnt_comp]['v_fixed'], zero)
                a[..., i, :] = tc.where(sat[n.prnt_comp]['is_sat'].unsqueeze(-1) == 1, zero, a[..., i, :])
                a[..., i, i] = tc.where(sat[n.prnt_comp]['is_sat'] == 1, tc.tensor(1.0, device=pu), a[..., i, i])

        # Create the matrix a by stacking all the rows of coefficients from the KCL equations then solve
        v = tc.linalg.solve(a, b)

        # Check for op amp saturation in the computed values
        more_sat = False
        for o in sat.values():
            new_sat = tc.where(v[..., o['n']] < o['min'], 1.0, o['is_sat'])
            new_sat = tc.where(v[..., o['n']] > o['max'], 1.0, new_sat)
            # If any of the batch circuits now have saturation we must fix voltages for that circuit and recalculate
            if not tc.allclose(o['is_sat'], new_sat):
                more_sat = True
                o['is_sat'] = new_sat
                o['v_fixed'] = tc.where(v[..., o['n']] < o['min'], o['min'], o['v_fixed'])
                o['v_fixed'] = tc.where(v[..., o['n']] > o['max'], o['max'], o['v_fixed'])
    return v
