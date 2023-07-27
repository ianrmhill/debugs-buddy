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
near_zero = tc.tensor(0.00001, device=pu)
shrt_res = tc.tensor(1e4, device=pu)
open_res = tc.tensor(1e-6, device=pu)
op_amp_gain = tc.tensor(1e5, device=pu)


def calc_coeff(type, prm, frequency=near_zero):
    match type:
        case 'r' | 'op-i' | 'op-o-o':
            return 1 / prm
        case 'l':
            return 1 / tc.complex(zero, frequency * prm)
        case 'c':
            return tc.complex(zero, frequency * prm)
        case 'op-o-p':
            return -(op_amp_gain / prm)
        case 'op-o-m':
            return op_amp_gain / prm
        case _:
            raise Exception('Invalid component connection type')


def solve_circuit_complex(source_vals, frequency, nodes, edge_states, prms):
    # The vector 'b' indicates whether any nodes in the circuit have fixed voltages (e.g. source nodes)
    # The corresponding row in the matrix 'a' will be all zeros except 1 for the fixed voltage node coefficient
    batch_dims = source_vals.shape[-1]
    row_dims = (batch_dims, len(nodes)) if type(batch_dims) == int else (*batch_dims, len(nodes))
    b = tc.zeros(row_dims, dtype=tc.cfloat, device=pu)
    a_list = []
    # Node order is fixed, and the source value order must match the order in which the sources appear in the node order
    s = 0
    for i, node in enumerate(nodes):
        a_row = tc.zeros(len(nodes), dtype=tc.cfloat, device=pu)
        if node.type == 'v_in':
            # Set the input values across all batch dimensions then increment the source counter
            b[..., i] = source_vals[..., s]
            s += 1
            # Set the row coefficients to 0 except for 1 in the position corresponding to the node itself
            a_row[..., i] = 1
            a_list.append(a_row)
        elif node.type == 'gnd':
            b[..., i] = tc.zeros(batch_dims, device=pu)
            a_row[..., i] = 1
            a_list.append(a_row)
        else:
            # NOTE: All state information about the circuit should be provided as arguments and coeffs computed here
            for j, conn in enumerate(node.conns):
                # For all nodes except the node itself (matrix diagonal), add the resistance for the connection state
                if i != j:
                    a_row[..., j] -= tc.where(edge_states[..., i, j] == 1, shrt_res, open_res)
                    a_row[..., i] += tc.where(edge_states[..., i, j] == 1, shrt_res, open_res)
                    if conn != '':
                        # Must compute two coefficients as they will be different for an op amp output node
                        a_row[..., j] -= calc_coeff(conn, prms[node.prnt_comp][conn[:4]], frequency)
                        a_row[..., i] += calc_coeff(conn, prms[node.prnt_comp][conn[:4]], frequency)
            a_list.append(a_row)

    # Create the matrix a by stacking all the rows of coefficients from the KCL equations then solve
    a = tc.stack(a_list, -2)
    return tc.linalg.solve(a, tc.transpose(b, -2, -1))
