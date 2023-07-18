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


class Node:
    def __init__(self, type, conns):
        self.type = type
        self.conns = conns

def sim_circuit():
    # Define an RLC series circuit
    w = tc.tensor(100.0)
    r = tc.tensor(10.0)
    l = tc.tensor(0.5)
    c = tc.tensor(0.02)
    z = tc.tensor(0.0)
    n1 = Node('v_in', [1, 0])
    n2 = Node('res', [-(1/r), (1/r) + tc.complex(z, w * c)])
    #n2 = Node('res', [-(1/r), (1/r) + tc.complex(z, w * c), - tc.complex(z, w * c)])
    #n3 = Node('res', [0, -tc.complex(z, w * c), tc.complex(z, w * c) + (1 / tc.complex(z, w * l))])
    v_in = tc.complex(tc.tensor(1.5), tc.tensor(0.0))
    z_c = tc.complex(tc.tensor(0.0), tc.tensor(0.0))
    node_voltages = solve_circuit_complex(v_in, [n1, n2])
    print(node_voltages)


def solve_circuit_complex(v_in, nodes):
    b = tc.zeros(len(nodes), dtype=tc.cfloat)
    a_list = []
    for i, node in enumerate(nodes):
        if node.type == 'v_in':
            b[i] = v_in
            a_row = tc.zeros(len(nodes), dtype=tc.cfloat)
            a_row[i] = 1
            a_list.append(a_row)
        elif node.type == 'res':
            a_row = tc.zeros(len(nodes), dtype=tc.cfloat)
            for j, conn in enumerate(node.conns):
                a_row[j] = conn
            a_list.append(a_row)

    a = tc.stack(a_list)
    print(a)
    print(b)
    return tc.linalg.solve(a, b)

if __name__ == '__main__':
    sim_circuit()
