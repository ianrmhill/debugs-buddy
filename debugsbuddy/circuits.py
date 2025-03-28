"""Defines the BOED-compatible circuit definitions for use in circuit debug."""

from typing import Callable
import torch as tc
import pyro
import pyro.distributions as dist

from .components import *
from .circuit_solver import solve_circuit_complex, solve_circuit

__all__ = ['Circuit']

MAX_RES = 1e6

pu = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")


class Node:
    def __init__(self, name: str, parent_component: str, coeff_func: Callable):
        self.name = name
        self.prnt_comp = parent_component
        self.calc_coeff = coeff_func
        self.lims = None


class Circuit:
    def __init__(self, components: list | dict, intended_connections, outputs, name: str = None):
        self.name = name
        if type(components) == list:
            self.comps = self.set_comp_names(components)
        else:
            self.comps = components
        self.nodes = self.build_nodes(self.comps)
        self.outputs = outputs
        self.intended_conns = self.build_conn_matrix(intended_connections)
        self.intended_prms = None
        self.actual_conns = None
        self.actual_prms = None
        self.ignore_faults = self.find_fake_faults(intended_connections)

    def set_actual_circuit(self, actual_connections, actual_parameters):
        self.actual_conns = self.build_conn_matrix(actual_connections)
        self.actual_prms = {}
        for comp in self.comps.values():
            if 'prms' in dir(comp):
                self.actual_prms[comp.name] = {}
                for prm in comp.prms:
                    self.actual_prms[comp.name][prm] = tc.tensor(actual_parameters[comp.name][prm], device=pu)

    def node_index_from_pin(self, pin: Pin | Component):
        p = pin if type(pin) == Pin else pin.p1
        for i, node in enumerate(self.nodes):
            if node.prnt_comp == p.prnt_comp and node.name == p.name:
                return i
        raise Exception('Pin not found in circuit specification')

    def nodal_name_from_index(self, index: int):
        for i, node in enumerate(self.nodes):
            if i == index:
                return f"{node.prnt_comp}-{node.name}"

    @staticmethod
    def set_comp_names(components: list):
        as_dict = {}
        for comp in components:
            # If the user explicitly gave the component a name use that
            if comp.name is not None:
                name = comp.name
            else:
                # Otherwise we count how many of this type of component have already been seen
                tally = 0
                for named in as_dict:
                    if as_dict[named].type == comp.type:
                        tally += 1
                # Essentially this is just sequential annotation
                name = comp.type + str(tally + 1)
            if name in as_dict:
                raise Exception('Duplicate component naming occurred when setting up circuit. Please fix.')
            # Now set the determined name in the dict and update the component names
            as_dict[name] = comp
            as_dict[name].name = name
            for pin in as_dict[name].list_pins():
                pin.prnt_comp = name
        return as_dict

    @staticmethod
    def build_nodes(components: dict):
        ord_nodes = []
        # First create a node for every pin on every component
        for comp in components.values():
            for pin in comp.list_pins():
                ord_nodes.append(Node(pin.name, comp.name, comp.get_coeff))
                if pin.name == 'o':
                    ord_nodes[-1].lims = pin.lims
        return ord_nodes

    def build_conn_matrix(self, conn_list: list[tuple]):
        c = tc.zeros((len(self.nodes), len(self.nodes)), device=pu)
        for conn in conn_list:
            i0 = self.node_index_from_pin(conn[0])
            i1 = self.node_index_from_pin(conn[1])
            c[i0, i1] = 1
            c[i1, i0] = 1
        return c

    def find_fake_faults(self, intended_conns):
        fakes = []
        srcs = []
        for i, node in enumerate(self.nodes):
            if 'vin' in node.prnt_comp or 'gnd' in node.prnt_comp:
                srcs.append(i)
        srcs.sort()
        # Mark short faults between sources which can't happen in practice without equipment failing or fires
        for i, src in enumerate(srcs):
            for j in range(i+1, len(srcs)):
                fakes.append(f"e-{srcs[i]}-{srcs[j]}")

        conn_counts = {node: [] for node in self.nodes}
        for conn in intended_conns:
            n1 = self.node_index_from_pin(conn[0] if type(conn[0]) == Pin else conn[0].p1)
            n2 = self.node_index_from_pin(conn[1] if type(conn[1]) == Pin else conn[1].p1)
            conn_counts[self.nodes[n1]].append(n2)
            conn_counts[self.nodes[n2]].append(n1)
        # Mark triangle connections where 3 or more nodes are connected, leading to intended short faults
        for node in conn_counts:
            conn_counts[node].sort()
            if len(conn_counts[node]) == 2:
                fakes.append(f"e-{conn_counts[node][0]}-{conn_counts[node][1]}")
            elif len(conn_counts[node]) == 3:
                fakes.append(f"e-{conn_counts[node][0]}-{conn_counts[node][1]}")
                fakes.append(f"e-{conn_counts[node][0]}-{conn_counts[node][2]}")
                fakes.append(f"e-{conn_counts[node][1]}-{conn_counts[node][2]}")
            elif len(conn_counts[node]) > 3:
                pass
                #raise NotImplementedError('Not yet supporting circuit nodes with five or more intended connections, sorry!')

        return fakes

    def simulate_actual(self, input_vals, complex=True):
        if complex:
            v = solve_circuit_complex(input_vals[..., :-1], input_vals[..., -1], self.nodes,
                                      self.actual_conns, self.actual_prms)
        else:
            v = solve_circuit(input_vals, self.nodes, self.actual_conns, self.actual_prms)
        # Assemble the observed voltages for return
        output_list = []
        for obs in self.outputs:
            i = self.node_index_from_pin(obs)
            if complex:
                output_list.append(v[..., i].abs())
                output_list.append(v[..., i].angle())
            else:
                output_list.append(v[..., i])
        outputs = tc.stack(output_list, -1)
        return outputs

    def simulate_intended(self, input_vals):
        v = solve_circuit_complex(input_vals[..., :-1], input_vals[..., -1], self.nodes,
                                  self.intended_conns, self.intended_prms)
        # Assemble the observed voltages for return
        output_list = []
        for obs in self.outputs:
            i = self.node_index_from_pin(obs)
            output_list.append(v[..., i].abs())
            output_list.append(v[..., i].angle())
        outputs = tc.stack(output_list, -1)
        return outputs

    def get_obsrvd_lbls(self, complex=True):
        obsrvd_list = []
        for obs in self.outputs:
            if complex:
                obsrvd_list.append(f"{obs.prnt_comp}-{obs.name}-ampli")
                obsrvd_list.append(f"{obs.prnt_comp}-{obs.name}-phase")
            else:
                obsrvd_list.append(f"{obs.prnt_comp}-{obs.name}")
        return obsrvd_list

    def get_latent_lbls(self):
        ltnt_list = []
        for i in range(len(self.nodes)):
            for j in range(i+1, len(self.nodes)):
                ltnt_list.append(f"e-{i}-{j}")
        for comp in self.comps.values():
            if 'prms' in dir(comp):
                for prm in comp.prms:
                    ltnt_list.append(f"{comp.name}-{prm}")
        return ltnt_list

    def gen_init_priors(self, open_fault_prob, shrt_fault_prob, comp_prm_spread):
        priors = {}
        for i in range(len(self.nodes)):
            for j in range(i+1, len(self.nodes)):
                if self.intended_conns[i][j] == 1:
                    priors[f"e-{i}-{j}"] = tc.tensor(1.0 - open_fault_prob, device=pu)
                else:
                    priors[f"e-{i}-{j}"] = tc.tensor(shrt_fault_prob, device=pu)
        for comp in self.comps.values():
            if 'prms' in dir(comp):
                priors[comp.name] = {}
                for prm in comp.prms.keys():
                    priors[comp.name][prm] = tc.tensor([comp.prms[prm], comp.prms[prm] * comp_prm_spread], device=pu)
        return priors

    def gen_fault_mdl(self, prior_beliefs: dict = None,
                      shrt_res=tc.tensor(1e3, device=pu), open_res=tc.tensor(1e-3, device=pu),
                      open_fault_prob=0.1, shrt_fault_prob=0.05, comp_prm_spread=0.2, meas_error=0.002, complex=True):
        if not prior_beliefs:
            prior_beliefs = self.gen_init_priors(open_fault_prob, shrt_fault_prob, comp_prm_spread)
        # Save the current set of beliefs so we can compare them to sets after inference
        self.curr_beliefs = prior_beliefs

        def fault_mdl(inputs):
            with pyro.plate_stack('iso-plate', inputs.shape[:-1]):
                # Sample all the parameters from prior belief distributions
                batch_dims = inputs.shape[:-1]
                num_nodes = len(self.nodes)
                row_dims = (batch_dims, num_nodes, num_nodes)\
                    if type(batch_dims) == int else (*batch_dims, num_nodes, num_nodes)
                conn_states = tc.zeros(row_dims, device=pu)
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        edge_state = pyro.sample(f"e-{i}-{j}", dist.Bernoulli(probs=prior_beliefs[f"e-{i}-{j}"]))
                        conn_states[..., i, j] = edge_state
                        conn_states[..., j, i] = edge_state
                comp_prms = {}
                for comp in self.comps.values():
                    # Only need to sample values for components that have parameters
                    if 'prms' in dir(comp):
                        comp_prms[comp.name] = {}
                        for prm in comp.prms:
                            #deviation = pyro.sample(f"{comp.name}-{prm}-n", dist.Normal(*prior_beliefs[comp.name][prm]['n']))
                            #wide = pyro.sample(f"{comp.name}-{prm}-u", dist.Uniform(*prior_beliefs[comp.name][prm]['u']))
                            #slct = pyro.sample(f"{comp.name}-{prm}-s", dist.Bernoulli(0.7), infer={'enumerate': 'parallel'})
                            #comp_prms[comp.name][prm] = pyro.sample(
                            #    f"{comp.name}-{prm}", dist.MaskedMixture(slct == 0, deviation, wide))
                            comp_prms[comp.name][prm] = pyro.sample(
                                f"{comp.name}-{prm}", dist.Normal(*prior_beliefs[comp.name][prm]))

                # Solve the circuit for the sampled values
                if complex:
                    v_ins = inputs[..., :-1]
                    freqs = inputs[..., -1]
                    node_voltages = solve_circuit_complex(v_ins, freqs, self.nodes, conn_states, comp_prms,
                                                      shrt_res, open_res)
                else:
                    node_voltages = solve_circuit(inputs, self.nodes, conn_states, comp_prms, shrt_res, open_res)

                # Assemble the observed voltages for return
                output_list = []
                for obs in self.outputs:
                    i = self.node_index_from_pin(obs)
                    if complex:
                        output_list.append(pyro.sample(
                            f"{obs.prnt_comp}-{obs.name}-ampli",
                            dist.Normal(node_voltages[..., i].abs(), tc.tensor(meas_error, device=pu))))
                        output_list.append(pyro.sample(
                            f"{obs.prnt_comp}-{obs.name}-phase",
                            dist.Normal(node_voltages[..., i].angle(), tc.tensor(meas_error, device=pu))))
                    else:
                        output_list.append(pyro.sample(
                            f"{obs.prnt_comp}-{obs.name}",
                            dist.Normal(node_voltages[..., i], tc.tensor(meas_error, device=pu))))
                outputs = tc.stack(output_list, -1)
                return outputs

        return fault_mdl
