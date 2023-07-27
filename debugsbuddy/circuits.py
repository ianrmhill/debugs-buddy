"""Defines the BOED-compatible linear circuit solver for use in circuit debug."""

import torch as tc
import pyro
import pyro.distributions as dist

from .components import *
from .circuit_solver import solve_circuit_complex

__all__ = ['FaultyCircuit']

MAX_RES = 1e6

pu = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")


class Node:
    def __init__(self, type: str, parent_component: str, parent_pin: str, hard_conns: list[str], name: str = None):
        self.name = name
        self.type = type
        self.prnt_pin = parent_pin
        self.prnt_comp = parent_component
        self.hard_conns = hard_conns


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

    def node_index_from_pin(self, pin: Pin | Component):
        p = pin if type(pin) == Pin else pin.p1
        for i, node in enumerate(self.nodes):
            if node.prnt_comp == p.prnt_comp and node.prnt_pin == p.name:
                return i
        raise Exception('Pin not found in circuit specification')


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
        for comp in components:
            for pin in components[comp].list_pins():
                ord_nodes.append(Node(pin.type, comp, pin.name, []))
        # Now construct the hard connection matrix which indicates where parameters relate certain nodes
        for i, n1 in enumerate(ord_nodes):
            n1.hard_conns = ['' for _ in range(len(ord_nodes))]
            n1.hard_conns[i] = 'self'
            # Pins have special relationships to other pins on the same component
            for j, n2 in enumerate(ord_nodes):
                if n1 != n2 and n1.prnt_comp == n2.prnt_comp:
                    n1.hard_conns[j] = components[n1.prnt_comp].get_relation(n1.type, n2.type)
        return ord_nodes

    def build_conn_matrix(self, conn_list: list[tuple]):
        c = tc.zeros((len(self.nodes), len(self.nodes)))
        for conn in conn_list:
            i0 = self.node_index_from_pin(conn[0])
            i1 = self.node_index_from_pin(conn[1])
            c[i0, i1] = 1
            c[i1, i0] = 1
        return c

    def simulate(self, input_vals):
        pass

    def get_obsrvd_lbls(self):
        pass

    def get_latent_lbls(self):
        pass

    def gen_fault_model(self, prior_beliefs: dict = None):
        if not prior_beliefs:
            prior_beliefs = self.gen_init_priors()

        def fault_mdl(inputs):
            with pyro.plate_stack('iso-plate', inputs.shape[:-1]):
                # Sample all the parameters from prior belief distributions
                conn_states = tc.zeros()
                for i in range(len(self.nodes)):
                    for j in range(i + 1, len(self.nodes)):
                        conn_states[..., i, j] = pyro.sample(
                            f"e-{i}-{j}", dist.Bernoulli(probs=prior_beliefs[f"e-{i}-{j}"]))
                comp_prms = {}
                for comp in self.comps:
                    comp_prms[comp.name] = {}
                    # TODO: Make work for components with more than one parameter, review whether tc.round needed
                    comp_prms[comp.name][comp.type] = pyro.sample(
                        f"{comp.name}-{comp.type}", dist.Normal(*prior_beliefs[comp.name][comp.type]))

                # Solve the circuit for the sampled values
                v_ins = inputs[..., :-1]
                freqs = inputs[..., -1]
                node_voltages = solve_circuit_complex(v_ins, freqs, self.nodes, conn_states, comp_prms)

                # Assemble the observed voltages for return
                output_list = []
                for obs in self.outputs:
                    i = self.node_index_from_pin(obs)
                    output_list.append(pyro.sample(
                        # TODO: decide on random variance addition logic and make naming work for single-node components
                        f"{obs.prnt_comp}-{obs.name}", dist.Normal(node_voltages[..., i], tc.tensor(0.02, device=pu))))
                outputs = tc.stack(output_list, -1)
                return outputs

        return fault_mdl


class CircuitNode:
    def __init__(self, name, comp, type, all_nodes, connections):
        self.name = name
        self.parent_comp = comp
        self.type = type
        self.nodes = all_nodes
        self.conns = connections

    def get_kcl_eqn(self, mode='sim', comp_prms=None, edge_states=None, batch_shape=None):
        if mode == 'sim':
            return get_eqn(self.name, self.parent_comp, self.type, self.nodes, self.conns, comp_prms)
        elif mode == 'predict':
            return get_pred_eqn(self.name, self.parent_comp, self.type, self.nodes, edge_states, comp_prms, batch_shape)
        else:
            raise Exception(f"Invalid KCL equation construction mode: {mode}")


class FaultyCircuit:
    def __init__(self, components, faulty_conns, intended_conns, prms, meas_nodes, restricted_mode=False):
        self.components = components
        self.nodes = self._construct_nodes(components, faulty_conns)
        self.to_meas = meas_nodes
        self.edges = self.get_edges() if not restricted_mode else self.get_edges(intended_conns)
        self.priors = self._construct_priors(intended_conns, prms)
        self.correct = self._construct_priors(intended_conns, prms, exact=True)
        self.comp_prms = prms

    @staticmethod
    def _construct_nodes(components, conns):
        # First need to determine what nodes will exist to know the length of our connection listings
        ordered = []
        for comp_type in components:
            for comp_name in components[comp_type]:
                ordered.extend(get_comp_nodes(comp_name, comp_type))

        # Now build the node objects with the prior knowledge of which nodes will exist
        nodes = []
        for comp_type in components:
            for comp_name in components[comp_type]:
                if comp_type == 'res':
                    nodes.append(CircuitNode(comp_name + '.1', comp_name, comp_type, ordered, conns))
                    nodes.append(CircuitNode(comp_name + '.2', comp_name, comp_type, ordered, conns))
                elif comp_type == 'opamp5':
                    nodes.append(CircuitNode(comp_name + '.-', comp_name, comp_type, ordered, conns))
                    nodes.append(CircuitNode(comp_name + '.+', comp_name, comp_type, ordered, conns))
                    nodes.append(CircuitNode(comp_name + '.o', comp_name, comp_type, ordered, conns))
                    nodes.append(CircuitNode(comp_name + '.vcc', comp_name, comp_type, ordered, conns))
                    nodes.append(CircuitNode(comp_name + '.vee', comp_name, comp_type, ordered, conns))
                elif comp_type == 'opamp3':
                    nodes.append(CircuitNode(comp_name + '.-', comp_name, comp_type, ordered, conns))
                    nodes.append(CircuitNode(comp_name + '.+', comp_name, comp_type, ordered, conns))
                    nodes.append(CircuitNode(comp_name + '.o', comp_name, comp_type, ordered, conns))
                else:
                    nodes.append(CircuitNode(comp_name, comp_name, comp_type, ordered, conns))
        return nodes

    def _construct_priors(self, expected_conns, prms, exact=False):
        priors = {}
        for prm in prms:
            priors[prm] = tc.tensor([prms[prm], prms[prm] * 0.05], device=pu) if exact else\
                          tc.tensor([prms[prm], prms[prm] * 0.05], device=pu)
        for edge in self.edges:
            edge_name = str(sorted(tuple(edge)))
            if edge in expected_conns:
                priors[edge_name] = tc.tensor(1.0, device=pu) if exact else tc.tensor(0.9, device=pu)
            else:
                priors[edge_name] = tc.tensor(0.0, device=pu) if exact else tc.tensor(0.0, device=pu)
        return priors

    def get_obs_lbls(self):
        obs_lbls = []
        for node in self.nodes:
            if node.type == 'v_out':
                obs_lbls.append(node.name)
        return obs_lbls

    def get_latent_lbls(self):
        ltnt_lbls = []
        #for prm in self.comp_prms:
        #    ltnt_lbls.append(prm)
        for node1 in self.nodes:
            for node2 in self.nodes:
                edge_name = str(sorted(tuple({node1.name, node2.name})))
                if node1.name != node2.name and f"{edge_name}" not in ltnt_lbls:
                    ltnt_lbls.append(f"{edge_name}")
        return ltnt_lbls

    def get_edges(self, conns=None):
        if conns is not None:
            edges = conns
        else:
            edges = []
            for node1 in self.nodes:
                for node2 in self.nodes:
                    if node1.name != node2.name and {node1.name, node2.name} not in edges:
                        if not (node1.type == 'v_in' and node2.type == 'v_in'):
                            edges.append({node1.name, node2.name})
        return edges

    def kcl_solver(self, v_ins, forced_nodes=None):
        # Default to no enforced node voltage overrides
        if not forced_nodes:
            forced_nodes = {}
        # For our KCL equations we will always accumulate all terms on one side of the equality, thus B is a 0 vector
        # except for at fixed input voltage nodes
        b = tc.zeros(len(self.nodes))
        j = 0
        for i, node in enumerate(self.nodes):
            if node.type == 'v_in':
                b[i] = v_ins[j]
                j += 1
            elif node.name in forced_nodes:
                b[i] = forced_nodes[node.name]
        # Check that the supplied number of input voltages was correct
        if len(v_ins) != j:
            raise Exception(f"Incorrect number of input voltages provided. Provided: {len(v_ins)}. Needed: {j}")

        # Build up set of KCL equations for all the terminal nodes
        a_list = []
        for i, node in enumerate(self.nodes):
            # First check that the node voltage isn't being forced to override
            if node.type == 'v_in' or node.name in forced_nodes:
                eqn = tc.zeros(len(self.nodes), dtype=tc.float)
                eqn[i] = 1
                a_list.append(eqn)
            else:
                if node.type == 'res':
                    prms = self.comp_prms[f"{node.parent_comp}-r"]
                elif node.type == 'opamp3' or node.type == 'opamp5':
                    prms = [self.comp_prms[f"{node.parent_comp}-g"],
                            self.comp_prms[f"{node.parent_comp}-ri"], self.comp_prms[f"{node.parent_comp}-ro"]]
                else:
                    prms = None
                a_list.append(node.get_kcl_eqn('sim', prms))
        a = tc.stack(a_list)

        # Now solve the system of equations
        v = tc.linalg.solve(a, b)
        return v

    def simulate_test(self, v_ins):
        # Solve the linear circuit given the voltage inputs
        v = self.kcl_solver(v_ins)

        # Non-linear circuit effect handling!!! For now just handling op amp power rail saturation
        for i, node in enumerate(self.nodes):
            if node.type == 'opamp5' and '.o' in node.name:
                v_min, v_max = None, None
                for j, node2 in enumerate(self.nodes):
                    if node2.parent_comp == node.parent_comp:
                        if '.vcc' in node2.name:
                            v_max = v[j]
                        elif '.vee' in node2.name:
                            v_min = v[j]
                if v_min is not None and v[i] < v_min:
                    v = self.kcl_solver(v_ins, {node.name: v_min})
                elif v_max is not None and v[i] > v_max:
                    v = self.kcl_solver(v_ins, {node.name: v_max})

        # Finally, return the observed output voltages
        out_list = []
        for i, node in enumerate(self.nodes):
            if node.name in self.to_meas:
                out_list.append(v[i])
        return tc.tensor(out_list)

    def inf_kcl_solver(self, v_ins, prms, shorts, is_forced=None, forced_vals=None):
        # Default to no enforced node voltage overrides
        if not is_forced:
            is_forced = {}
        if not forced_vals:
            forced_vals = {}
        # Setup fixed voltage vector
        b = tc.zeros((*v_ins.shape[:-1], len(self.nodes)), device=pu)
        j = 0
        for i, node in enumerate(self.nodes):
            if node.type == 'v_in':
                b[..., i] = v_ins[..., j]
                j += 1
            elif node.name in is_forced:
                b[..., i] = tc.where(is_forced[node.name] == 1, forced_vals[node.name], tc.tensor(0, device=pu))

        # Setup KCL node voltage equations
        a_list = []
        for i, node in enumerate(self.nodes):
            if node.type == 'v_in':
                eqn = tc.zeros((*v_ins.shape[:-1], len(self.nodes)), dtype=tc.float, device=pu)
                eqn[..., i] = 1
                a_list.append(eqn)
            else:
                if node.type == 'res':
                    kcl_prms = prms[f"{node.parent_comp}-r"]
                elif node.type == 'opamp3' or node.type == 'opamp5':
                    kcl_prms = [prms[f"{node.parent_comp}-g"],
                                prms[f"{node.parent_comp}-ri"], prms[f"{node.parent_comp}-ro"]]
                else:
                    kcl_prms = None
                eqn = node.get_kcl_eqn('predict', kcl_prms, shorts, v_ins.shape[:-1])
                # Now override the kcl equation if the value needs to be forced instead
                if node.name in is_forced:
                    for j, node2 in enumerate(self.nodes):
                        if node2.name == node.name:
                            eqn[..., j] = tc.where(is_forced[node.name] == 1, tc.tensor(1, device=pu), eqn[..., j])
                        else:
                            eqn[..., j] = tc.where(is_forced[node.name] == 1, tc.tensor(0, device=pu), eqn[..., j])
                a_list.append(eqn)
        a = tc.stack(a_list, -2)

        # Solve the system of equations to get the node voltages
        v = tc.linalg.solve(a, b)
        #index = (1000, 5)
        #print(a[index[0], index[1], :, :])
        #print(b[index[0], index[1], :])
        #print(v[index[0], index[1], :])
        return v

    def gen_fault_mdl(self, beliefs=None):
        # If no beliefs are provided it means we are using the initial intended connections for our priors
        if not beliefs:
            beliefs = self.priors

        def fault_mdl(test_ins):
            with pyro.plate_stack('iso-plate', test_ins.shape[:-1]):
                # Sample all our latent parameters
                prms = {}
                for comp in self.comp_prms:
                    prms[comp] = tc.round(pyro.sample(comp, dist.Normal(*beliefs[comp])))
                shorts = {}
                for edge in self.edges:
                    edge_name = str(sorted(tuple(edge)))
                    shorts[edge_name] = pyro.sample(
                        f"{edge_name}", dist.Bernoulli(probs=beliefs[edge_name]))

                v = self.inf_kcl_solver(test_ins, prms, shorts)

                # Non-linear circuit effect handling!!! For now just handling op amp power rail saturation
                is_forced = {}
                forced_vals = {}
                for i, node in enumerate(self.nodes):
                    if node.type == 'opamp5' and '.o' in node.name:
                        v_min, v_max = None, None
                        for j, node2 in enumerate(self.nodes):
                            if node2.parent_comp == node.parent_comp:
                                if '.vcc' in node2.name:
                                    v_max = v[..., j]
                                elif '.vee' in node2.name:
                                    v_min = v[..., j]
                        if v_min is not None or v_max is not None:
                            # The is_forced tensor masks which individual samples in the batch have forced values
                            # The forced_vals tensor provides those forced values
                            ones = tc.ones(test_ins.shape[:-1], dtype=tc.float, device=pu)
                            is_forced[node.name] = tc.zeros(test_ins.shape[:-1], dtype=tc.float, device=pu)
                            forced_vals[node.name] = tc.zeros(test_ins.shape[:-1], dtype=tc.float, device=pu)
                            if v_min is not None:
                                vmin_forced = tc.where(v[..., i] < v_min, ones, is_forced[node.name])
                                forced_vals[node.name] = tc.where(vmin_forced == 1, v_min, forced_vals[node.name])
                            if v_max is not None:
                                vmax_forced = tc.where(v[..., i] < v_max, ones, is_forced[node.name])
                                forced_vals[node.name] = tc.where(vmax_forced == 1, v_max, forced_vals[node.name])
                            is_forced[node.name] = tc.where(vmin_forced == 1, ones, is_forced[node.name])
                            is_forced[node.name] = tc.where(vmax_forced == 1, ones, is_forced[node.name])
                            v = self.inf_kcl_solver(test_ins, prms, shorts, is_forced, forced_vals)

                # Only return the measured node voltages
                out_list = []
                for i, node in enumerate(self.nodes):
                    if node.name in self.to_meas:
                        out_list.append(pyro.sample(f"{node.name}", dist.Normal(v[..., i], tc.tensor(0.02, device=pu))))
                outs = tc.stack(out_list, -1)
                return outs

        return fault_mdl

