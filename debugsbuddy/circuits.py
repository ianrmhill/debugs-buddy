"""Defines the BOED-compatible linear circuit solver for use in circuit debug."""

import torch as tc
import pyro
import pyro.distributions as dist

from .components import *

__all__ = ['FaultyCircuit']

MAX_RES = 1e6

pu = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")


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

