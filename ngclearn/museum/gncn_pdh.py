import os
import sys
import copy
#from config import Config
import tensorflow as tf
import numpy as np

from ngclearn.engine.nodes.snode import SNode
from ngclearn.engine.nodes.enode import ENode
from ngclearn.engine.cables.dcable import DCable
from ngclearn.engine.ngc_graph import NGCGraph

from ngclearn.engine.nodes.fnode import FNode
from ngclearn.engine.proj_graph import ProjectionGraph

from ngclearn.utils.io_utils import parse_simulation_info

class GNCN_PDH:
    """
    Structure for constructing the model proposed in:

    Ororbia, A., and Kifer, D. The neural coding framework for learning
    generative models. Nature Communications 13, 2064 (2022).

    This model, under the NGC computational framework, is referred to as
    the GNCN-PDH, according to the naming convention in
    (Ororbia & Kifer 2022).

    | Historical Note:
    | (The arXiv paper that preceded the publication above is shown below:)
    | Ororbia, Alexander, and Daniel Kifer. "The neural coding framework for
    | learning generative models." arXiv preprint arXiv:2012.03405 (2020).

    | Node Name Structure:
    | z3 -(z3-mu2)-> mu2 ;e2; z2 -(z2-mu1)-> mu1 ;e1; z1 -(z1-mu0-)-> mu0 ;e0; z0
    | z3 -(z3-mu1)-> mu1; z2 -(z2-mu0)-> mu0
    | e2 -> e2 * Sigma2; e1 -> e1 * Sigma1  // Precision weighting
    | z3 -> z3 * Lat3;  z2 -> z2 * Lat2;  z1 -> z1 * Lat1 // Lateral competition
    | e2 -(e2-z3)-> z3; e1 -(e1-z2)-> z2; e0 -(e0-z1)-> z1  // Error feedback

    Args:
        args: a Config dictionary containing necessary meta-parameters for the GNCN-PDH

    | DEFINITION NOTE:
    | args should contain values for the following:
    | * batch_size - the fixed batch-size to be fed into this model
    | * z_top_dim: # of latent variables in layer z3 (top-most layer)
    | * z_dim: # of latent variables in layers z1 and z2
    | * x_dim: # of latent variables in layer z0 or sensory x
    | * seed: number to control determinism of weight initialization
    | * wght_sd: standard deviation of Gaussian initialization of weights
    | * beta: latent state update factor
    | * leak: strength of the leak variable in the latent states
    | * K: # of steps to take when conducting iterative inference/settling
    | * act_fx: activation function for layers z1, z2, and z3
    | * out_fx: activation function for layer mu0 (prediction of z0) (Default: sigmoid)
    | * n_group: number of neurons w/in a competition group for z2 and z2 (sizes of z2
        and z1 should be divisible by this number)
    | * n_top_group: number of neurons w/in a competition group for z3 (size of z3
        should be divisible by this number)
    | * alpha_scale: the strength of self-excitation
    | * beta_scale: the strength of cross-inhibition
    """
    def __init__(self, args):
        self.args = args

        batch_size = int(self.args.getArg("batch_size"))
        z_top_dim = int(self.args.getArg("z_top_dim"))
        z_dim = int(self.args.getArg("z_dim"))
        x_dim = int(self.args.getArg("x_dim"))

        seed = int(self.args.getArg("seed")) #69
        beta = float(self.args.getArg("beta"))
        K = int(self.args.getArg("K"))
        act_fx = self.args.getArg("act_fx") #"tanh"
        out_fx = "sigmoid"
        if self.args.hasArg("out_fx") == True:
            out_fx = self.args.getArg("out_fx")
        leak = float(self.args.getArg("leak")) #0.0

        wght_sd = float(self.args.getArg("wght_sd"))


        self.batch_size = batch_size
        self.z_top_dim = z_top_dim
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.seed = seed
        self.beta = beta
        self.K = K
        self.leak = leak
        self.wght_sd = wght_sd

        integrate_cfg = {"integrate_type" : "euler"}
        precis_cfg = ("uniform", 0.01)
        constraint_cfg = {"clip_type":"norm_clip","clip_mag":1.0,"clip_axis":0}

        z3 = SNode(name="z3", dim=z_top_dim, beta=beta, leak=leak, act_fx=act_fx,
                   integrate_kernel=integrate_cfg)
        z2 = SNode(name="z2", dim=z_dim, beta=beta, leak=leak, act_fx=act_fx,
                   integrate_kernel=integrate_cfg)
        z1 = SNode(name="z1", dim=z_dim, beta=beta, leak=leak, act_fx=act_fx,
                   integrate_kernel=integrate_cfg)
        z0 = SNode(name="z0", dim=x_dim, beta=beta, integrate_kernel=integrate_cfg, leak=0.0)

        mu2 = SNode(name="mu2", dim=z_dim, act_fx="relu", zeta=0.0)
        mu1 = SNode(name="mu1", dim=z_dim, act_fx="relu", zeta=0.0)
        mu0 = SNode(name="mu0", dim=x_dim, act_fx=out_fx, zeta=0.0)

        e2 = ENode(name="e2", dim=z_dim)
        e1 = ENode(name="e1", dim=z_dim)
        e0 = ENode(name="e0", dim=x_dim)


        # create cable wiring scheme relating nodes to one another
        init_kernels = {"A_init" : ("gaussian",wght_sd)}
        dcable_cfg = {"type": "dense", "init_kernels" : init_kernels, "seed" : seed}
        ecable_cfg = {"type": "dense", "init_kernels" : init_kernels, "seed" : seed}


        z3_mu2 = z3.wire_to(mu2, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg,
                            short_name="W3")
        z2_mu1 = z2.wire_to(mu1, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg,
                            short_name="W2")
        z1_mu0 = z1.wire_to(mu0, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg,
                            short_name="W1")
        # z3_mu1 = z3.wire_to(mu1, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg,
        #                     short_name="S3")
        # z2_mu0 = z2.wire_to(mu0, src_comp="phi(z)", dest_comp="dz_td", cable_kernel=dcable_cfg,
        #                     short_name="S2")

        z3_mu2.set_constraint(constraint_cfg)
        z2_mu1.set_constraint(constraint_cfg)
        z1_mu0.set_constraint(constraint_cfg)
        # z3_mu1.set_constraint(constraint_cfg)
        # z2_mu0.set_constraint(constraint_cfg)


        e2_z3 = e2.wire_to(z3, src_comp="phi(z)", dest_comp="dz_bu", cable_kernel=ecable_cfg,
                           short_name="E3")
        e1_z2 = e1.wire_to(z2, src_comp="phi(z)", dest_comp="dz_bu", cable_kernel=ecable_cfg,
                           short_name="E2")
        e0_z1 = e0.wire_to(z1, src_comp="phi(z)", dest_comp="dz_bu", cable_kernel=ecable_cfg,
                           short_name="E1")
        e2_z3.set_constraint(constraint_cfg)
        e1_z2.set_constraint(constraint_cfg)
        e0_z1.set_constraint(constraint_cfg)




        # set up update rules and make relevant edges aware of these
        # z3_mu1.set_update_rule(preact=(z3,"phi(z)"), postact=(e1,"phi(z)"), param=["A"])
        # z2_mu0.set_update_rule(preact=(z2,"phi(z)"), postact=(e0,"phi(z)"), param=["A"])

        z3_mu2.set_update_rule(preact=(z3,"phi(z)"), postact=(e2,"phi(z)"), param=["A"])
        z2_mu1.set_update_rule(preact=(z2,"phi(z)"), postact=(e1,"phi(z)"), param=["A"])
        z1_mu0.set_update_rule(preact=(z1,"phi(z)"), postact=(e0,"phi(z)"), param=["A"])
        e_gamma = 1.0
        e2_z3.set_update_rule(preact=(e2,"phi(z)"), postact=(z3,"phi(z)"), gamma=e_gamma, param=["A"])
        e1_z2.set_update_rule(preact=(e1,"phi(z)"), postact=(z2,"phi(z)"), gamma=e_gamma, param=["A"])
        e0_z1.set_update_rule(preact=(e0,"phi(z)"), postact=(z1,"phi(z)"), gamma=e_gamma, param=["A"])

        # Set up graph - execution cycle/order
        print(" > Constructing NGC graph")
        ngc_model = NGCGraph(K=K, name="gncn_pdh")
        ngc_model.set_cycle(nodes=[z3, z2, z1, z0])
        ngc_model.set_cycle(nodes=[mu2, mu1, mu0])
        ngc_model.set_cycle(nodes=[e2, e1, e0])
        print(f"{[th.name for th in ngc_model.theta]}")
        info = ngc_model.compile(batch_size=batch_size)
        self.info = parse_simulation_info(info)
        # ngc_model.apply_constraints()
        self.ngc_model = ngc_model
        self.clip_weights()

        # build this NGC model's sampling graph
        z3_dim = ngc_model.getNode("z3").dim
        z2_dim = ngc_model.getNode("z2").dim
        z1_dim = ngc_model.getNode("z1").dim
        z0_dim = ngc_model.getNode("z0").dim
        # Set up complementary sampling graph to use in conjunction w/ NGC-graph
        s3 = FNode(name="s3", dim=z3_dim, act_fx=act_fx)
        s2 = FNode(name="s2", dim=z2_dim, act_fx=act_fx)
        s1 = FNode(name="s1", dim=z1_dim, act_fx=act_fx)
        s0 = FNode(name="s0", dim=z0_dim, act_fx=out_fx)
        s3_s2 = s3.wire_to(s2, src_comp="phi(z)", dest_comp="dz", mirror_path_kernel=(z3_mu2,"A"))
        s2_s1 = s2.wire_to(s1, src_comp="phi(z)", dest_comp="dz", mirror_path_kernel=(z2_mu1,"A"))
        # s3_s1 = s3.wire_to(s1, src_comp="phi(z)", dest_comp="dz", mirror_path_kernel=(z3_mu1,"A"))
        s1_s0 = s1.wire_to(s0, src_comp="phi(z)", dest_comp="dz", mirror_path_kernel=(z1_mu0,"A"))
        # s2_s0 = s2.wire_to(s0, src_comp="phi(z)", dest_comp="dz", mirror_path_kernel=(z2_mu0,"A"))
        sampler = ProjectionGraph()
        sampler.set_cycle(nodes=[s3, s2, s1, s0])
        sampler_info = sampler.compile()
        self.sampler_info = parse_simulation_info(sampler_info)
        self.ngc_sampler = sampler

        self.delta = None

    def project(self, z_sample):
        """
        Run projection scheme to get a sample of the underlying directed
        generative model given the clamped variable *z_sample*

        Args:
            z_sample: the input noise sample to project through the NGC graph

        Returns:
            x_sample (sample(s) of the underlying generative model)
        """
        readouts = self.ngc_sampler.project(
                        clamped_vars=[("s3","z",tf.cast(z_sample,dtype=tf.float32))],
                        readout_vars=[("s0","phi(z)")]
                    )
        x_sample = readouts[0][2]
        return x_sample

    def settle(self, x, calc_update=True):
        """
        Run an iterative settling process to find latent states given clamped
        input and output variables

        Args:
            x: sensory input to reconstruct/predict

            calc_update: if True, computes synaptic updates @ end of settling
                process (Default = True)

        Returns:
            x_hat (predicted x)
        """
        readouts, delta = self.ngc_model.settle(
                            clamped_vars=[("z0","z", x)],
                            readout_vars=[("mu0","phi(z)"),("mu1","phi(z)"),("mu2","phi(z)")],
                            calc_delta=calc_update
                          )

        self.delta = delta # store delta to constructor for later retrieval
        x_hat = readouts[0][2]
        return x_hat

    def settle2(self, x, calc_update=True):
        """
        Run an iterative settling process to find latent states given clamped
        input and output variables

        Args:
            x: sensory input to reconstruct/predict

            calc_update: if True, computes synaptic updates @ end of settling
                process (Default = True)

        Returns:
            x_hat (predicted x)
        """

        batch_size = x.shape[0]

        z3_node = self.ngc_model.nodes['z3']
        z2_node = self.ngc_model.nodes['z2']
        z1_node = self.ngc_model.nodes['z1']
        z0_node = self.ngc_model.nodes['z0']

        # clamp
        z0_node.compartments["z"] = x


        # Initialize the values of every non-clamped node

        z3_node.compartments["z"] = tf.zeros([batch_size, self.z_top_dim])
        z2_node.compartments["z"] = tf.zeros([batch_size, self.z_dim])
        z1_node.compartments["z"] = tf.zeros([batch_size, self.z_dim])


        e2 = tf.zeros([batch_size, self.z_dim])
        e1 = tf.zeros([batch_size, self.z_dim])
        e0 = tf.zeros([batch_size, self.x_dim])

        # main iterative loop
        delta = None
        node_values = None

        E3_cable = self.ngc_model.cables['e2-to-z3_dense']
        E2_cable = self.ngc_model.cables['e1-to-z2_dense']
        E1_cable = self.ngc_model.cables['e0-to-z1_dense']

        W3_cable = self.ngc_model.cables['z3-to-mu2_dense']
        W2_cable = self.ngc_model.cables['z2-to-mu1_dense']
        W1_cable = self.ngc_model.cables['z1-to-mu0_dense']

        E3 = E3_cable.params["A"]
        E2 = E2_cable.params["A"]
        E1 = E1_cable.params["A"]
        W3 = W3_cable.params["A"]
        W2 = W2_cable.params["A"]
        W1 = W1_cable.params["A"]

        for k in range(self.K):
            node_values = []

            '''
            dz = dz_td + dz_bu - z * node.leak
            z = z * node.zeta + dz * node.beta
            '''

            z3_z = z3_node.compartments["z"]
            z2_z = z2_node.compartments["z"]
            z1_z = z1_node.compartments["z"]
            z0_z = z0_node.compartments["z"]

            z3_z = z3_z + self.beta * (- self.leak * z3_z + e2 @ E3)
            z2_z = z2_z + self.beta * (- self.leak * z2_z + e1 @ E2 - e2)
            z1_z = z1_z + self.beta * (- self.leak * z1_z + e0 @ E1 - e1)

            z3_node.compartments["z"] = z3_z
            z3_node.compartments["phi(z)"] = tf.nn.relu(z3_z)

            z2_node.compartments["z"] = z2_z
            z2_node.compartments["phi(z)"] = tf.nn.relu(z2_z)

            z1_node.compartments["z"] = z1_z
            z1_node.compartments["phi(z)"] = tf.nn.relu(z1_z)

            z0_node.compartments["phi(z)"] = z0_z

            node_vals = []
            for comp_name in z3_node.compartments:
                comp_value = z3_node.compartments.get(comp_name)
                node_vals.append((z3_node.name, comp_name, comp_value))
            node_values = node_values + node_vals

            node_vals = []
            for comp_name in z2_node.compartments:
                comp_value = z2_node.compartments.get(comp_name)
                node_vals.append((z2_node.name, comp_name, comp_value))
            node_values = node_values + node_vals

            node_vals = []
            for comp_name in z1_node.compartments:
                comp_value = z1_node.compartments.get(comp_name)
                node_vals.append((z1_node.name, comp_name, comp_value))
            node_values = node_values + node_vals

            node_vals = []
            for comp_name in z0_node.compartments:
                comp_value = z0_node.compartments.get(comp_name)
                node_vals.append((z0_node.name, comp_name, comp_value))
            node_values = node_values + node_vals

            # predictions

            z3 = z3_node.compartments["phi(z)"]
            z2 = z2_node.compartments["phi(z)"]
            z1 = z1_node.compartments["phi(z)"]

            mu2_z = z3 @ W3
            mu1_z = z2 @ W2
            mu0_z = z1 @ W1

            mu2 = tf.nn.relu(mu2_z)
            mu1 = tf.nn.relu(mu1_z)
            mu0 = tf.math.sigmoid(mu0_z)


            # calculate error nodes
            z2 = z2_node.compartments['phi(z)']
            z1 = z1_node.compartments['phi(z)']
            z0 = z0_node.compartments['phi(z)']

            e2 = z2 - mu2
            e1 = z1 - mu1
            e0 = z0 - mu0
            L_batch2 = tf.reduce_sum(e2 * e2, axis=1, keepdims=True)
            self.L2 = tf.reduce_sum(L_batch2)
            L_batch1 = tf.reduce_sum(e1 * e1, axis=1, keepdims=True)
            self.L1 = tf.reduce_sum(L_batch1)
            L_batch0 = tf.reduce_sum(e0 * e0, axis=1, keepdims=True)
            self.L0 = tf.reduce_sum(L_batch0)




        # parse results from static graph & place correct shallow-copied items in system dictionary
        self.ngc_model.parse_node_values(node_values)


        x_hat = mu0

        ##### manual update #####


        deltas = []
        # ['A_e2-to-z3_dense:0', 'A_e1-to-z2_dense:0', 'A_e0-to-z1_dense:0', 'A_z3-to-mu2_dense:0', 'A_z2-to-mu1_dense:0', 'A_z1-to-mu0_dense:0']
        z3 = self.ngc_model.nodes['z3'].compartments['phi(z)']
        z2 = self.ngc_model.nodes['z2'].compartments['phi(z)']
        z1 = self.ngc_model.nodes['z1'].compartments['phi(z)']

        avg_factor = 1.0 / self.batch_size
        deltas.append(-avg_factor * tf.matmul(e2, z3, transpose_a=True))
        deltas.append(-avg_factor * tf.matmul(e1, z2, transpose_a=True))
        deltas.append(-avg_factor * tf.matmul(e0, z1, transpose_a=True))
        deltas.append(-avg_factor * tf.matmul(z3, e2, transpose_a=True))
        deltas.append(-avg_factor * tf.matmul(z2, e1, transpose_a=True))
        deltas.append(-avg_factor * tf.matmul(z1, e0, transpose_a=True))


        # e2_z3.set_update_rule(preact=(e2,"phi(z)"), postact=(z3,"phi(z)"), gamma=e_gamma, param=["A"])
        # e1_z2.set_update_rule(preact=(e1,"phi(z)"), postact=(z2,"phi(z)"), gamma=e_gamma, param=["A"])
        # e0_z1.set_update_rule(preact=(e0,"phi(z)"), postact=(z1,"phi(z)"), gamma=e_gamma, param=["A"])
        # z3_mu2.set_update_rule(preact=(z3,"phi(z)"), postact=(e2,"phi(z)"), param=["A"])
        # z2_mu1.set_update_rule(preact=(z2,"phi(z)"), postact=(e1,"phi(z)"), param=["A"])
        # z1_mu0.set_update_rule(preact=(z1,"phi(z)"), postact=(e0,"phi(z)"), param=["A"])
        # update = tf.matmul(preact_term * w0, postact_term * w1, transpose_a=True)

        return x_hat, deltas




    def clip_weights(self):
        for (name, cable) in self.ngc_model.cables.items():
            if isinstance(cable, DCable):
                A = cable.params.get("A")
                A.assign(tf.clip_by_norm(A, 1.0, axes=[0]))

    def calc_updates(self, avg_update=True):
        """
        Calculate adjustments to parameters under this given model and its
        current internal state values

        Returns:
            delta, a list of synaptic matrix updates (that follow order of .theta)
        """
        Ns = self.ngc_model.extract("z0","phi(z)").shape[0]
        #delta = self.ngc_model.calc_updates()
        delta = self.delta
        if avg_update is True:
            for p in range(len(delta)):
                delta[p] = delta[p] * (1.0/(Ns * 1.0))
        return delta

    def get_parameters(self):
        E3_cable = self.ngc_model.cables['e2-to-z3_dense']
        E2_cable = self.ngc_model.cables['e1-to-z2_dense']
        E1_cable = self.ngc_model.cables['e0-to-z1_dense']

        W3_cable = self.ngc_model.cables['z3-to-mu2_dense']
        W2_cable = self.ngc_model.cables['z2-to-mu1_dense']
        W1_cable = self.ngc_model.cables['z1-to-mu0_dense']

        return [
            E3_cable.params["A"],
            E2_cable.params["A"],
            E1_cable.params["A"],
            W3_cable.params["A"],
            W2_cable.params["A"],
            W1_cable.params["A"]
        ]

    def get_total_discrepancy(self):
        L2 = self.L2
        L1 = self.L1
        L0 = self.L0
        return -(L2 + L1 + L0)


    def clear(self):
        """Clears the states/values of the stateful nodes in this NGC system"""
        self.ngc_model.clear()
        self.ngc_sampler.clear()
        self.delta = None

    def print_norms(self):
        """Prints the Frobenius norms of each parameter of this system"""
        str = ""
        for param in self.ngc_model.theta:
            str = "{} | {} : {}".format(str, param.name, tf.norm(param,ord=2))
        #str = "{}\n".format(str)
        return str

