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

        self.delta = None


        E3 = tf.random.normal((self.z_dim, self.z_top_dim), stddev=self.wght_sd, seed=self.seed)
        E2 = tf.random.normal((self.z_dim, self.z_dim), stddev=self.wght_sd, seed=self.seed)
        E1 = tf.random.normal((self.x_dim, self.z_dim), stddev=self.wght_sd, seed=self.seed)
        W3 = tf.random.normal((self.z_top_dim, self.z_dim), stddev=self.wght_sd, seed=self.seed)
        W2 = tf.random.normal((self.z_dim, self.z_dim), stddev=self.wght_sd, seed=self.seed)
        W1 = tf.random.normal((self.z_dim, self.x_dim), stddev=self.wght_sd, seed=self.seed)
        self.E3 = tf.Variable(E3)
        self.E2 = tf.Variable(E2)
        self.E1 = tf.Variable(E1)
        self.W3 = tf.Variable(W3)
        self.W2 = tf.Variable(W2)
        self.W1 = tf.Variable(W1)
        self.clip_weights()

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

        # clamp
        z0_z = x

        # Initialize the values of every non-clamped node
        z3_z = tf.zeros([batch_size, self.z_top_dim])
        z2_z = tf.zeros([batch_size, self.z_dim])
        z1_z = tf.zeros([batch_size, self.z_dim])

        e2 = tf.zeros([batch_size, self.z_dim])
        e1 = tf.zeros([batch_size, self.z_dim])
        e0 = tf.zeros([batch_size, self.x_dim])

        # main iterative loop
        for k in range(self.K):
            '''
            dz = dz_td + dz_bu - z * node.leak
            z = z * node.zeta + dz * node.beta
            '''
            z3_z = z3_z + self.beta * (- self.leak * z3_z + e2 @ self.E3)
            z2_z = z2_z + self.beta * (- self.leak * z2_z + e1 @ self.E2 - e2)
            z1_z = z1_z + self.beta * (- self.leak * z1_z + e0 @ self.E1 - e1)
            z3_out = tf.nn.relu(z3_z)
            z2_out = tf.nn.relu(z2_z)
            z1_out = tf.nn.relu(z1_z)
            z0_out = z0_z

            # predictions
            mu2_z = z3_out @ self.W3
            mu1_z = z2_out @ self.W2
            mu0_z = z1_out @ self.W1
            mu2 = tf.nn.relu(mu2_z)
            mu1 = tf.nn.relu(mu1_z)
            mu0 = tf.math.sigmoid(mu0_z)

            # calculate error nodes
            # NOTE: paper says it should below, but this doesnt work
            # e2 = z2_z - mu2
            # e1 = z1_z - mu1
            # e0 = z0_z - mu0

            e2 = z2_out - mu2
            e1 = z1_out - mu1
            e0 = z0_out - mu0

            L_batch2 = tf.reduce_sum(e2 * e2, axis=1, keepdims=True)
            self.L2 = tf.reduce_sum(L_batch2)
            L_batch1 = tf.reduce_sum(e1 * e1, axis=1, keepdims=True)
            self.L1 = tf.reduce_sum(L_batch1)
            L_batch0 = tf.reduce_sum(e0 * e0, axis=1, keepdims=True)
            self.L0 = tf.reduce_sum(L_batch0)


        x_hat = mu0

        ##### manual update #####

        deltas = []
        # ['A_e2-to-z3_dense:0', 'A_e1-to-z2_dense:0', 'A_e0-to-z1_dense:0', 'A_z3-to-mu2_dense:0', 'A_z2-to-mu1_dense:0', 'A_z1-to-mu0_dense:0']

        if calc_update == True:
            avg_factor = 1.0 / self.batch_size
            deltas.append(-avg_factor * tf.matmul(e2, z3_out, transpose_a=True))
            deltas.append(-avg_factor * tf.matmul(e1, z2_out, transpose_a=True))
            deltas.append(-avg_factor * tf.matmul(e0, z1_out, transpose_a=True))
            deltas.append(-avg_factor * tf.matmul(z3_out, e2, transpose_a=True))
            deltas.append(-avg_factor * tf.matmul(z2_out, e1, transpose_a=True))
            deltas.append(-avg_factor * tf.matmul(z1_out, e0, transpose_a=True))

        return x_hat, deltas




    def clip_weights(self):
        for param in self.get_parameters():
            param.assign(tf.clip_by_norm(param, 1.0, axes=[0]))

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
        return [
            self.E3,
            self.E2,
            self.E1,
            self.W3,
            self.W2,
            self.W1
        ]

    def get_total_discrepancy(self):
        L2 = self.L2
        L1 = self.L1
        L0 = self.L0
        return -(L2 + L1 + L0)


    def clear(self):
        """Clears the states/values of the stateful nodes in this NGC system"""
        self.delta = None

    def print_norms(self):
        """Prints the Frobenius norms of each parameter of this system"""
        str = ""
        for param in self.ngc_model.theta:
            str = "{} | {} : {}".format(str, param.name, tf.norm(param,ord=2))
        #str = "{}\n".format(str)
        return str

