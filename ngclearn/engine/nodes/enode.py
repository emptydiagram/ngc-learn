import tensorflow as tf
import sys
import numpy as np
import copy
from ngclearn.engine.nodes.node import Node
from ngclearn.utils import transform_utils

class ENode(Node):
    """
    | Implements a (rate-coded) error node simplified to its fixed-point form:
    |   e = target - mu // in the case of squared error (Gaussian error units)
    |   e = signum(target - mu) // in the case of absolute error (Laplace error units)
    | where:
    |   target - a desired target activity value (target = pred_targ)
    |   mu - an external prediction signal of the target activity value (mu = pred_mu)

    | Compartments:
    |   * pred_mu - prediction signals (deposited signals summed)
    |   * pred_targ - target signals (deposited signals summed)
    |   * z - the error neural activities, set as z = e
    |   * phi(z) -  the post-activation of the error activities in z
    |   * L - the local loss represented by the error activities
    |   * avg_scalar - multiplies L and z by (1/avg_scalar)

    Args:
        name: the name/label of this node

        dim: number of neurons this node will contain/model

        error_type: type of distance/error measured by this error node. Setting this
            to "mse" will set up squared-error neuronal units (derived from
            L = 0.5 * ( Sum_j (target - mu)^2_j )), and "mae" will set up
            mean absolute error neuronal units (derived from L = Sum_j \|target - mu\| ).

        act_fx: activation function -- phi(v) -- to apply to error activities (Default = "identity")

        batch_size: batch-size this node should assume (for use with static graph optimization)

        precis_kernel: 2-Tuple defining the initialization of the precision weighting synapses that will
            modulate the error neural activities. For example, an argument could be: ("uniform", 0.01)
            The value types inside each slot of the tuple are specified below:

            :init_scheme (Tuple[0]): initialization scheme, e.g., "uniform", "gaussian".

            :init_scale (Tuple[1]): scalar factor controlling the scale/magnitude of initialization distribution, e.g., 0.01.

            :Note: specifying None will result in precision weighting being applied to the error neurons.
                Understand that care should be taken w/ respect to this particular argument as precision
                synapses involve an approximate inversion throughout simulation steps

        constraint_kernel: Dict defining the constraint type to be applied to the learnable parameters
            of this node. The expected keys and corresponding value types are specified below:

            :`'clip_type'`: type of clipping constraint to be applied to learnable parameters/synapses.
                If "norm_clip" is specified, then norm-clipping will be applied (with a check if the
                norm exceeds "clip_mag"), and if "forced_norm_clip" then norm-clipping will be applied
                regardless each time apply_constraint() is called.

            :`'clip_mag'`: the magnitude of the worse-case bounds of the clip to apply/enforce.

            :`'clip_axis'`: the axis along which the clipping is to be applied (to each matrix).

            :Note: specifying None will mean no constraints are applied to this node's parameters

        ex_scale: a scale factor to amplify error neuron signals (Default = 1)
    """
    def __init__(self, name, dim, error_type="mse", act_fx="identity", batch_size=1,
                 precis_kernel=None, constraint_kernel=None):
        node_type = "error"
        super().__init__(node_type, name, dim)
        self.dim = dim
        self.batch_size = batch_size
        self.error_type = error_type
        self.act_fx = act_fx
        self.fx = tf.identity
        self.dfx = None
        self.is_clamped = False

        self.compartment_names = ["pred_mu", "pred_targ", "z", "phi(z)", "L"]
        self.compartments = {}
        for name in self.compartment_names:
            name_v = None
            if "phi(z)" in name:
                name_v = "{}_phi_z".format(self.name)
                self.compartments[name] = tf.Variable(tf.zeros([batch_size,dim]), name=name_v)
            elif "L" in name:
                name_v = "{}_{}".format(self.name, name)
                self.compartments[name] = tf.Variable(tf.zeros([1,1]), name=name_v)
            else:
                name_v = "{}_{}".format(self.name, name)
                self.compartments[name] = tf.Variable(tf.zeros([batch_size,dim]), name=name_v)

        self.connected_cables = []

        self.constraint_kernel = constraint_kernel

    def compile(self):
        info = super().compile()
        # we have to special re-compile the L compartment to be (1 x 1)
        self.compartments["L"] = tf.Variable(tf.zeros([1,1]), name="{}_L".format(self.name))

        info["error_type"] = self.error_type
        info["phi(x)"] = self.act_fx
        return info

    def set_constraint(self, constraint_kernel):
        self.constraint_kernel = constraint_kernel

    def step(self, injection_table=None, skip_core_calc=False):

        ########################################################################
        if skip_core_calc == False:
            if self.is_clamped == False:
                # clear any relevant compartments that are NOT stateful before accruing
                # new deposits (this is crucial to ensure any desired stateless properties)
                self.compartments["pred_mu"] = (self.compartments["pred_mu"] * 0)
                self.compartments["pred_targ"]= (self.compartments["pred_targ"] * 0)

                # gather deposits from any connected nodes & insert them into the
                # right compartments/regions -- deposits in this logic are linearly combined
                for cable in self.connected_cables:
                    deposit = cable.propagate()
                    dest_comp = cable.dest_comp
                    self.compartments[dest_comp] = (deposit + self.compartments[dest_comp])

                # core logic for the (node-internal) dendritic calculation
                # error neurons are a fixed-point result/calculation as below:
                pred_targ = self.compartments["pred_targ"]
                pred_mu = self.compartments["pred_mu"]

                z = None
                L_batch = None
                L = None
                if self.error_type == "mse": # squared error neurons
                    z = e = pred_targ - pred_mu
                    # print(f"[{self.name}]  ||pred_targ|| = {tf.norm(pred_targ)}, ||pred_mu|| = {tf.norm(pred_mu)}, ||e|| = {tf.norm(e)}")
                    # compute local loss that this error node represents
                    L_batch = tf.reduce_sum(e * e, axis=1, keepdims=True) #/(e.shape[0] * 2.0)

                L = tf.reduce_sum(L_batch) # sum across dimensions

                self.compartments["L"] = np.asarray([[L]])

                self.compartments["z"] = z

            # else, no deposits are accrued (b/c this node is hard-clamped to a signal)
            ########################################################################

        # apply post-activation non-linearity
        self.compartments["phi(z)"] = (self.fx(self.compartments["z"]))


        ########################################################################
        self.t += 1

        # a node returns a list of its named component values
        values = []
        for comp_name in self.compartments:
            comp_value = self.compartments.get(comp_name)
            values.append((self.name, comp_name, comp_value))
        return values
