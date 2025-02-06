r"""physics-informed MLP-based network"""
import torch
from torch import nn
import const as con

class pMLP(nn.Module):
    def __init__(self, sample, comp, use_buffer, is_wholebody):
        r"""
        Args:
            :param sample --> an I/O instance from the dataset
            ;param comp --> idyntree-based KinDynComputation instance
            :param use_buffer --> flag if leverage historical states as inupt
            :param is_wholebody --> flag is is wholebody task 
        """
        super().__init__()

        # input feature sizes
        self.x_acc_size = sample['acc'].size()[-1]
        self.x_ori_size = sample['ori'].size()[-1]
        self.s_buffer_size = sample['s_buffer'].size()[-1]
        self.sdot_buffer_size = sample['sdot_buffer'].size()[-1]

        self.use_buffer = use_buffer
        self.input_channel = 4 if use_buffer else 2

        # sliding window shape
        self.window_size = sample['acc'].size()[1]
        self.buffer_length = sample['s_buffer'].size()[1]
        if not self.window_size == self.buffer_length:
            raise ValueError(f"Expect window size equalt to buffer length, but got {self.window_size} and {self.buffer_length}")

        # output feature sizes
        self.s_step_size = sample['s'].size()[-1]
        self.sdot_step_size = sample['sdot'].size()[-1]
        self.prediction_horizon = sample['s'].size()[1]

        self.is_wholebody = is_wholebody
        if is_wholebody:
            self.s_upper_step_size = 12
            self.sdot_upper_step_size = 12
            self.s_lower_step_size = 19
            self.sdot_lower_step_size = 19

        self.comp = comp

        # network layer dimensions
        n0, n1, n2 = 256, 512, 512
        self.activation = nn.ELU()

        # input layers
        self.input_layers = nn.ModuleDict(
            {
                "acc": nn.Linear(self.x_acc_size*self.window_size, n0).double(),
                "ori": nn.Linear(self.x_ori_size*self.window_size, n0).double()
            }
        )
        if self.use_buffer:
            self.input_layers.update(
                {
                    "s_buffer": nn.Linear(self.s_buffer_size*self.buffer_length, n0).double(),
                    "sdot_buffer": nn.Linear(self.sdot_buffer_size*self.buffer_length, n0).double()
                }
            )
        # fully connected layers
        self.fcnn = nn.Sequential(
            nn.Linear(n0*self.input_channel, n1).double(),
            self.activation,
            nn.Linear(n1, n2).double(),
            self.activation
        )
        if self.is_wholebody:
            self.upper_body_layers = self._create_body_layer(n2)
            self.lower_body_layers = self._create_body_layer(n2)
        else:
            self.deep_fcnn = nn.Sequential(
                nn.Linear(n2, n2).double(),
                self.activation,
                nn.Linear(n2, n2).double(),
                self.activation
            )
        
        # output layers
        self.output_layers = self._create_output_layer(n2)

    def _create_body_layer(self, dim):
        r"""create fully connected layer for half-body sehmentation"""
        return nn.Sequential(
            nn.Linear(dim, dim).double(),
            self.activation
        )
    
    def _creat_output_layer(self, dim):
        r"""create output layers for s and sdot predictions"""
        output_layers = nn.ModuleDict(
            {
                "s": nn.Linear(dim, self.s_step_size*self.prediction_horizon).double(),
                "sdot": nn.Linear(dim, self.sdot_step_size*self.prediction_horizon).double()
            }
        )
        if self.is_wholebody:
            output_layers.update(
                {
                    "s_upper": nn.Linear(dim, self.s_upper_step_size*self.prediction_horizon).double(),
                    "sdot_upper": nn.Linear(dim, self.sdot_upper_step_size*self.prediction_horizon).double(),
                    "s_lower": nn.Linear(dim, self.s_lower_step_size*self.prediction_horizon).double(),
                    "sdot_lower": nn.Linear(dim, self.sdot_lower_step_size*self.prediction_horizon).double()
                }
            )
        return output_layers
    
    @torch.autocast(device_type="cuda")
    def forward(self, acc, ori, s_buffer=None, sdot_buffer=None):
        # flatten the time axis
        acc = torch.flatten(acc, start_dim=1, end_dim=2)
        ori = torch.flatten(ori, start_dim=1, end_dim=2)

        # process input features accordingly
        x = [self.activation(self.input_layers["acc"](acc))]
        x.append(self.activation(self.input_layers["ori"](ori)))

        if self.use_buffer:
            s_buffer = torch.flatten(s_buffer, start_dim=1, end_dim=2)
            sdot_buffer = torch.flatten(sdot_buffer, start_dim=1, end_dim=2)
            x.append(self.activation(self.input_layers["s_buffer"](s_buffer)))
            x.append(self.activation(self.input_layers["sdot_buffer"](sdot_buffer)))
        x = torch.cat(x, dim=1)

        # fully connected layers
        y = self.fcnn(x)

        if not self.is_wholebody:
            y = self.deep_fcnn(y)
            y_s = self.output_layers["s"](y)
            y_sdot = self.output_layers['sdot'](y)
        else:
            y_upper = self.upper_body_layers(y)
            y_lower = self.lower_body_layers(y)

            y_s_upper = self.output_layers["s_upper"](y_upper)
            y_sdot_upper = self.output_layers["sdot_upper"](y_upper)
            y_s_lower = self.output_layers["s_lower"](y_lower)
            y_sdot_lower = self.output_layers["sdot_lower"](y_lower)

            y_s = torch.cat((y_s_upper, y_s_lower), dim=-1)
            y_sdot = torch.cat((y_sdot_upper, y_sdot_lower), dim=-1)

        # reshape the outputs
        y_s = y_s.view(-1, self.prediction_horizon, self.s_step_size)
        y_sdot = y_sdot.view(-1, self.prediction_horizon, self.sdot_step_size)

        return y_s, y_sdot
    
    @staticmethod
    def compute_grad(x, y):
        r"""
        Compute the partial derivation of output w.r.t. input.
        Args:
            :param x --> input, shape (bs, T1, D1) tensor
            :param y --> output, shape (bs, T2, D2) tensor
        """
        return torch.autograd.grad(
            y, x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            allow_unused=True
        )
    
    @staticmethod
    def forward_kinematics(comp, pb_ref, rb_ref, s_pred, link_names, pi_step):
        r"""batched forward kinematics"""
        bs = s_pred.shape[0]

        # get the gt base data
        pb_ref_step = pb_ref[:, pi_step, :].reshape((-1, 3, 1))
        rb_ref_step = rb_ref[:, pi_step, :].reshape((-1, 3, 3))

        Hb_ref_step = torch.cat((torch.cat((rb_ref_step, pb_ref_step), dim=2), 
                                 torch.tensor(bs*[0, 0, 0, 1]).reshape(-1, 1, 4)), dim=1)
        
        # get the jpos pred
        s_pred_step = s_pred[:, pi_step, :]
        
        # vectorize the fk function
        fk_vmap = torch.vmap(comp.forward_kinematics_vmap, in_dims=(None, 0, 0))

        """LeftLowerLeg link"""
        H_link1 = fk_vmap(link_names[0], Hb_ref_step, s_pred_step)
        p_link1 = H_link1[:, :3, 3].reshape((-1, 1, 3))
        r_link1 = H_link1[:, :3, :3].reshape((-1, 1, 9))

        """RightLowerLeg link"""
        H_link2 = fk_vmap(link_names[1], Hb_ref_step, s_pred_step)
        p_link2 = H_link2[:, :3, 3].reshape((-1, 1, 3))
        r_link2 = H_link2[:, :3, :3].reshape((-1, 1, 9))

        return p_link1, r_link1, p_link2, r_link2
    
    @staticmethod
    def differential_kinematics(comp, 
                                pb_ref, rb_ref, vb_ref, 
                                s_pred, sdot_pred, 
                                link_names, is_wholebody,
                                pi_step, dofs):
        r"""
        Args:
            :param::comp -> Adam.pytorch KinDynComputation object
            :param::pb_ref -> base position gt
            :param::rb_ref -> base orientation gt (rotation matrix)
            :param::vb_ref -> base 6D velocity gt (linear/angular)
            :param::s_pred -> joint position prediction
            :param::sdot_pred -> joint velocity prediction
            :param::link_names -> list of links
        Return:
            v_links -> 6D link velocity
        """
        bs = s_pred.shape[0]

        # get one step data
        pb_step = pb_ref[:, pi_step, :3].reshape((-1, 3, 1))
        rb_step = rb_ref[:, pi_step, :9].reshape((-1, 3, 3))
        vb_step = vb_ref[:, pi_step, :6]

        s_pred_step = s_pred[:, pi_step, :]
        sdot_pred_step = sdot_pred[:, pi_step, :]

        # prepare the Hb matrix and nu vector
        Hb_step = torch.cat((torch.cat((rb_step, pb_step), dim=2), 
                             torch.tensor(bs*[0, 0, 0, 1]).reshape(-1, 1, 4)), dim=1)
        nu_pred_step = torch.cat((vb_step, sdot_pred_step), dim=-1).reshape((-1, 6+dofs, 1))

        # prepare the jacobian function
        jacobian_vmap = torch.vmap(comp.jacobian, in_dims=(None, 0, 0))

        # compute the velocity of the selected link
        J1_pred = jacobian_vmap(link_names[0], Hb_step, s_pred_step)
        vlink1_pred = torch.matmul(J1_pred, nu_pred_step).reshape((-1, 1, 6))

        J2_pred = jacobian_vmap(link_names[1], Hb_step, s_pred_step)
        vlink2_pred = torch.matmul(J2_pred, nu_pred_step).reshape((-1, 1, 6))

        if is_wholebody:
            # check if the num of links correct
            if not len(link_names) == 4:
                raise ValueError(f"Expected num of links to be 4 for wholebody task, but got {len(link_names)}.")
            J3_pred = jacobian_vmap(link_names[2], Hb_step, s_pred_step)
            vlink3_pred = torch.matmul(J3_pred, nu_pred_step).reshape((-1, 1, 6))

            J4_pred = jacobian_vmap(link_names[3], Hb_step, s_pred_step)
            vlink4_pred = torch.matmul(J4_pred, nu_pred_step).reshape((-1, 1, 6))
            return [vlink1_pred, vlink2_pred, vlink3_pred, vlink4_pred]
        else:
            return [vlink1_pred, vlink2_pred]