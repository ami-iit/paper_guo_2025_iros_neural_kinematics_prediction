r"""MLP-based Diffkin-informed joint kinematics prediction network"""
import torch
from torch import nn
import roma
#from config import config as c

## pMLP ##
class pMLP(nn.Module):
    def __init__(self, sample, comp, use_buffer, wholebody):
        super(pMLP, self).__init__()
        if sample is not None:
            # input features
            self.x_acc_size = sample['acc'].size()[-1]
            self.x_ori_size = sample['ori'].size()[-1]
            self.s_buffer_size = sample['s_buffer'].size()[-1]
            self.sdot_buffer_size = sample['sdot_buffer'].size()[-1]

            # window shape
            self.window_size = sample['acc'].size()[1]
            self.buffer_length = sample['s_buffer'].size()[1]

            # output features
            self.s_step_size = sample['s'].size()[-1]
            self.sdot_step_size = sample['sdot'].size()[-1]
            self.prediction_steps = sample['s'].size()[1]
        else:
            self.x_acc_size = 3*5
            self.x_ori_size = 9*5
            self.s_buffer_size = 31
            self.sdot_buffer_size = 31

            # window shape
            self.window_size = 10
            self.buffer_length = 10

            # output features
            self.s_step_size = 31
            self.sdot_step_size = 31
            self.prediction_steps = 60

        self.use_buffer = use_buffer
        if self.use_buffer:
            """[acc, ori, s_buffer, sdot_buffer]"""
            self.input_channel = 4
        else:
            """[acc, ori]"""
            self.input_channel = 2
        
        # body segmented network structure
        self.wholebody = wholebody

        if self.wholebody:
            self.s_upper_step_size = 12 
            self.sdot_upper_step_size = 12

            self.s_lower_step_size = 19
            self.sdot_lower_step_size = 19

        # kindyncomputation instance
        self.comp = comp

        # network layer width
        n0 = 256
        n1 = 512
        n2 = 512

        # activation func 
        self.activation = nn.ELU()
  
        # input layers for two channels
        self.acc_input_layer = nn.Linear(self.x_acc_size*self.window_size, n0).double() 
        self.ori_input_layer = nn.Linear(self.x_ori_size*self.window_size, n0).double() 
        self.s_buffer_layer = nn.Linear(self.s_buffer_size*self.buffer_length, n0).double()
        self.sdot_buffer_layer = nn.Linear(self.sdot_buffer_size*self.buffer_length, n0).double()

        # define fc layers
        self.fcnn1 = nn.Linear(n0*self.input_channel, n1).double() 
        self.fcnn2 = nn.Linear(n1, n2).double() 
        self.fcnn3 = nn.Linear(n2, n2).double()
        self.fcnn4 = nn.Linear(n2, n2).double() 

        if self.wholebody:
            self.fcnn_upper_layer = nn.Linear(n2, n2).double()
            self.fcnn_lower_layer = nn.Linear(n2, n2).double()

        # define output layers
        self.s_output_layer = nn.Linear(n2, self.s_step_size*self.prediction_steps).double() 
        self.sdot_output_layer = nn.Linear(n2, self.sdot_step_size*self.prediction_steps).double() 

        if self.wholebody:
            self.s_output_upper_layer = nn.Linear(n2, self.s_upper_step_size*self.prediction_steps).double()
            self.sdot_output_upper_layer = nn.Linear(n2, self.sdot_upper_step_size*self.prediction_steps).double()

            self.s_output_lower_layer = nn.Linear(n2, self.s_lower_step_size*self.prediction_steps).double()
            self.sdot_output_lower_layer = nn.Linear(n2, self.sdot_lower_step_size*self.prediction_steps).double()


    @torch.autocast(device_type="cuda")
    def forward(self, acc, ori, s_buffer, sdot_buffer):
        # flatten the acc and ori along the time axis
        acc = torch.flatten(acc, start_dim=1, end_dim=2) # e.g., (bs, 10*3*5)
        ori = torch.flatten(ori, start_dim=1, end_dim=2) # e.g., (bs, 10*9*5)
       
        # apply input layers accordingly
        acc = self.acc_input_layer(acc) # (bs, 10*3*5)->(bs, 512)
        acc = self.activation(acc)

        ori = self.ori_input_layer(ori) # (bs, 10*9*5)->(bs, 512)
        ori = self.activation(ori)

        if self.use_buffer:
            s_buffer = torch.flatten(s_buffer, start_dim=1, end_dim=2)
            sdot_buffer = torch.flatten(sdot_buffer, start_dim=1, end_dim=2)

            s_buffer = self.s_buffer_layer(s_buffer)
            s_buffer = self.activation(s_buffer)

            sdot_buffer = self.sdot_buffer_layer(sdot_buffer)
            sdot_buffer = self.activation(sdot_buffer)

            x = torch.concat((acc, ori, s_buffer, sdot_buffer), -1) # (bs, 512*2)
        else:
            x = torch.concat((acc, ori), -1)

        # layer1
        y0 = self.fcnn1(x) # (bs, 512*2) -> (bs, 1024)
        y0 = self.activation(y0)

        # layer2
        y1 = self.fcnn2(y0) # (bs, 1024) -> (bs, 1024)
        y1 = self.activation(y1)

        if not self.wholebody:
            # layer3
            y2 = self.fcnn3(y1) # (bs, 1024) -> (bs, 1024)
            y2 = self.activation(y2)

            # layer4
            y3 = self.fcnn4(y2) # (bs, 1024) -> (bs, 1024)
            y3 = self.activation(y3)

            # output layers
            y_s = self.s_output_layer(y3) # (bs, 1024) -> (bs, 66*10)
            unflatten_s = nn.Unflatten(1, torch.Size([self.prediction_steps, self.s_step_size]))
            y_s_unflatten = unflatten_s(y_s) # (bs, 10, 66)

            y_sdot = self.sdot_output_layer(y3) # (bs, 1024) -> (bs, 66*10)
            unflatten_sdot = nn.Unflatten(1, torch.Size([self.prediction_steps, self.sdot_step_size]))
            y_sdot_unflatten = unflatten_sdot(y_sdot) # (bs, 10, 66)
        else:
            # upper body layer
            y_upper = self.fcnn_upper_layer(y1)
            y_upper = self.activation(y_upper)

            # upper body s and sdot layer
            y_s_upper = self.s_output_upper_layer(y_upper)
            unflatten_s_upper = nn.Unflatten(1, torch.Size([self.prediction_steps, self.s_upper_step_size]))
            y_s_upper_unflatten = unflatten_s_upper(y_s_upper) # (bs, 10, 12)

            y_sdot_upper = self.sdot_output_upper_layer(y_upper)
            unflatten_sdot_upper = nn.Unflatten(1, torch.Size([self.prediction_steps, self.sdot_upper_step_size]))
            y_sdot_upper_unflatten = unflatten_sdot_upper(y_sdot_upper) # (bs, 10, 12)

            # lower body layer
            y_lower = self.fcnn_lower_layer(y1)
            y_lower = self.activation(y_lower)

            # lower body s and sdot layer
            y_s_lower = self.s_output_lower_layer(y_lower)
            unflatten_s_lower = nn.Unflatten(1, torch.Size([self.prediction_steps, self.s_lower_step_size]))
            y_s_lower_unflatten = unflatten_s_lower(y_s_lower) # (bs, 10, 19)

            y_sdot_lower = self.sdot_output_lower_layer(y_lower)
            unflatten_sdot_lower = nn.Unflatten(1, torch.Size([self.prediction_steps, self.sdot_lower_step_size]))
            y_sdot_lower_unflatten = unflatten_sdot_lower(y_sdot_lower) # (bs, 10, 19)

            # concatenate upper and lower body predictions
            y_s_unflatten = torch.cat((y_s_upper_unflatten, y_s_lower_unflatten), -1)
            y_sdot_unflatten = torch.cat((y_sdot_upper_unflatten, y_sdot_lower_unflatten), -1)

        return y_s_unflatten, y_sdot_unflatten
    
    @staticmethod
    def compute_grad(output, input):
        r"""Compute the partial derivative of output w.r.t. specified input.
        Args:
            - output: (bs, T1, D1) tensor
            - input: (bs, T2, D2) tensor
        """
        #print('check output requires grad: ', output.requires_grad)
        #print('check input requires grad: ', input.requires_grad)
        return torch.autograd.grad(output, input, 
                                   grad_outputs=torch.ones_like(output), 
                                   create_graph=True,
                                   allow_unused=True)
    
    @staticmethod
    def forward_kinematics(comp, pb_ref, rb_ref, s_pred, link_names):
        r"""batched forward kinematics"""
        #step = c.train_consts["pi_step"]
        step = 0
        bs = s_pred.shape[0]

        # get the gt base data
        pb_ref_step = pb_ref[:, step, :].reshape((-1, 3, 1))
        rb_ref_step = rb_ref[:, step, :].reshape((-1, 3, 3))

        Hb_ref_step = torch.cat((torch.cat((rb_ref_step, pb_ref_step), dim=2), 
                                 torch.tensor(bs*[0, 0, 0, 1]).reshape(-1, 1, 4)), dim=1)
        
        # get the jpos pred
        s_pred_step = s_pred[:, step, :]
        
        # vectorize the fk function
        fk_vmap = torch.vmap(comp.forward_kinematics_vmap, in_dims=(None, 0, 0))

        """LeftLowerLeg link"""
        H_link1 = fk_vmap(link_names[0], Hb_ref_step, s_pred_step)
        p_link1 = H_link1[:, :3, 3].reshape((-1, 1, 3))
        r_link1 = H_link1[:, :3, :3].reshape((-1, 1, 9))
        r_link1_svd = roma.special_procrustes(r_link1.reshape((-1, 1, 3, 3)))

        """RightLowerLeg link"""
        H_link2 = fk_vmap(link_names[1], Hb_ref_step, s_pred_step)
        p_link2 = H_link2[:, :3, 3].reshape((-1, 1, 3))
        r_link2 = H_link2[:, :3, :3].reshape((-1, 1, 9))
        r_link2_svd = roma.special_procrustes(r_link2.reshape((-1, 1, 3, 3)))

        #p_links = torch.cat((p_leftlowerleg, p_rightlowerleg), 2)

        """ H_link = fk_vmap(link_name, Hb_ref_step, s_pred_step)
        p_link = H_link[:, :3, 3].reshape((-1, 1, 3))
        R_link = H_link[:, :3, :3].reshape((-1, 3, 3)) """

        plinks = [p_link1, p_link2]
        rlinks = [r_link1_svd.reshape((-1, 1, 9)), r_link2_svd.reshape((-1, 1, 9))]
        return plinks, rlinks
    

    @staticmethod
    def differential_kinematics(comp, pb_ref, rb_ref, vb_ref, s_pred, sdot_pred, 
                                link_names, use_wholebody):
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
        # some configurations
        #step = c.train_consts["pi_step"]
        step = 0
        #dofs = c.train_consts["dofs"]
        dofs = 31
        bs = s_pred.shape[0]

        # get one step data
        pb_step = pb_ref[:, step, :3].reshape((-1, 3, 1))
        rb_step = rb_ref[:, step, :9].reshape((-1, 3, 3))
        vb_step = vb_ref[:, step, :6]

        s_pred_step = s_pred[:, step, :]
        sdot_pred_step = sdot_pred[:, step, :]

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

        if use_wholebody:
            J3_pred = jacobian_vmap(link_names[2], Hb_step, s_pred_step)
            vlink3_pred = torch.matmul(J3_pred, nu_pred_step).reshape((-1, 1, 6))

            J4_pred = jacobian_vmap(link_names[3], Hb_step, s_pred_step)
            vlink4_pred = torch.matmul(J4_pred, nu_pred_step).reshape((-1, 1, 6))
            return [vlink1_pred, vlink2_pred, vlink3_pred, vlink4_pred]
        else:
            return [vlink1_pred, vlink2_pred]