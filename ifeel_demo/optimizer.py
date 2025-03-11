import numpy as np
import pandas as pd
import casadi as ca

def build(comp, links, dofs, wholebody):
    r"""Create the NLP instance"""
    # decision variables
    s_var = ca.SX.sym('s', dofs)
    sdot_var = ca.SX.sym('sdot', dofs)
    x_var = ca.vertcat(s_var, sdot_var)

    # parameters
    vb_var = ca.SX.sym('vb', 6)
    pb_var = ca.SX.sym('pb', 3)
    rb_var = ca.SX.sym('rb', 9)

    # create Hb based on rb and pb
    Rb_var = ca.vertcat(rb_var[:3].T, rb_var[3:6].T, rb_var[6:].T)
    Hb_var = ca.blockcat([[Rb_var, pb_var], 
                            [0, 0, 0, 1]])  
    nu_var = ca.vertcat(vb_var, sdot_var)

    vleftfoot_var = ca.SX.sym('vleftfoot', 6)
    vrightfoot_var = ca.SX.sym('vrightfoot', 6)

    jacob_leftfoot = comp.jacobian_fun(links[0])
    jacob_rightfoot = comp.jacobian_fun(links[1])

    leftfoot_cost = ca.sumsqr(jacob_leftfoot(Hb_var, s_var)@nu_var-vleftfoot_var)
    rightfoot_cost = ca.sumsqr(jacob_rightfoot(Hb_var, s_var)@nu_var-vrightfoot_var)
    if not wholebody:
        p_var = ca.vertcat(
            vb_var, 
            pb_var, 
            rb_var, 
            vleftfoot_var, 
            vrightfoot_var)
        # cost function: left and right feet
        const_func = leftfoot_cost + rightfoot_cost
    else:
        vleftarm_var = ca.SX.sym('vleftarm', 6)
        vrightarm_var = ca.SX.sym('vrightarm', 6)
        p_var = ca.vertcat(
            vb_var, 
            pb_var, 
            rb_var, 
            vleftfoot_var, 
            vrightfoot_var,
            vleftarm_var,
            vrightarm_var)
        jacob_leftarm = comp.jacobian_fun(links[2])
        jacob_rightarm = comp.jacobian_fun(links[3])
        # cost function: left/right foot/arm
        leftarm_cost = ca.sumsqr(jacob_leftarm(Hb_var, s_var)@nu_var-vleftarm_var)
        rightarm_cost = ca.sumsqr(jacob_rightarm(Hb_var, s_var)@nu_var-vrightarm_var)

        const_func = leftfoot_cost + rightfoot_cost + leftarm_cost + rightarm_cost


   
    args = {
            'x0' : None,
            'p': None,
            'lbx': None,
            'ubx': None
            }   
    nlp = {'x': x_var, 'f': const_func, 'p': p_var}
    opts = {'ipopt': {'print_level': 0,
                      'tol': 1e-3},
            'print_time': 0}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    return args, solver


def optimize(args, solver, joint_num, base_num, link_num, dofs, wholebody):
    s_num = joint_num[0].reshape((dofs))
    sdot_num = joint_num[1].reshape((dofs))
    x_num = ca.vertcat(s_num, sdot_num)

    vb_num = base_num[0].reshape((6))
    rb_num = base_num[1].reshape((9))
    pb_num = base_num[2].reshape((3))

    # the first two always left/right foot numerical values
    vleftfoot_num = link_num[0].reshape((6))
    vrightfoot_num = link_num[1].reshape((6))

    if wholebody:
        vleftarm_num = link_num[2].reshape((6))
        vrightarm_num = link_num[3].reshape((6))
        p_num = ca.vertcat(
            vb_num, 
            pb_num, 
            rb_num, 
            vleftfoot_num, 
            vrightfoot_num,
            vleftarm_num,
            vrightarm_num
            )
    else:
        p_num = ca.vertcat(
            vb_num, 
            pb_num, 
            rb_num, 
            vleftfoot_num, 
            vrightfoot_num
            )

    args['x0'] = x_num
    args['p'] = p_num

    args['lbx'] = x_num - 3
    args['ubx'] = x_num + 3

    sol = solver(x0=args['x0'], p=args['p'], lbx=args['lbx'], ubx=args['ubx'])
    s_opt = sol['x'][:31].reshape((1, 31))
    sdot_opt = sol['x'][31:].reshape((1, 31))
    return s_opt, sdot_opt