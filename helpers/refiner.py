import casadi as ca
from typing import Tuple, Dict, Any

def build(comp, links: dict, dofs: int, is_wholebody: bool) -> Tuple[dict, ca.nlpsol]:
    r"""Create a NLP instance."""
    # decision vars
    s_var = ca.SX.sym('s', dofs)
    sdot_var = ca.SX.sym('sdot', dofs)
    x_var = ca.vertcat(s_var, sdot_var)

    # parameters
    vb_var = ca.SX.sym('vb', 6)
    pb_var = ca.SX.sym('pb', 3)
    rb_var = ca.SX.sym('rb', 9) # flattened rotation matrix

    # construct Hb (homogeneous transformation matrix)
    Rb_var = ca.vertcat(rb_var[:3].T, rb_var[3:6].T, rb_var[6:].T)
    Hb_var = ca.blockcat([
        [Rb_var, pb_var], 
        [0, 0, 0, 1]
    ])  

    # generalized velocity
    nu_var = ca.vertcat(vb_var, sdot_var)

    # lowerleg velocity vars
    vleftlowerleg_var = ca.SX.sym('vleftlowerleg', 6)
    vrightlowerleg_var = ca.SX.sym('vrightlowerleg', 6)

    # compute jacobian for lowerlegs
    jacob_left_lowerleg = comp.jacobian_fun(links['LeftLowerLeg'])
    jacob_right_lowerleg = comp.jacobian_fun(links['RightLowerLeg'])

    # compute cost for lowerlegs
    cost_left_lowerleg = ca.sumsqr(jacob_left_lowerleg(Hb_var, s_var) @ nu_var - vleftlowerleg_var)
    cost_right_lowerleg = ca.sumsqr(jacob_right_lowerleg(Hb_var, s_var) @ nu_var - vrightlowerleg_var)

    # initialize params and cost functions
    p_var = ca.vertcat(vb_var, pb_var, rb_var, vleftlowerleg_var, vrightlowerleg_var)
    cost_funcs = cost_left_lowerleg + cost_right_lowerleg

    # if wholebody dynamics is considered
    if is_wholebody:
        # check if the links include wholebody links
        if len(is_wholebody) != 4:
            raise ValueError(f"Wholbody links should include 4 links, but got {len(links)}.")
        vleftarm_var = ca.SX.sym('vleftarm', 6)
        vrightarm_var = ca.SX.sym('vrightarm', 6)
        # append new params
        p_var = ca.vertcat(p_var, vleftarm_var, vrightarm_var)
        # compute jacobians for left and right arms
        jacob_leftarm = comp.jacobian_fun(links['LeftForeArm'])
        jacob_rightarm = comp.jacobian_fun(links['RightForeArm'])

        # compute costs for arms
        cost_leftarm = ca.sumsqr(jacob_leftarm(Hb_var, s_var) @ nu_var - vleftarm_var)
        cost_rightarm = ca.sumsqr(jacob_rightarm(Hb_var, s_var) @ nu_var - vrightarm_var)

        # update total cost functions
        cost_funcs += cost_leftarm + cost_rightarm
    # nlp problem definition
    nlp = {'x': x_var, 'p': p_var, 'f': cost_funcs}

    # solver options
    solver_opts = {
        'ipopt': {
            'print_level': 0,
            'acceptable_tol': 1e-3,
            'tol': 1e-3
        },
        'print_time': 0
    }

    # create solver instance
    solver = ca.nlpsol('solver', 'ipopt', nlp, solver_opts)

    # arguments placeholder
    args = {'x0': None, 'p': None, 'lbx': None, 'ubx': None}
    return args, solver

def refine(
        args: Dict[str, Any],
        solver: ca.nlpsol,
        joint_num: Tuple[ca.SX, ca.SX],
        base_num: Tuple[ca.SX, ca.SX, ca.SX],
        link_num: Tuple[ca.SX, ...],
        dofs: int,
        is_wholebody: bool
) -> Tuple[ca.SX, ca. SX]:
    r"""Optimize the joint states using the provided solver.
        Args:
            args (Dict[str, Any]): Optimization arguments including constraints.
            solver (ca.Function): CasADi NLP solver.
            joint_num (Tuple[ca.SX, ca.SX]): Joint position and velocity.
            base_num (Tuple[ca.SX, ca.SX, ca.SX]): Base velocity, rotation, and position.
            link_num (Tuple[ca.SX, ...]): Link velocities (left foot, right foot, arms if wholebody=True).
            dofs (int): Degrees of freedom.
            wholebody (bool): Whether to optimize whole-body motion.
        Returns:
            Tuple[ca.SX, ca.SX]: Optimized joint position and velocity.
    """
    # ensure link_num has enough elements before accessing indices
    required_links = 4 if is_wholebody else 2
    if len(link_num) < required_links:
        raise ValueError(f"Expected {required_links} links, but got {len(link_num)}.")
    # extract joint states
    s_num = joint_num[0].reshape((-1,))
    sdot_num = joint_num[1].reshape((-1,))
    x_num = ca.vertcat(s_num, sdot_num)

    # extract base values
    vb_num = base_num[0].reshape((-1,))
    rb_num = base_num[1].reshape((-1,))
    pb_num = base_num[2].reshape((-1,))
    
    # extract lowerlegs velocities
    vleftlowerelg_num = link_num[0].reshape((-1,))
    vrightlowerleg_num = link_num[1].reshape((-1,))

    # handle wholebody case
    if is_wholebody:
        vleftarm_num = link_num[2].reshape((-1,))
        vrightarm_num = link_num[3].reshape((-1,))
        p_num = ca.vertcat(
            vb_num, pb_num, rb_num, vleftlowerelg_num, vrightlowerleg_num, vleftarm_num, vrightarm_num
        )
    else:
        p_num = ca.vertcat(
            vb_num, pb_num, rb_num, vleftlowerelg_num, vrightlowerleg_num
        )
    
    # update aruments for solver
    args['x0'] = x_num
    args['p'] = p_num
    args['lbx'] = x_num - 3
    args['ubx'] = x_num + 3

    # solve nlp
    sol = solver(x0=args['x0'], p=args['p'], lbx=args['lbx'], ubx=args['ubx'])

    # reshape optimized states
    s_opt = sol['x'][:dofs].reshape((1, dofs))
    sdot_opt = sol['x'][dofs:].reshape((1, dofs))
    return s_opt, sdot_opt


