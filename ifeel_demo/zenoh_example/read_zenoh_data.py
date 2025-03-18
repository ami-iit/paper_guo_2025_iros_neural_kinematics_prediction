import zenoh
import time
import logging
import struct
import config as cfg
import visualizer as vis
import numpy as np
import maths

def extend_joint_state_preds(s, sdot, old_joint_list, new_joint_list):
    r"""Return the extended joint predictions."""
    new_dof = len(new_joint_list)
    s_new, sdot_new = np.zeros(new_dof, ), np.zeros(new_dof, )
    joint_map = {joint: i for i, joint in enumerate(new_joint_list)}
    for i, joint in enumerate(old_joint_list):
        joint_index = joint_map[joint]
        s_new[joint_index] = s[i]
        sdot_new[joint_index] = sdot[i]
    return s_new, sdot_new


urdf_path = "./urdf/humanSubject01_66dof.urdf"
# initialize the visualizer
visualizer = vis.HumanURDFVisualizer(path=urdf_path, model_names=["gt", "pred"])
visualizer.load_model(colors=[(0.2 , 0.2, 0.2, 0.6), (1.0 , 0.2, 0.2, 0.3)])
Hb_gt = np.matrix([[1.0, 0., 0., 0.], [0., 1.0, 0., 0.],
                    [0., 0., 1.0, 0.], [0., 0., 0., 1.0]])
Hb_pred = np.matrix([[1.0, 0., 0., 0.], [0., 1.0, 0., 0.],
                        [0., 0., 1.0, 0.], [0., 0., 0., 1.0]])


# Abilita i log di Zenoh
logging.basicConfig(level=logging.DEBUG)

dicts = {}
# Funzione di callback per gestire i messaggi ricevuti
def listener(sample):
    global dicts
    #print(f"sample: {sample} \n")
    # Decodifica il payload come array di float64
    payload = sample.payload
    if len(payload) % 8 != 0:
        print("Errore: la lunghezza del payload non è un multiplo di 8 (dimensione di un float64)")
        return
    # Usa struct per decodificare i byte in float64
    floats = []
    for i in range(0, len(payload), 8):
        # Decodifica 8 byte in un float64
        value = struct.unpack('<d', payload[i:i+8])[0]
        floats.append(value)
    #print(f"Listener called! Received {sample.key_expr}: {floats} \n")
    dicts[sample.key_expr] = floats
 
# Configurazione di Zenoh
config = zenoh.Config()
config.insert_json5("mode", '"peer"')
config.insert_json5("connect/endpoints", '["tcp/localhost:7447", "tcp/localhost:7448"]')
 
# Inizializzazione della sessione Zenoh
session = zenoh.open(config)
 
# Sottoscrizione alla chiave iFeel/**
sub = session.declare_subscriber("iFeel/**", listener)
 
# Manteniamo il subscriber attivo
counter = 0
try:
    while True:
        time.sleep(1)
        for key, value in dicts.items():
            print(f"==============counter: {counter}=============")
            print(f"key: {key}")
            print(f"value: {value}")
            # get the joint positions
            jpos_step = dicts["iFeel/joints_state/positions"]
            jvel_step = dicts["iFeel/joints_state/velocities"]
            jpos_step_new, jvel_step_new = extend_joint_state_preds(
                jpos_step,
                jvel_step,
                cfg.joints_31dof,
                cfg.joints_66dof
            )
            # get the base pose
            pb = dicts["iFeel/human_state/base_position"]
            rb_as_q = dicts["iFeel/human_state/base_orientation"]
            rb_as_R = maths.quaternion_to_rotation_matrix(rb_as_q)

            # update visualizer
            Hb_gt[:3, :3] = rb_as_R.reshape((3, 3))
            Hb_gt[:3, 3] = pb.reshape((3, 1))
            visualizer.update(
                [jpos_step_new, jpos_step_new],
                [Hb_gt, Hb_gt],
                False, None
            )
            visualizer.run()
            counter += 1
except KeyboardInterrupt:
    pass

