import yarp
from typing import List, Tuple

class Yarper:
    def __init__(self, 
                 gt_button: List[bool], gt_names: List[str], 
                 pred_button: list[bool], pred_names: List[str]):
        r""""
        YARP interface for ground truth (gt) and prediction (pred) data streaming.

        Args:
            gt_button (List[bool]): [s, sdot] - decides which ports to open for ground truth data.
            gt_names (List[str]): Port names for ground truth data.
            pred_button (List[bool]): [s, sdot] - decides which ports to open for prediction data.
            pred_names (List[str]): Port names for prediction data.
        """
        # initialize yarp network
        yarp.Network.init()
        if not yarp.Network.checkNetwork():
            raise RuntimeError("[ERROR] Unable to connect to a YARP Network.")
        print(f"[INFO] Running a YARP Network happily.")

        self.gt_button = gt_button
        self.gt_names = gt_names
        self.pred_button = pred_button
        self.pred_names = pred_names

        self.ports = {}

    def _open_port(self, port_name: str) -> yarp.BufferedPortBottle:
        """Helper function to create and open a single yarp port."""
        port = yarp.BufferedPortBottle()
        port.open(port_name)
        self.ports[port_name] = port
        return port

    def open(self):
        """Opens YARP ports based on the provided configuration."""
        if self.gt_button[0]:
            self.s_gt_port = self._open_port(self.gt_names[0])
        if self.gt_button[1]:
            self.sdot_gt_port = self._open_port(self.gt_names[1])

        if self.pred_button[0]:
            self.s_pred_port = self._open_port(self.pred_names[0])
        if self.pred_button[1]:
            self.sdot_pred_port = self._open_port(self.pred_names[1])

    def _write_to_port(self, port: yarp.BufferedPortBottle, data: List[float]):
        """Helper function to prepare and write data to a yarp port."""
        bottle = port.prepare()
        bottle.clear()
        for value in data:
            bottle.addFlaot64(value)
        port.write()

    def write(self, gt: List[List[float]], pred: List[List[float]]):
        r"""
        Publishes ground truth (gt) and predicted (pred) data to YARP.

        Args:
            gt (List[List[float]]): [s_gt, sdot_gt] - ground truth values.
            pred (List[List[float]]): [s_pred, sdot_pred] - predicted values.
        """
        # publish s gt data
        if self.gt_button[0]:
            self._write_to_port(self.s_gt_port, gt[0])

        # publish s pred data
        if self.pred_button[0]:
            self._write_to_port(self.s_pred_port, pred[0].flatten()) # avoid nested loops

        # publish sdot gt data
        if self.gt_button[1]:
            self._write_to_port(self.sdot_gt_port, gt[1])

        # publish sdot pred data
        if self.pred_button[1]:
            self._write_to_port(self.sdot_pred_port, pred[1].flatten())

    def close(self):
        """Close all open yarp ports."""
        for port_name, port in self.ports.item():
            port.close()
            print(f"[INFO] Closing port: {port_name}...")
        self.ports.clear()