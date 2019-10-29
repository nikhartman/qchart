import io
import copy
from warnings import warn
from pathlib import Path
import simplejson as json
import numpy as np
from qchart.client import DataSender, NumpyJSONEncoder
from qchart.listener import listener_is_running
from qchart.config import config


def _convert_array(text: bytes) -> np.ndarray:
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def get_data_structure(indep_params, meas_params):
    """
    Return the structure of the dataset, i.e., a dictionary in the form
        {'parameter' : {
            'unit' : unit,
            'axes' : list of dependencies,
            'values' : [],
            },
        ...
        }
    """

    data_struct = {}

    # entries for independent parameters
    for param in indep_params:
        name = param.name
        unit = param.unit
        data_struct[name] = {"values": [], "unit": unit}

    # axes for measured parameters
    axes = [p.name for p in indep_params]
    for param in meas_params:
        name = param.name
        unit = param.unit
        data_struct[name] = {"values": [], "unit": unit, "axes": axes}

    return data_struct


class QcodesSubscriber(object):
    def __init__(self, dataset, subscriber_logs=False):

        # get some run info from the qcodes dataset
        self.ds = dataset
        self.db_path = Path(self.ds.path_to_db)
        self.data_id = "{} # run ID = {}".format(self.db_path, self.ds.run_id)

        self.sender = DataSender(self.data_id)

        # create log folder, if needed
        self.log_id = None
        if subscriber_logs:
            self.log_id = 0
            self.log_dir = Path(
                config["logging"]["directory"],
                "data_sent/run-{}".format(self.ds.run_id),
            )
            self.log_dir.mkdir(parents=True, exist_ok=True)

        ### get all of these from dataset ###
        # independent parameters are expected in the order (x, y)
        # measured parameters can be in any order

        self.all_param_names = []
        self.independ_params = []
        self.measured_params = []
        for param in self.ds.get_parameters():
            self.all_param_names.append(param.name)
            if param.depends_on_:
                # this parameter has setpoints
                self.measured_params.append(param)
                for ind_param_name in param.depends_on_:
                    # get the independent parameter
                    ind_param = self.ds.paramspecs[ind_param_name]
                    if ind_param not in self.independ_params:
                        # add to list, if not there
                        self.independ_params.append(ind_param)

        # if:
        # not 1 or 2 independent params
        # not >0 measured params
        # cannot plot anything for this dataset
        # mark it as useless
        # else: create data structure
        if (len(self.independ_params) in [1, 2]) and (len(self.measured_params) > 0):
            self.data_structure = get_data_structure(
                self.independ_params, self.measured_params
            )
        else:
            self.data_structure = None
            warn(f"Cannot create 1 or 2d plots for {self.data_id}")

    def _log_data_send(self):
        if self.log_id is not None:

            fp = Path(self.log_dir, "call-{}.json".format(self.log_id))
            with fp.open("w", encoding="utf-8") as f:
                json.dump(
                    dict(
                        dataStructure=self.data_structure, senderDict=self.sender.data
                    ),
                    fp=f,
                    ignore_nan=True,
                    cls=NumpyJSONEncoder,
                )
            self.log_id += 1

    def __call__(self, results, length, state):
        """ this function is called by qcodes when new data
            is written to the dataset/database """

        newData = dict(zip(self.all_param_names, list(zip(*results))))

        if self.data_structure is None:
            return

        data = copy.deepcopy(self.data_structure)

        # if the paramtype was array we'll have byte encoding now.
        # we need to revert that to numeric lists to be able to
        # send it via json.
        # we also need to expand the numeric values appropriately.

        # the rule here is the following, in agreement with the qcodes dataset:
        # * we can have either scalar/numeric values in the results, or arrays.
        # * numerics have size 1, per definition
        # * arrays can have arbitrary length, but all arrays must have the same length
        # to extract the data correctly, we must check 1) if arrays are present in the results,
        # 2) what the length is, and 3) if appropriate, expand the numerics such
        # that all results have the same length.

        # get length of arrays in the case of array parameters

        arrLen = None
        for k, v in newData.items():
            if len(v) > 0 and isinstance(v[0], bytes):
                arrLen = _convert_array(v[0]).size

        # put new data into dict
        for k, v in newData.items():

            # handling array data
            if len(v) > 0 and isinstance(v[0], bytes):
                v2 = []
                for x in v:
                    arr = _convert_array(x)
                    v2 += arr.tolist()
                data[k]["values"] = v2
            else:
                # reshaping scalar data to match arrays
                if arrLen is not None:
                    _vals = []
                    for x in v:
                        _vals += [x for i in range(arrLen)]
                    data[k]["values"] = _vals

                # scalara data
                else:
                    data[k]["values"] = list(v)

        self.sender.data["action"] = "new_data"
        self.sender.data["datasets"] = data

        self.sender.send_data()
        self._log_data_send()
