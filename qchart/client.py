import logging
import zmq
import simplejson as json
import time
import numpy as np
from qchart.config import config

class NumpyJSONEncoder(json.JSONEncoder):
    """
    This JSON encoder adds support for serializing types that the built-in
    ``json`` module does not support out-of-the-box. See the docstring of the
    ``default`` method for the description of all conversions.

    Shamelessly stolen from qcodes.utils.helpers
    """

    def default(self, obj):
        """
        List of conversions that this encoder performs:
        * ``numpy.generic`` (all integer, floating, and other types) gets
        converted to its python equivalent using its ``item`` method (see
        ``numpy`` docs for more information,
        https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html).
        * ``numpy.ndarray`` gets converted to python list using its ``tolist``
        method.
        * Complex number (a number that conforms to ``numbers.Complex`` ABC) gets
        converted to a dictionary with fields ``re`` and ``im`` containing floating
        numbers for the real and imaginary parts respectively, and a field
        ``__dtype__`` containing value ``complex``.
        * Object with a ``_JSONEncoder`` method get converted the return value of
        that method.
        * Objects which support the pickle protocol get converted using the
        data provided by that protocol.
        * Other objects which cannot be serialized get converted to their
        string representation (suing the ``str`` function).
        """
        if isinstance(obj, np.generic) and not isinstance(obj, np.complexfloating):
            # for numpy scalars
            return obj.item()
        elif isinstance(obj, np.ndarray):
            # for numpy arrays
            return obj.tolist()
        elif isinstance(obj, numbers.Complex) and not isinstance(obj, numbers.Real):
            return {
                "__dtype__": "complex",
                "re": float(obj.real),
                "im": float(obj.imag),
            }
        elif hasattr(obj, "_JSONEncoder"):
            # Use object's custom JSON encoder
            return obj._JSONEncoder()
        else:
            try:
                s = super(NumpyJSONEncoder, self).default(obj)
            except TypeError:
                # See if the object supports the pickle protocol.
                # If so, we should be able to use that to serialize.
                if hasattr(obj, "__getnewargs__"):
                    return {
                        "__class__": type(obj).__name__,
                        "__args__": obj.__getnewargs__(),
                    }
                else:
                    # we cannot convert the object to JSON, just take a string
                    s = str(obj)
            return s


class DataSender(object):
    def __init__(self, data_id):
        self.id = data_id
        self.data = {'id': self.id, 'datasets': {}}

    def send_data(self, payload, timeout=None):

        self.data['datasets'] = payload
        jsData = json.dumps(self.data, allow_nan=True, cls=NumpyJSONEncoder)
        encData = jsData.encode(encoding="UTF-8")

        addr = config["network"]["addr"]
        port = config["network"]["port"]
        srvr = f"tcp://{addr}:{port}"

        if timeout is None:
            timeout = config["client"]["send_timeout"]

        context = zmq.Context()
        context.setsockopt(zmq.LINGER, timeout)
        socket = context.socket(zmq.PUSH)
        socket.connect(srvr)

        t0 = time.time()
        socket.send(encData)
        socket.close()
        context.term()

        if (time.time() - t0) > (timeout / 1000.0):
            print("Timeout during sending!")

        self.data['datasets'] = {}
