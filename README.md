# qchart (live) plotting

A simple GUI tool for plotting measurement data (e.g., for live plotting). It runs as a standalone Qt app, and data can be sent to it via a zmq socket, which makes it fairly independent of the tools used to measure.

This is a stripped down version of the work here: https://github.com/data-plottr/plottr/tree/plottr-original It contains only live plotting with the minimal number of useful widgets.

## Installation:

* Activate whatever virtual environment you like
* Navigate to the `qchart` directory
* Install a development version using `pip`.

``` pip install -e ./ ```
* Done

## Usage:
* Start the app from your script with `qchart.start_listener()`
* In your working process (i.e., ipython session, jupyter notebook, ...) use one of the client tools to package the data correctly (or do it yourself) and send it (see examples!).
* If you're using qcodes with the dataset (v2), there is a subscriber that
  can be used. See examples.

# Requirements:
* python >= 3.6 (f-strings...)
* pandas >= 0.22
* xarray
* numpy
* matplotlib
* simplejson
