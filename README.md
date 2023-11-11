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
* In your working process use DataSender to send data from your client to the plotting tool

# Requirements:
* python >= 3.6 (f-strings...)
* pandas >= 0.22
* xarray
* numpy
* matplotlib
* simplejson
