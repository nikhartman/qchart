"""
qchart. A simple server application that can plot data streamed through
network sockets from other processes.

original author: Wolfgang Pfaff <wolfgangpfff@gmail.com>
qchart maintainer: Nik Hartman
"""

# TO DO:
# clean up use of logging

import sys
import time
from collections import OrderedDict
import simplejson as json
import zmq
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
from matplotlib import rcParams
from matplotlib.ticker import EngFormatter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavBar
from matplotlib.figure import Figure

from qchart.qt_base import QtCore, QtGui, QtWidgets, mkQApp
from qchart.config import config
from qchart.client import NumpyJSONEncoder

### setup LOGGER ###
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def get_log_directory():
    log_directory = Path(config['logging']['directory'])
    log_directory.mkdir(parents=True, exist_ok=True)
    return log_directory

def create_logger():
    filename = Path(get_log_directory(), 'qchart.log')
    logger = logging.getLogger(__name__)
    log_handler = RotatingFileHandler(filename, maxBytes=1048576, backupCount=5)
    log_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s %(levelname)s: '
            '%(message)s '
            '[in %(pathname)s:%(lineno)d]'
        )
    )
    log_level = logging.getLevelName(config['logging']['level'])
    logger.setLevel(log_level)
    logger.addHandler(log_handler)
    return logger

LOGGER = create_logger()

### APP ###

APPTITLE = "qchart"
TIMEFMT = "[%Y-%m-%d %H:%M:%S]"

def get_time_stamp(timeTuple=None):
    if not timeTuple:
        timeTuple = time.localtime()
    return time.strftime(TIMEFMT, timeTuple)

def get_app_title():
    return f"{APPTITLE}"


### matplotlib tools ###


def mpl_formatter():
    return EngFormatter(places=1, sep=u"")


def set_matplotlib_defaults():

    # grid
    rcParams['axes.grid'] = True
    rcParams['grid.color'] = u'b0b0b0'
    rcParams['grid.linestyle'] = ':'
    rcParams['grid.linewidth'] = 0.6
    rcParams['axes.axisbelow'] = True

    # font
    rcParams['font.family'] = 'Arial'
    rcParams['font.size'] = 8

    # line styles
    rcParams['lines.markersize'] = 4
    rcParams['lines.linestyle'] = '-'
    rcParams['lines.linewidth'] = 1

    # colormap
    rcParams['image.cmap'] = 'plasma'

    # save figs
    rcParams['savefig.transparent'] = False


def get_plot_dims(data, x, y):

    if data is None:
        return 0

    if x is None and y is None:
        return 0
    elif x is not None and y is None:
        return 1
    elif x is not None and y.size < 2:
        return 1
    elif x is not None and y is not None and y.size > 1:
        return 2


def get_axis_lims(arr):

    amin = np.nanmin(arr)
    amax = np.nanmax(arr)
    diff = np.abs(amin-amax)
    vmin = amin - np.abs(0.025*diff)
    vmax = amax + np.abs(0.025*diff)

    return vmin, vmax


def get_color_lims(data_array, cutoff_percentile=3):
    # stolen from qcodes v0.2
    """
    Get the min and max range of the provided array that excludes outliers
    following the IQR rule.
    This function computes the inter-quartile-range (IQR), defined by Q3-Q1,
    i.e. the percentiles for 75% and 25% of the destribution. The region
    without outliers is defined by [Q1-1.5*IQR, Q3+1.5*IQR].
    Args:
        data_array: numpy array of arbitrary dimension containing the
            statistical data
        cutoff_percentile: percentile of data that may maximally be clipped
            on both sides of the distribution.
    Returns:
        region limits [vmin, vmax]
    """

    z_array = data_array.flatten()
    try:
        z_max = np.nanmax(z_array)
        z_min = np.nanmin(z_array)
    except:
        return -1, 1

    z_range = z_max - zmin
    p_min, third_quarter, first_quarter, p_max = np.nanpercentile(
        z_array,
        [cutoff_percentile, 75, 25, 100 - cutoff_percentile]
    )
    inner_range = third_quarter - first_quarter

    # handle corner case of all data zero, such that IQR is zero
    # to counter numerical artifacts do not test IQR == 0, but IQR on its
    # natural scale (zrange) to be smaller than some very small number.
    # also test for zrange to be 0.0 to avoid division by 0.
    # all This is possibly to careful...
    if z_range == 0.0 or inner_range/z_range < 1e-8:
        vmin = z_min
        vmax = z_max
    else:
        vmin = max(q1 - 1.5*inner_range, z_min)
        vmax = min(q3 + 1.5*inner_range, z_max)

        # do not clip more than cutoff_percentile:
        vmin = min(vmin, p_min)
        vmax = max(vmax, p_max)
        return vmin, vmax


def centers_to_edges(arr):
    edges = (arr[1:] + arr[:-1]) / 2.
    edges = np.concatenate(([arr[0] - (edges[0] - arr[0])], edges))
    edges = np.concatenate((edges, [arr[-1] + (arr[-1] - edges[-1])]))
    return edges


def make_pcolor_grid(x_array, y_array):
    x_edges = centers_to_edges(x_array)
    y_edges = centers_to_edges(y_array)
    x_grid, y_grid = np.meshgrid(x_edges, y_edges)
    return x_grid, y_grid


### structure tools ###


def get_data_structure(data_frame):
    data_struct = {}
    data_struct['nValues'] = int(data_frame.size)
    data_struct['axes'] = OrderedDict({})

    for idx_name, idx_level in zip(data_frame.index.names, data_frame.index.levels):
        data_struct['axes'][idx_name] = {}
        data_struct['axes'][idx_name]['uniqueValues'] = idx_level.values
        data_struct['axes'][idx_name]['nValues'] = len(idx_level)

    return data_struct


def combine_dicts(dict1, dict2):
    # only works one level deep
    if dict1 != {}:
        for k in dict1.keys():
            dict1[k]['values'] += dict2[k]['values']
        return dict1
    else:
        return dict2


def dict_to_data_frames(data_dict, drop_nan=True, sort_index=True):

    dfs = []
    for param in data_dict:
        if 'axes' not in data_dict[param]:
            continue

        vals = np.array(data_dict[param]['values'], dtype=np.float)

        coord_vals = []
        coord_names = []
        for axis in data_dict[param]['axes']:
            coord_vals.append(data_dict[axis]['values'])
            unit = data_dict[axis].get('unit', '')
            axis_label = axis
            if unit != '':
                axis_label += f" ({unit})"
            coord_names.append(axis_label)
        coords = list(zip(coord_names, coord_vals))

        unit = data_dict[param].get('unit', '')
        param_label = param
        if unit != '':
            param_label += f" ({unit})"

        multi_idx = pd.MultiIndex.from_tuples(list(zip(*[v for n, v in coords])), names=coord_names)
        param_df = pd.DataFrame(vals, multi_idx, columns=[param_label])

        if sort_index:
            param_df = param_df.sort_index()

        if drop_nan:
            dfs.append(param_df.dropna())
        else:
            dfs.append(param_df)

    return dfs


def data_frame_to_xarray(df):
    """
    Convert pandas DataFrame with MultiIndex to an xarray DataArray.
    """
    arr = xr.DataArray(df)

    # remove automatically generated indices.
    col_name = list(df.columns)[0]
    for xr_idx in arr.indexes:
        idx = arr.indexes[xr_idx]
        if 'dim_' in xr_idx or xr_idx == col_name:
            if isinstance(idx, pd.MultiIndex):
                arr = arr.unstack(xr_idx)
            else:
                arr = arr.squeeze(xr_idx).drop(xr_idx)

    return arr


def append_new_data(input_frame_1, input_frame_2, sort_index=True):
    output_frame = input_frame_1.append(input_frame_2)
    if sort_index:
        output_frame = output_frame.sort_index()
    return output_frame


class MPLPlot(FigCanvas):

    def __init__(self, parent=None, width=4, height=3, dpi=150):
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        self.axes = self.fig.add_subplot(111)

        super().__init__(self.fig)
        self.setParent(parent)

    def clear_figure(self):
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        self.draw()


class DataStructure(QtWidgets.QTreeWidget):

    data_updated = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setColumnCount(2)
        self.setHeaderLabels(['Array', 'Properties'])
        self.setSelectionMode(QtWidgets.QTreeWidget.SingleSelection)


    @QtCore.Slot(dict)
    def update(self, structure):

        for key, val in structure.items():
            items = self.findItems(key, QtCore.Qt.MatchExactly)
            if len(items) == 0:
                # add a new option to the structure widget
                item = QtWidgets.QTreeWidgetItem([key, '{} points'.format(val['nValues'])])
                self.addTopLevelItem(item)

            else:
                item = items[0]
                item.setText(1, '{} points'.format(val['nValues']))

        current_selection = self.selectedItems()
        if len(current_selection) == 0:
            item = self.topLevelItem(0)
            if item:
                item.setSelected(True)


class PlotChoice(QtWidgets.QWidget):

    choice_updated = QtCore.Signal(object)

    def __init__(self, parent=None):

        super().__init__(parent)

        self.x_selection = QtWidgets.QComboBox()
        self.y_selection = QtWidgets.QComboBox()

        axis_choice_box = QtWidgets.QGroupBox('Plot axes')
        axis_choice_layout = QtWidgets.QFormLayout()
        axis_choice_layout.addRow(QtWidgets.QLabel('x axis'), self.x_selection)
        axis_choice_layout.addRow(QtWidgets.QLabel('y axis'), self.y_selection)
        axis_choice_box.setLayout(axis_choice_layout)

        self.subtract_avg_box = QtWidgets.QGroupBox('Subtract average')
        self.subtract_avg_button_group = QtWidgets.QButtonGroup()
        self.subtract_col_avg_button = QtWidgets.QRadioButton('From each column (vertical axis)')
        self.subtract_avg_button_group.addButton(self.subtract_col_avg_button, 0)
        self.subtract_row_avg_button = QtWidgets.QRadioButton('From each row (horizontal axis)')
        self.subtract_avg_button_group.addButton(self.subtract_row_avg_button, 1)
        self.subtract_avg_none_button = QtWidgets.QRadioButton('None')
        self.subtract_avg_button_group.addButton(self.subtract_avg_none_button, 2)
        self.subtract_avg_none_button.setChecked(True)
        self.subtract_avg_layout = QtWidgets.QFormLayout()
        self.subtract_avg_layout.addRow(self.subtract_col_avg_button)
        self.subtract_avg_layout.addRow(self.subtract_row_avg_button)
        self.subtract_avg_layout.addRow(self.subtract_avg_none_button)
        self.subtract_avg_box.setLayout(self.subtract_avg_layout)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(axis_choice_box)
        main_layout.addWidget(self.subtract_avg_box)

        self.emit_choice_update = False
        self.x_selection.currentTextChanged.connect(self.x_selected)
        self.y_selection.currentTextChanged.connect(self.y_selected)
        self.subtract_avg_button_group.buttonClicked.connect(self.subtract_average_changed)

        self.empty_selection_name = '<None>'
        self.axes_names = []
        self.choice_info = {}


    @QtCore.Slot(str)
    def x_selected(self, val):
        self.update_options(self.x_selection, val)

    @QtCore.Slot(str)
    def y_selected(self, val):
        self.update_options(self.y_selection, val)

    @QtCore.Slot(QtWidgets.QAbstractButton)
    def subtract_average_changed(self, button):
        self.update_options(None, None)

    def _axis_in_use(self, name):
        # for opt in self.avgSelection, self.x_selection, self.y_selection:
        for opt in self.x_selection, self.y_selection:
            if name == opt.currentText():
                return True
        return False

    def update_options(self, changed_option, new_val):
        """
        After changing the role of a data axis manually, we need to make
        sure this axis isn't used anywhere else.
        """

        for opt in self.x_selection, self.y_selection:
            if opt != changed_option and opt.currentText() == new_val:
                opt.setCurrentIndex(0)

        subtract_average = None
        if self.subtract_row_avg_button.isChecked():
            subtract_average = 'byRow'
        elif self.subtract_col_avg_button.isChecked():
            subtract_average = 'byColumn'

        self.choice_info = {
            'xAxis' : {
                'idx' : self.x_selection.currentIndex() - 1,
                'name' : self.x_selection.currentText(),
            },
            'yAxis' : {
                'idx' : self.y_selection.currentIndex() - 1,
                'name' : self.y_selection.currentText(),
            },
            'subtractAverage' : subtract_average,
        }

        if self.emit_choice_update:
            self.choice_updated.emit(self.choice_info)

    @QtCore.Slot(dict)
    def set_options(self, data_struct):
        """
        Populates the data choice widgets initially.
        """
        self.emit_choice_update = False
        self.axes_names = list(data_struct['axes'].keys())

        # Need an option that indicates that the choice is 'empty'
        while self.empty_selection_name in self.axes_names:
            self.empty_selection_name = '<' + self.empty_selection_name + '>'
        self.axes_names.insert(0, self.empty_selection_name)

        # add all options
        for opt in self.x_selection, self.y_selection:
            opt.clear()
            opt.addItems(self.axes_names)

        # see which options remain for x and y, apply the first that work
        xopts = self.axes_names.copy()
        xopts.pop(0)

        if len(xopts) > 0:
            self.x_selection.setCurrentText(xopts[0])
        if len(xopts) > 1:
            self.y_selection.setCurrentText(xopts[1])

        self.emit_choice_update = True
        self.choice_updated.emit(self.choice_info)


class PlotData(QtCore.QObject):

    data_processed = QtCore.Signal(object, object, object, bool)

    def set_data(self, data_frame, choice_info):
        self.df = data_frame
        self.choice_info = choice_info

    def process_data(self):

        try:
            xarr = data_frame_to_xarray(self.df)
            data = xarr.values[:]

            if data.size != 0:
                filled_elements = np.isfinite(data).sum()
                total_elements = data.size
                if filled_elements/total_elements < 0.05:
                    raise ValueError('grid is too sparse')


            data_shape = list(data.shape)
            exclude = [
                self.choice_info['xAxis']['idx'],
                self.choice_info['yAxis']['idx'],
            ]
            squeeze_dims = tuple(
                i for i in range(len(data_shape)) if (i not in exclude) and (data_shape[i] == 1)
            )
            plot_data = data.squeeze(squeeze_dims)
            plot_data = np.ma.masked_where(np.isnan(plot_data), plot_data)

            if plot_data.size < 1:
                LOGGER.debug('Data has size 0')
                return

            if self.choice_info['xAxis']['idx'] > -1:
                x_array = xarr.coords[self.choice_info['xAxis']['name']].values
            else:
                x_array = None

            if self.choice_info['yAxis']['idx'] > -1:
                y_array = xarr.coords[self.choice_info['yAxis']['name']].values
            else:
                y_array = None

            if self.choice_info['subtractAverage']:
                # This feature is only for 2D data
                if x_array is not None and y_array is not None:
                    # x axis, horizontal one - axis 0
                    # y axis, vertical one - axis 1
                    if self.choice_info['subtractAverage'] == 'byRow':
                        # rows / x axis / horizontal axis
                        row_means = plot_data.mean(0)
                        row_means_matrix = row_means[np.newaxis, :]
                        plot_data = plot_data - row_means_matrix
                    elif self.choice_info['subtractAverage'] == 'byColumn':
                        # columns / y axis / vertical one
                        col_means = plot_data.mean(1)
                        col_means_matrix = col_means[:, np.newaxis]
                        plot_data = plot_data - col_means_matrix

            self.data_processed.emit(plot_data, x_array, y_array, True)
            return

        except (ValueError, IndexError):
            LOGGER.debug('PlotData.process_data: No grid for the data.')
            LOGGER.debug('Fall back to scatter plot')

        if self.choice_info['xAxis']['idx'] > -1:
            x_label = self.choice_info['xAxis']['name']
            x_array = self.df.index.get_level_values(x_label).values
        else:
            x_label = None
            x_array = None

        if self.choice_info['yAxis']['idx'] > -1:
            y_label = self.choice_info['yAxis']['name']
            y_array = self.df.index.get_level_values(y_label).values
        else:
            y_label = None
            y_array = None

        plot_data = self.df.values.flatten()
        self.data_processed.emit(plot_data, x_array, y_array, False)
        return


class DataAdder(QtCore.QObject):

    data_updated = QtCore.Signal(object, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_data = None
        self.current_struct = {}
        self.new_data_dict = {}

    def set_data(self, current_data, current_struct, new_data_dict):
        self.current_data = current_data
        self.current_struct = current_struct
        self.new_data_dict = new_data_dict

    def run(self):

        new_data_frames = dict_to_data_frames(self.new_data_dict)
        data_struct = self.current_struct
        data = {}

        for data_frame in new_data_frames:
            col_name = list(data_frame.columns)[0]

            if self.current_data == {}:
                data[col_name] = data_frame
                data_struct[col_name] = get_data_structure(data_frame)
            elif col_name in self.current_data:
                data[col_name] = append_new_data(self.current_data[col_name], data_frame)
                data_struct[col_name] = get_data_structure(data[col_name])

        self.data_updated.emit(data, data_struct)


class DataWindow(QtWidgets.QMainWindow):

    data_added = QtCore.Signal(dict)
    data_activated = QtCore.Signal(dict)
    windowClosed = QtCore.Signal(str)

    def __init__(self, data_id, parent=None):
        super().__init__(parent)

        self.data_id = data_id

        search_str = 'run ID = '
        idx = self.data_id.find(search_str) + len(search_str)
        run_id = int(self.data_id[idx:].strip())
        self.setWindowTitle(f"{get_app_title()} (#{run_id})")

        self.active_dataset = None
        self.data = {}
        self.data_struct = {}
        self.adding_queue = {}
        self.plot_data_pending = False
        self.current_plot_choice_info = None

        # plot settings
        set_matplotlib_defaults()

        # data chosing widgets
        self.structure_widget = DataStructure()
        self.plot_choice = PlotChoice()
        chooser_layout = QtWidgets.QVBoxLayout()
        chooser_layout.addWidget(self.structure_widget)
        chooser_layout.addWidget(self.plot_choice)

        # plot control widgets
        self.plot = MPLPlot(width=5, height=4)
        plot_layout = QtWidgets.QVBoxLayout()
        plot_layout.addWidget(self.plot)
        plot_layout.addWidget(NavBar(self.plot, self))

        # Main layout
        self.frame = QtWidgets.QFrame()
        main_layout = QtWidgets.QHBoxLayout(self.frame)
        main_layout.addLayout(chooser_layout)
        main_layout.addLayout(plot_layout)

        # data processing threads
        self.data_adder = DataAdder()
        self.data_adder_thread = QtCore.QThread()
        self.data_adder.moveToThread(self.data_adder_thread)
        self.data_adder.data_updated.connect(self.data_from_adder)
        self.data_adder.data_updated.connect(self.data_adder_thread.quit)
        self.data_adder_thread.started.connect(self.data_adder.run)

        self.plot_data = PlotData()
        self.plot_data_thread = QtCore.QThread()
        self.plot_data.moveToThread(self.plot_data_thread)
        self.plot_data.data_processed.connect(self.update_plot)
        self.plot_data.data_processed.connect(self.plot_data_thread.quit)
        self.plot_data_thread.started.connect(self.plot_data.process_data)

        # signals/slots for data selection etc.
        self.data_added.connect(self.structure_widget.update)
        self.data_added.connect(self.update_plot_data)

        self.structure_widget.itemSelectionChanged.connect(self.activate_data)
        self.data_activated.connect(self.plot_choice.set_options)

        self.plot_choice.choice_updated.connect(self.update_plot_data)

        # activate window
        self.frame.setFocus()
        self.setCentralWidget(self.frame)
        self.activateWindow()

    @QtCore.Slot()
    def activate_data(self):
        item = self.structure_widget.selectedItems()[0]
        self.active_dataset = item.text(0)
        self.data_activated.emit(self.data_struct[self.active_dataset])

    @QtCore.Slot()
    def update_plot_data(self):
        if self.plot_data_thread.isRunning():
            self.plot_data_pending = True
        else:
            self.current_plot_choice_info = self.plot_choice.choice_info
            self.plot_data_pending = False
            self.plot_data.set_data(
                self.data[self.active_dataset],
                self.current_plot_choice_info
            )
            self.plot_data_thread.start()

    def _plot_1d_line(self, x, data):

        marker = '.'
        marker_size = 4
        marker_color = 'k'
        self.plot.axes.yaxis.set_major_formatter(mpl_formatter())

        x = x.flatten() # assume this is cool
        if (len(x) == data.shape[0]) or (len(x) == len(data)):
            self.plot.axes.plot(
                x, data,
                marker=marker,
                markerfacecolor=marker_color,
                markeredgecolor=marker_color,
                markersize=marker_size,
                )
        elif len(x) == data.shape[1]:
            self.plot.axes.plot(
                x,
                data.transpose(),
                marker=marker,
                markerfacecolor=marker_color,
                markeredgecolor=marker_color,
                markersize=marker_size,
                )
        else:
            raise ValueError('Cannot find a sensible shape for _plot_1D_line')

        try:
            xmin, xmax = get_axis_lims(x)
            self.plot.axes.set_xlim(xmin, xmax)
        except Exception as e:
            LOGGER.debug(e)

        self.plot.axes.set_xlabel(self.current_plot_choice_info['xAxis']['name'])
        self.plot.axes.set_ylabel(self.active_dataset)

    def _plot_1d_scatter(self, x, data):

        self.plot.axes.yaxis.set_major_formatter(mpl_formatter())

        x = x.flatten() # assume this is cool
        if (len(x) == data.shape[0]) or (len(x) == len(data)):
            self.plot.axes.scatter(x, data)
        elif len(x) == data.shape[1]:
            self.plot.axes.scatter(x, data.transpose())
        else:
            raise ValueError('Cannot find a sensible shape for _plot_1D_scatter')

        try:
            xmin, xmax = get_axis_lims(x)
            self.plot.axes.set_xlim(xmin, xmax)
        except Exception as e:
            LOGGER.debug(e)

        try:
            ymin, ymax = get_axis_lims(data)
            self.plot.axes.set_ylim(ymin, ymax)
        except Exception as e:
            LOGGER.debug(e)

        self.plot.axes.set_xlabel(self.current_plot_choice_info['xAxis']['name'])
        self.plot.axes.set_ylabel(self.active_dataset)

    def _plot_2d_pcolor(self, x, y, data):

        x_grid, y_grid = make_pcolor_grid(x, y)

        if (
            self.current_plot_choice_info['xAxis']['idx'] <
            self.current_plot_choice_info['yAxis']['idx']
        ):
            img = self.plot.axes.pcolormesh(x_grid, y_grid, data.transpose())
        else:
            img = self.plot.axes.pcolormesh(x_grid, y_grid, data)

        self.plot.axes.set_xlabel(self.current_plot_choice_info['xAxis']['name'])
        self.plot.axes.set_ylabel(self.current_plot_choice_info['yAxis']['name'])

        cbar = self.plot.fig.colorbar(
            img,
            format=mpl_formatter(),
        )
        cbar.set_label(self.active_dataset)

    def _plot_2d_scatter(self, x, y, data):

        img = self.plot.axes.scatter(x, y, c=data)
        try:
            xmin, xmax = get_axis_lims(x)
            self.plot.axes.set_xlim(xmin, xmax)
            ymin, ymax = get_axis_lims(y)
            self.plot.axes.set_ylim(ymin, ymax)
        except Exception as e:
            LOGGER.debug(e)

        self.plot.axes.set_xlabel(self.current_plot_choice_info['xAxis']['name'])
        self.plot.axes.set_ylabel(self.current_plot_choice_info['yAxis']['name'])

        cbar = self.plot.fig.colorbar(
            img,
            format=mpl_formatter(),
        )
        cbar.set_label(self.active_dataset)

    @QtCore.Slot(object, object, object, bool)
    def update_plot(self, data, x_array, y_array, grid_found):
        self.plot.clear_figure()

        try:
            pdims = get_plot_dims(data, x_array, y_array)
            if pdims == 0:
                raise ValueError('No data sent to DataWindow.update_plot')

            if grid_found:
                if pdims == 1:
                    self._plot_1d_line(x_array, data)
                elif pdims == 2:
                    self._plot_2d_pcolor(x_array, y_array, data)
            else:
                if pdims == 1:
                    self._plot_1d_scatter(x_array, data)
                elif pdims == 2:
                    self._plot_2d_scatter(x_array, y_array, data)

            self.plot.axes.set_title(f"{self.data_id}", size='x-small')
            self.plot.draw()

        except Exception as e:
            LOGGER.debug('Could not plot selected data')
            LOGGER.debug(f'Exception raised: {e}')

        if self.plot_data_pending:
            self.update_plot_data()

    def add_data(self, data_dict):
        """
        Here we receive new data from the listener.
        We'll use a separate thread for processing and combining (numerics might be costly).
        If the thread is already running, we'll put the new data into a queue that will
        be resolved during the next call of add_data (i.e, queue will grow until current
        adding thread is done.)
        """

        data_dict = data_dict.get('datasets', {})

        if self.data_adder_thread.isRunning():
            if self.adding_queue == {}:
                self.adding_queue = data_dict
            else:
                self.adding_queue = combine_dicts(self.adding_queue, data_dict)
        else:
            if self.adding_queue != {}:
                data_dict = combine_dicts(self.adding_queue, data_dict)

            if data_dict != {}:
                # move data to data_adder obj and start data_adder_thread
                self.data_adder.set_data(self.data, self.data_struct, data_dict)
                self.data_adder_thread.start()
                self.adding_queue = {}

    @QtCore.Slot(object, dict)
    def data_from_adder(self, data, data_struct):
        self.data = data
        self.data_struct = data_struct
        self.data_added.emit(self.data_struct)

    # clean-up
    def closeEvent(self, event):
        print(f'close {self.data_id}. event: {event}')
        self.windowClosed.emit(self.data_id)


class DataReceiver(QtCore.QObject):

    send_info = QtCore.Signal(str)
    send_data = QtCore.Signal(dict)

    def __init__(self):
        super().__init__()

        context = zmq.Context()
        port = config['network']['port']
        addr = config['network']['addr']
        self.socket = context.socket(zmq.PULL)
        self.socket.bind(f"tcp://{addr}:{port}")
        self.running = True

    @QtCore.Slot()
    def loop(self):

        self.send_info.emit("Listening...")

        while self.running:
            data_bytes = self.socket.recv()
            data = json.loads(data_bytes.decode(encoding='UTF-8'))

            if 'id' in data.keys():
                # a proper data set requires an 'id'
                data_id = data['id']
                self.send_info.emit(f'Received data for dataset: {data_id}')
                self.send_data.emit(data)
                LOGGER.debug(f'\n\t DataReceiver received: {data} \n')
            elif 'ping' in data.keys():
                # so this doesn't look like an error
                # when checking if the server is running
                self.send_info.emit(f'Received ping.')
            else:
                self.send_info.emit(
                    f'Received invalid message '
                    f'(expected DataDict or ping):\n{data}'
                )

class Logger(QtWidgets.QPlainTextEdit):
    '''
        Plain text logger that lives inside the main app window.
    '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)

    @QtCore.Slot(str)
    def add_message(self, message):
        fmt_message = f"{get_time_stamp()} {message}"
        self.appendPlainText(fmt_message)

class QchartMain(QtWidgets.QMainWindow):
    '''
        Main app window. Includes plain text box for logging
    '''

    def __init__(self, parent=None):

        LOGGER.debug('QchartMain opened.')

        super().__init__(parent)

        self.setWindowTitle(get_app_title())
        self.resize(900, 300)
        self.activateWindow()

        # layout of basic widgets
        self.logger = Logger()
        self.frame = QtWidgets.QFrame()
        layout = QtWidgets.QVBoxLayout(self.frame)
        layout.addWidget(self.logger)

        self.setCentralWidget(self.frame)
        self.frame.setFocus()

        # basic setup of the data handling
        self.data_handlers = {}

        # setting up the ZMQ thread
        self.listening_thread = QtCore.QThread()
        self.listener = DataReceiver()
        self.listener.moveToThread(self.listening_thread)

        # communication with the ZMQ thread
        self.listening_thread.started.connect(self.listener.loop)
        self.listener.send_info.connect(self.logger.add_message)
        self.listener.send_data.connect(self.process_data)

        # go!
        self.listening_thread.start()


    @QtCore.Slot(dict)
    def process_data(self, data):

        data_id = data['id']

        if data_id not in self.data_handlers:
            self.data_handlers[data_id] = DataWindow(data_id=data_id)
            self.data_handlers[data_id].show()
            self.logger.add_message(f'Started new data window for {data_id}')
            self.data_handlers[data_id].windowClosed.connect(self.data_window_closed)

        data_window = self.data_handlers[data_id]
        data_window.add_data(data)

    def closeEvent(self, event):
        self.listener.running = False
        self.listening_thread.quit()

        handler_objs = [h for d, h in self.data_handlers.items()]
        for handler in handler_objs:
            handler.close()

    @QtCore.Slot(str)
    def data_window_closed(self, data_id):
        self.logger.add_message(f'Data window closed: {data_id}.')
        self.data_handlers[data_id].close()
        del self.data_handlers[data_id]

def console_entry():
    """
    Entry point for launching the app from a console script
    """

    LOGGER.debug('Starting qchart...')

    app = mkQApp()
    main = QchartMain()
    main.show()
    sys.exit(app.exec_())
