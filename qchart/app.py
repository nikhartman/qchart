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
from pathlib import Path
import zmq
import numpy as np
import pandas as pd
import xarray as xr

# Use PySide2 for GUI
from PyQt5.QtCore import (
    QObject, Qt, QThread,
    pyqtSignal, pyqtSlot,
)
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow,
    QFrame, QPlainTextEdit, QLabel,
    QComboBox, QFormLayout,
    QGroupBox, QHBoxLayout, QVBoxLayout,
    QWidget, QTreeWidget, QTreeWidgetItem,
    QRadioButton, QButtonGroup, QAbstractButton,
)

# Embed mpl plots into QT GUI
import matplotlib
from matplotlib import rcParams
from matplotlib.ticker import EngFormatter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavBar
from matplotlib.figure import Figure

from qchart.config import config
from qchart.client import NumpyJSONEncoder
import logging
from logging.handlers import RotatingFileHandler


def get_log_directory():
    ld = Path(config['logging']['directory'])
    ld.mkdir(parents=True, exist_ok=True)
    return ld

def createLogger():
    filename = Path(get_log_directory(), 'plots.log')
    lh = RotatingFileHandler(filename, maxBytes=1048576, backupCount=5)
    lh.setFormatter(logging.Formatter( '%(asctime)s %(levelname)s: \
            %(message)s ' '[in %(pathname)s:%(lineno)d]'))
    lggr = logging.getLogger('plots')
    lggr.setLevel(logging.getLevelName(config['logging']['level']))
    lggr.addHandler(lh)
    return lggr

logger = createLogger()


def dumpData(data):

    jdata = {}
    for key, df in data.items():
        jdata[key] = df.to_json(orient='split')

    dtime = int(1000*time.time())
    fp = Path(get_log_directory(),
              '{0:d}_{1:s}.json'.format(int(1000*time.time()), 'data'))

    with fp.open("w") as f:
        json.dump(jdata, fp=f, allow_nan=True, cls=NumpyJSONEncoder)


def dump_data_structure(ds):
    dtime = int(1000*time.time())
    fp = Path(get_log_directory(),
              '{0:d}_{1:s}.json'.format(int(1000*time.time()), 'data_struct'))
    with fp.open("w") as f:
        json.dump(ds, fp=f, allow_nan=True, cls=NumpyJSONEncoder)


### app ###

APPTITLE = "qchart"
TIMEFMT = "[%Y-%m-%d %H:%M:%S]"

def get_time_stamp(timeTuple=None):
    if not timeTuple:
        timeTuple = time.localtime()
    return time.strftime(TIMEFMT, timeTuple)

def get_app_title():
    return f"{APPTITLE}"

def mpl_formatter():
    return EngFormatter(places=1, sep=u"")

### matplotlib tools ###

def setMplDefaults():

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
    else:
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

    z = data_array.flatten()
    try:
        zmax = np.nanmax(z)
        zmin = np.nanmin(z)
    except:
        return -1, 1

    zrange = zmax-zmin
    pmin, q3, q1, pmax = np.nanpercentile(z,
                            [cutoff_percentile, 75, 25, 100-cutoff_percentile])
    IQR = q3-q1

    # handle corner case of all data zero, such that IQR is zero
    # to counter numerical artifacts do not test IQR == 0, but IQR on its
    # natural scale (zrange) to be smaller than some very small number.
    # also test for zrange to be 0.0 to avoid division by 0.
    # all This is possibly to careful...
    if zrange == 0.0 or IQR/zrange < 1e-8:
        vmin = zmin
        vmax = zmax
    else:
        vmin = max(q1 - 1.5*IQR, zmin)
        vmax = min(q3 + 1.5*IQR, zmax)
        # do not clip more than cutoff_percentile:
        vmin = min(vmin, pmin)
        vmax = max(vmax, pmax)
        return vmin, vmax


def centers2edges(arr):
    e = (arr[1:] + arr[:-1]) / 2.
    e = np.concatenate(([arr[0] - (e[0] - arr[0])], e))
    e = np.concatenate((e, [arr[-1] + (arr[-1] - e[-1])]))
    return e


def pcolorgrid(xaxis, yaxis):
    xedges = centers2edges(xaxis)
    yedges = centers2edges(yaxis)
    xx, yy = np.meshgrid(xedges, yedges)
    return xx, yy


### structure tools
def combineDicts(dict1, dict2):
    if dict1 != {}:
        for k in dict1.keys():
            dict1[k]['values'] += dict2[k]['values']
        return dict1
    else:
        return dict2


def dictToDataFrames(dataDict, dropNaN=True, sortIndex=True):

    dfs = []
    for n in dataDict:
        if 'axes' not in dataDict[n]:
            continue

        vals = np.array(dataDict[n]['values'], dtype=np.float)

        coord_vals = []
        coord_names = []
        for a in dataDict[n]['axes']:
            coord_vals.append(dataDict[a]['values'])
            m = a
            unit = dataDict[m].get('unit', '')
            if unit != '':
                m += f" ({unit})"
            coord_names.append(m)
        coords = list(zip(coord_names, coord_vals))

        mi = pd.MultiIndex.from_tuples(list(zip(*[v for n, v in coords])), names=coord_names)

        name = n
        unit = dataDict[n].get('unit', '')
        if unit != '':
            name += f" ({unit})"
        df = pd.DataFrame(vals, mi, columns=[name])

        if sortIndex:
            df = df.sort_index()

        if dropNaN:
            dfs.append(df.dropna())
        else:
            dfs.append(df)

    return dfs


def data_frame_to_xarray(df):
    """
    Convert pandas DataFrame with MultiIndex to an xarray DataArray.
    """
    arr = xr.DataArray(df)

    # remove automatically generated indices.
    col_name = list(df.columns)[0]
    for idxn in arr.indexes:
        idx = arr.indexes[idxn]
        if 'dim_' in idxn or idxn == col_name:
            if isinstance(idx, pd.MultiIndex):
                arr = arr.unstack(idxn)
            else:
                arr = arr.squeeze(idxn).drop(idxn)

    return arr


def append_new_data(input_frame_1, input_frame_2, sortIndex=True):
    output_frame = input_frame_1.append(input_frame_2)
    if sortIndex:
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


class DataStructure(QTreeWidget):

    data_updated = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setColumnCount(2)
        self.setHeaderLabels(['Array', 'Properties'])
        self.setSelectionMode(QTreeWidget.SingleSelection)


    @pyqtSlot(dict)
    def update(self, structure):

        for n, v in structure.items():
            items = self.findItems(n, Qt.MatchExactly)
            if len(items) == 0:
                # add a new option to the structure widget
                item = QTreeWidgetItem([n, '{} points'.format(v['nValues'])])
                self.addTopLevelItem(item)

            else:
                item = items[0]
                item.setText(1, '{} points'.format(v['nValues']))

        current_selection = self.selectedItems()
        if len(current_selection) == 0:
            item = self.topLevelItem(0)
            if item:
                item.setSelected(True)


class PlotChoice(QWidget):

    choiceUpdated = pyqtSignal(object)

    def __init__(self, parent=None):

        super().__init__(parent)

        # self.avgSelection = QComboBox()
        self.x_selection = QComboBox()
        self.y_selection = QComboBox()

        axis_choice_box = QGroupBox('Plot axes')
        axis_choice_layout = QFormLayout()
        axis_choice_layout.addRow(QLabel('x axis'), self.x_selection)
        axis_choice_layout.addRow(QLabel('y axis'), self.y_selection)
        axis_choice_box.setLayout(axis_choice_layout)

        self.subtract_avg_box = QGroupBox('Subtract average')
        self.subtract_avg_button_group = QButtonGroup()
        self.subtract_col_avg_button = QRadioButton('From each column (vertical axis)')
        self.subtract_avg_button_group.addButton(self.subtract_col_avg_button, 0)
        self.subtract_row_avg_button = QRadioButton('From each row (horizontal axis)')
        self.subtract_avg_button_group.addButton(self.subtract_row_avg_button, 1)
        self.subtractAverageNoneButton = QRadioButton('None')
        self.subtract_avg_button_group.addButton(self.subtractAverageNoneButton, 2)
        self.subtractAverageNoneButton.setChecked(True)
        self.subtractAverageLayout = QFormLayout()
        self.subtractAverageLayout.addRow(self.subtract_col_avg_button)
        self.subtractAverageLayout.addRow(self.subtract_row_avg_button)
        self.subtractAverageLayout.addRow(self.subtractAverageNoneButton)
        self.subtract_avg_box.setLayout(self.subtractAverageLayout)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(axisChoiceBox)
        main_layout.addWidget(self.subtract_avg_box)

        self.doEmitChoiceUpdate = False
        self.x_selection.currentTextChanged.connect(self.x_selected)
        self.y_selection.currentTextChanged.connect(self.y_selected)
        self.subtract_avg_button_group.buttonClicked.connect(self.subtractAverageChanged)


    @pyqtSlot(str)
    def x_selected(self, val):
        self.update_options(self.x_selection, val)

    @pyqtSlot(str)
    def y_selected(self, val):
        self.update_options(self.y_selection, val)

    @pyqtSlot(QAbstractButton)
    def subtractAverageChanged(self, button):
        self.update_options(None, None)


    def _axis_in_use(self, name):
        # for opt in self.avgSelection, self.x_selection, self.y_selection:
        for opt in self.x_selection, self.y_selection:
            if name == opt.currentText():
                return True
        return False


    def update_options(self, changedOption, newVal):
        """
        After changing the role of a data axis manually, we need to make
        sure this axis isn't used anywhere else.
        """

        for opt in self.x_selection, self.y_selection:
            if opt != changedOption and opt.currentText() == newVal:
                opt.setCurrentIndex(0)

        subtractAverage = None
        if self.subtract_row_avg_button.isChecked():
            subtractAverage = 'byRow'
        elif self.subtract_col_avg_button.isChecked():
            subtractAverage = 'byColumn'

        self.choiceInfo = {
            'xAxis' : {
                'idx' : self.x_selection.currentIndex() - 1,
                'name' : self.x_selection.currentText(),
            },
            'yAxis' : {
                'idx' : self.y_selection.currentIndex() - 1,
                'name' : self.y_selection.currentText(),
            },
            'subtractAverage' : subtractAverage,
        }

        if self.doEmitChoiceUpdate:
            self.choiceUpdated.emit(self.choiceInfo)

    @pyqtSlot(dict)
    def setOptions(self, data_struct):
        """
        Populates the data choice widgets initially.
        """
        self.doEmitChoiceUpdate = False
        self.axesNames = [ n for n, k in data_struct['axes'].items() ]

        # Need an option that indicates that the choice is 'empty'
        self.noSelName = '<None>'
        while self.noSelName in self.axesNames:
            self.noSelName = '<' + self.noSelName + '>'
        self.axesNames.insert(0, self.noSelName)

        # add all options
        for opt in self.x_selection, self.y_selection:
            opt.clear()
            opt.addItems(self.axesNames)

        # see which options remain for x and y, apply the first that work
        xopts = self.axesNames.copy()
        xopts.pop(0)

        if len(xopts) > 0:
            self.x_selection.setCurrentText(xopts[0])
        if len(xopts) > 1:
            self.y_selection.setCurrentText(xopts[1])

        self.doEmitChoiceUpdate = True
        self.choiceUpdated.emit(self.choiceInfo)


class PlotData(QObject):

    data_processed = pyqtSignal(object, object, object, bool)

    def __init__(self, parent=None):
        super().__init__(parent)

    def set_data(self, df, choiceInfo):
        self.df = df
        self.choiceInfo = choiceInfo

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
            squeezeExclude = [
                self.choiceInfo['xAxis']['idx'],
                self.choiceInfo['yAxis']['idx'],
            ]
            squeezeDims = tuple(
                [i for i in range(len(data_shape)) if (i not in squeezeExclude) and (data_shape[i] == 1)]
            )
            plot_data = data.squeeze(squeezeDims)
            plot_data = np.ma.masked_where(np.isnan(plot_data), plot_data)

            if plot_data.size < 1:
                logger.debug('Data has size 0')
                return

            if self.choiceInfo['xAxis']['idx'] > -1:
                xVals = xarr.coords[self.choiceInfo['xAxis']['name']].values
            else:
                xVals = None

            if self.choiceInfo['yAxis']['idx'] > -1:
                yVals = xarr.coords[self.choiceInfo['yAxis']['name']].values
            else:
                yVals = None

            if self.choiceInfo['subtractAverage']:
                # This feature is only for 2D data
                if xVals is not None and yVals is not None:
                    # x axis, horizontal one - axis 0
                    # y axis, vertical one - axis 1
                    if self.choiceInfo['subtractAverage'] == 'byRow':
                        # rows / x axis / horizontal axis
                        rowMeans = plot_data.mean(0)
                        rowMeansMatrix = rowMeans[np.newaxis, :]
                        plot_data = plot_data - rowMeansMatrix
                    elif self.choiceInfo['subtractAverage'] == 'byColumn':
                        # columns / y axis / vertical one
                        columnMeans = plot_data.mean(1)
                        columnMeansMatrix = columnMeans[:, np.newaxis]
                        plot_data = plot_data - columnMeansMatrix

            self.data_processed.emit(plot_data, xVals, yVals, True)
            return

        except (ValueError, IndexError):
            logger.debug('PlotData.process_data: No grid for the data.')
            logger.debug('Fall back to scatter plot')

        if self.choiceInfo['xAxis']['idx'] > -1:
            xVar = self.choiceInfo['xAxis']['name']
            xVals = self.df.index.get_level_values(xVar).values
        else:
            xVar = None
            xVals = None

        if self.choiceInfo['yAxis']['idx'] > -1:
            yVar = self.choiceInfo['yAxis']['name']
            yVals = self.df.index.get_level_values(yVar).values
        else:
            yVar = None
            yVals = None

        plot_data = self.df.values.flatten()
        self.data_processed.emit(plot_data, xVals, yVals, False)
        return


class DataAdder(QObject):

    data_updated = pyqtSignal(object, dict)

    def _get_data_structure(self, data_frame):
        data_struct = {}
        data_struct['nValues'] = int(data_frame.size)
        data_struct['axes'] = OrderedDict({})
        # logger.debug(ds['axes'])

        for idx_name, idx_level in zip(data_frame.index.names, data_frame.index.levels):
            data_struct['axes'][idx_name] = {}
            data_struct['axes'][idx_name]['uniqueValues'] = idx_level.values
            data_struct['axes'][idx_name]['nValues'] = len(idx_level)

        return data_struct

    def set_data(self, current_data, current_struct, new_data_dict):
        self.current_data = current_data
        self.current_struct = current_struct
        self.new_data_dict = new_data_dict

    def run(self):

        # logger.debug('step 3.2 = DataAdder.run to add queued data to existing')
        new_data_frames = dictToDataFrames(self.new_data_dict)

        data_struct = self.current_struct
        data = {}

        for data_frame in new_data_frames:
            col_name = list(data_frame.columns)[0]

            if self.current_data == {}:
                data[col_name] = data_frame
                data_struct[col_name] = self._get_data_structure(data_frame)
            elif col_name in self.current_data:
                data[col_name] = append_new_data(self.current_data[col_name], data_frame)
                data_struct[col_name] = self._get_data_structure(data[col_name])

        self.data_updated.emit(data, data_struct)


class DataWindow(QMainWindow):

    data_added = pyqtSignal(dict)
    data_activated = pyqtSignal(dict)
    windowClosed = pyqtSignal(str)

    def __init__(self, data_id, parent=None):
        super().__init__(parent)

        self.data_id = data_id

        search_str = 'run ID = '
        idx = self.data_id.find(search_str) + len(search_str)
        run_id = int(self.data_id[idx:].strip())
        self.setWindowTitle(f"{get_app_title()} (#{run_id})")

        self.data = {}     # this is going to be the full dataset
        self.data_struct = {}

        self.addingQueue = {}
        self.currentlyProcessingPlotData = False
        self.pendingPlotData = False

        # plot settings
        setMplDefaults()

        # data chosing widgets
        self.structure_widget = DataStructure()
        self.plotChoice = PlotChoice()
        chooser_layout = QVBoxLayout()
        chooser_layout.addWidget(self.structure_widget)
        chooser_layout.addWidget(self.plotChoice)

        # plot control widgets
        self.plot = MPLPlot(width=5, height=4)
        plotLayout = QVBoxLayout()
        plotLayout.addWidget(self.plot)
        plotLayout.addWidget(NavBar(self.plot, self))

        # Main layout
        self.frame = QFrame()
        main_layout = QHBoxLayout(self.frame)
        main_layout.addLayout(chooser_layout)
        main_layout.addLayout(plotLayout)

        # data processing threads
        self.dataAdder = DataAdder()
        self.data_adder_thread = QThread()
        self.dataAdder.moveToThread(self.data_adder_thread)
        self.dataAdder.data_updated.connect(self.data_from_adder)
        self.dataAdder.data_updated.connect(self.data_adder_thread.quit)
        self.data_adder_thread.started.connect(self.dataAdder.run)

        self.plot_data = PlotData()
        self.plot_data_thread = QThread()
        self.plot_data.moveToThread(self.plot_data_thread)
        self.plot_data.data_processed.connect(self.updatePlot)
        self.plot_data.data_processed.connect(self.plot_data_thread.quit)
        self.plot_data_thread.started.connect(self.plot_data.process_data)

        # signals/slots for data selection etc.
        self.data_added.connect(self.structure_widget.update)
        self.data_added.connect(self.update_plot_data)

        self.structure_widget.itemSelectionChanged.connect(self.activate_data)
        self.data_activated.connect(self.plotChoice.setOptions)

        self.plotChoice.choiceUpdated.connect(self.update_plot_data)

        # activate window
        self.frame.setFocus()
        self.setCentralWidget(self.frame)
        self.activateWindow()

    @pyqtSlot()
    def activate_data(self):
        item = self.structure_widget.selectedItems()[0]
        self.activeDataSet = item.text(0)
        self.data_activated.emit(self.data_struct[self.activeDataSet])

    @pyqtSlot()
    def update_plot_data(self):
        if self.plot_data_thread.isRunning():
            self.pendingPlotData = True
        else:
            self.currentPlotChoiceInfo = self.plotChoice.choiceInfo
            self.pendingPlotData = False
            self.plot_data.set_data(self.data[self.activeDataSet], self.currentPlotChoiceInfo)
            self.plot_data_thread.start()


    def _plot1D_line(self, x, data):

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
            logger.debug(e)

        self.plot.axes.set_xlabel(self.currentPlotChoiceInfo['xAxis']['name'])
        self.plot.axes.set_ylabel(self.activeDataSet)

    def _plot1D_scatter(self, x, data):

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
            logger.debug(e)

        try:
            ymin, ymax = get_axis_lims(data)
            self.plot.axes.set_ylim(ymin, ymax)
        except Exception as e:
            logger.debug(e)

        self.plot.axes.set_xlabel(self.currentPlotChoiceInfo['xAxis']['name'])
        self.plot.axes.set_ylabel(self.activeDataSet)

    def _plot2D_pcolor(self, x, y, data):

        xx, yy = pcolorgrid(x, y)

        if self.currentPlotChoiceInfo['xAxis']['idx'] < self.currentPlotChoiceInfo['yAxis']['idx']:
            im = self.plot.axes.pcolormesh(xx, yy, data.transpose())
        else:
            im = self.plot.axes.pcolormesh(xx, yy, data)

        self.plot.axes.set_xlabel(self.currentPlotChoiceInfo['xAxis']['name'])
        self.plot.axes.set_ylabel(self.currentPlotChoiceInfo['yAxis']['name'])

        cb = self.plot.fig.colorbar(
            im,
            format=mpl_formatter(),
        )
        cb.set_label(self.activeDataSet)

    def _plot2D_scatter(self, x, y, data):

        sc = self.plot.axes.scatter(x, y, c=data)
        try:
            xmin, xmax = get_axis_lims(x)
            self.plot.axes.set_xlim(xmin, xmax)
            ymin, ymax = get_axis_lims(y)
            self.plot.axes.set_ylim(ymin, ymax)
        except Exception as e:
            logger.debug(e)

        self.plot.axes.set_xlabel(self.currentPlotChoiceInfo['xAxis']['name'])
        self.plot.axes.set_ylabel(self.currentPlotChoiceInfo['yAxis']['name'])

        cb = self.plot.fig.colorbar(
            im,
            format=mpl_formatter(),
        )
        cb.set_label(self.activeDataSet)

    @pyqtSlot(object, object, object, bool)
    def updatePlot(self, data, x, y, grid_found):
        self.plot.clear_figure()

        try:
            pdims = get_plot_dims(data, x, y)
            if pdims == 0:
                raise ValueError('No data sent to DataWindow.updatePlot')

            if grid_found:
                if pdims == 1:
                    self._plot1D_line(x, data)
                elif pdims == 2:
                    try:
                        self._plot2D_pcolor(x, y, data)
                    except Exception as e:
                        logger.debug('2D plot -- {}'.format(e))
            else:
                if pdims == 1:
                    self._plot1D_scatter(x, data)
                elif pdims == 2:
                    self._plot2D_scatter(x, y, data)

            self.plot.axes.set_title(f"{self.data_id}", size='x-small')
            self.plot.draw()

        except Exception as e:
            logging.debug('Could not plot selected data')
            logging.debug('Exception raised: {}'.format(e))

        if self.pendingPlotData:
            self.update_plot_data()

    def addData(self, dataDict):
        """
        Here we receive new data from the listener.
        We'll use a separate thread for processing and combining (numerics might be costly).
        If the thread is already running, we'll put the new data into a queue that will
        be resolved during the next call of addData (i.e, queue will grow until current
        adding thread is done.)
        """

        dataDict = dataDict.get('datasets', {})

        if self.data_adder_thread.isRunning():
            # logger.debug('step 2.1 = DataWindow.addData add data to queue')
            if self.addingQueue == {}:
                self.addingQueue = dataDict
            else:
                self.addingQueue = combineDicts(self.addingQueue, dataDict)
        else:
            if self.addingQueue != {}:
                dataDict = combineDicts(self.addingQueue, dataDict)

            if dataDict != {}:
                # move data to dataAdder obj and start data_adder_thread
                self.dataAdder.set_data(self.data, self.data_struct, dataDict)
                self.data_adder_thread.start()
                self.addingQueue = {}

    @pyqtSlot(object, dict)
    def data_from_adder(self, data, data_struct):
        self.data = data
        self.data_struct = data_struct
        self.data_added.emit(self.data_struct)

    # clean-up
    def close_event(self, event):
        self.windowClosed.emit(self.data_id)


class DataReceiver(QObject):

    sendInfo = pyqtSignal(str)
    sendData = pyqtSignal(dict)

    def __init__(self):
        super().__init__()

        context = zmq.Context()
        port = config['network']['port']
        addr = config['network']['addr']
        self.socket = context.socket(zmq.PULL)
        self.socket.bind(f"tcp://{addr}:{port}")
        self.running = True

    @pyqtSlot()
    def loop(self):
        self.sendInfo.emit("Listening...")

        while self.running:
            dataBytes = self.socket.recv()
            data = json.loads(dataBytes.decode(encoding='UTF-8'))

            if 'id' in data.keys():
                # a proper data set requires an 'id'
                data_id = data['id']
                self.sendInfo.emit(f'Received data for dataset: {data_id}')
                self.sendData.emit(data)
                logger.debug(f'\n\t DataReceiver received: {data} \n')
                continue
            elif 'ping' in data.keys():
                # so this doesn't look like an error
                # when checking if the server is running
                self.sendInfo.emit(f'Received ping.')
                continue
            else:
                self.sendInfo.emit(f'Received invalid message (expected DataDict or ping):\n{data}')
                continue

class Logger(QPlainTextEdit):
    '''
        Plain text logger that lives inside the main app window.
    '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)

    @pyqtSlot(str)
    def addMessage(self, message):
        fmt_message = "{} {}".format(get_time_stamp(), message)
        self.appendPlainText(fmt_message)

class QchartMain(QMainWindow):
    '''
        Main app window. Includes plain text box for logging
    '''

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle(get_app_title())
        self.resize(900, 300)
        self.activateWindow()

        # layout of basic widgets
        self.logger = Logger()
        self.frame = QFrame()
        layout = QVBoxLayout(self.frame)
        layout.addWidget(self.logger)

        self.setCentralWidget(self.frame)
        self.frame.setFocus()

        # basic setup of the data handling
        self.data_handlers = {}

        # setting up the ZMQ thread
        self.listening_thread = QThread()
        self.listener = DataReceiver()
        self.listener.moveToThread(self.listening_thread)

        # communication with the ZMQ thread
        self.listening_thread.started.connect(self.listener.loop)
        self.listener.sendInfo.connect(self.logger.addMessage)
        self.listener.sendData.connect(self.process_data)

        # go!
        self.listening_thread.start()


    @pyqtSlot(dict)
    def process_data(self, data):

        data_id = data['id']

        if data_id not in self.data_handlers:
            self.data_handlers[data_id] = DataWindow(data_id=data_id)
            self.data_handlers[data_id].show()
            self.logger.addMessage(f'Started new data window for {data_id}')
            self.data_handlers[data_id].windowClosed.connect(self.data_window_closed)

        data_window = self.data_handlers[data_id]
        data_window.addData(data)

    def close_event(self, event):
        self.listener.running = False
        self.listening_thread.quit()

        handler_objs = [h for d, h in self.data_handlers.items()]
        for handler in handler_objs:
            handler.close()

    @pyqtSlot(str)
    def data_window_closed(self, data_id):
        self.data_handlers[data_id].close()
        del self.data_handlers[data_id]

def console_entry():
    """
    Entry point for launching the app from a console script
    """

    logger.debug('Starting qchart...')

    app = QApplication(sys.argv)
    main = QchartMain()
    main.show()
    sys.exit(app.exec_())
