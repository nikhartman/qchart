"""
qchart. A simple server application that can plot data streamed through
network sockets from other processes.

OG author: Wolfgang Pfaff <wolfgangpfff@gmail.com>
"""

# TO DO:
#
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
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavBar
from matplotlib.figure import Figure

from qchart.config import config
from qchart.client import NumpyJSONEncoder
import logging
from logging.handlers import RotatingFileHandler


def getLogDirectory():
    ld = Path(config['logging']['directory'])
    ld.mkdir(parents=True, exist_ok=True)
    return ld

def createLogger():
    filename = Path(getLogDirectory(), 'plots.log')
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
    fp = Path(getLogDirectory(),
              '{0:d}_{1:s}.json'.format(int(1000*time.time()), 'data'))

    with fp.open("w") as f:
        json.dump(jdata, fp=f, allow_nan=True, cls=NumpyJSONEncoder)


def dumpDataStructure(ds):
    dtime = int(1000*time.time())
    fp = Path(getLogDirectory(),
              '{0:d}_{1:s}.json'.format(int(1000*time.time()), 'dataStructure'))
    with fp.open("w") as f:
        json.dump(ds, fp=f, allow_nan=True, cls=NumpyJSONEncoder)


### app ###

APPTITLE = "qchart"
TIMEFMT = "[%Y-%m-%d %H:%M:%S]"

def getTimestamp(timeTuple=None):
    if not timeTuple:
        timeTuple = time.localtime()
    return time.strftime(TIMEFMT, timeTuple)

def getAppTitle():
    return f"{APPTITLE}"

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


def dataFrameToXArray(df):
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


def appendNewData(df1, df2, sortIndex=True):
    df = df1.append(df2)
    if sortIndex:
        df = df.sort_index()
    return df


# def insertNewData(dfFull, df):
#     idx = dfFull.index.searchsorted(df.index)
#     dfFull.iloc[idx] = df
#     return dfFull


class MPLPlot(FigCanvas):

    def __init__(self, parent=None, width=4, height=3, dpi=150):
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        # TODO: option for multiple subplots
        self.axes = self.fig.add_subplot(111)

        super().__init__(self.fig)
        self.setParent(parent)

    def clearFig(self):
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        self.draw()


class DataStructure(QTreeWidget):

    dataUpdated = pyqtSignal(dict)

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

        curSelection = self.selectedItems()
        if len(curSelection) == 0:
            item = self.topLevelItem(0)
            if item:
                item.setSelected(True)


class PlotChoice(QWidget):

    choiceUpdated = pyqtSignal(object)

    def __init__(self, parent=None):

        super().__init__(parent)

        # self.avgSelection = QComboBox()
        self.xSelection = QComboBox()
        self.ySelection = QComboBox()

        axisChoiceBox = QGroupBox('Plot axes')
        axisChoiceLayout = QFormLayout()
        axisChoiceLayout.addRow(QLabel('x axis'), self.xSelection)
        axisChoiceLayout.addRow(QLabel('y axis'), self.ySelection)
        axisChoiceBox.setLayout(axisChoiceLayout)

        self.subtractAverageBox = QGroupBox('Subtract average')
        self.subtractAverageButtonGroup = QButtonGroup()
        self.subtractAverageByColumnButton = QRadioButton('From each column (vertical axis)')
        self.subtractAverageButtonGroup.addButton(self.subtractAverageByColumnButton, 0)
        self.subtractAverageByRowButton = QRadioButton('From each row (horizontal axis)')
        self.subtractAverageButtonGroup.addButton(self.subtractAverageByRowButton, 1)
        self.subtractAverageNoneButton = QRadioButton('None')
        self.subtractAverageButtonGroup.addButton(self.subtractAverageNoneButton, 2)
        self.subtractAverageNoneButton.setChecked(True)
        self.subtractAverageLayout = QFormLayout()
        self.subtractAverageLayout.addRow(self.subtractAverageByColumnButton)
        self.subtractAverageLayout.addRow(self.subtractAverageByRowButton)
        self.subtractAverageLayout.addRow(self.subtractAverageNoneButton)
        self.subtractAverageBox.setLayout(self.subtractAverageLayout)

        mainLayout = QVBoxLayout(self)
        mainLayout.addWidget(axisChoiceBox)
        mainLayout.addWidget(self.subtractAverageBox)

        self.doEmitChoiceUpdate = False
        self.xSelection.currentTextChanged.connect(self.xSelected)
        self.ySelection.currentTextChanged.connect(self.ySelected)
        self.subtractAverageButtonGroup.buttonClicked.connect(self.subtractAverageChanged)


    @pyqtSlot(str)
    def xSelected(self, val):
        self.updateOptions(self.xSelection, val)

    @pyqtSlot(str)
    def ySelected(self, val):
        self.updateOptions(self.ySelection, val)

    @pyqtSlot(QAbstractButton)
    def subtractAverageChanged(self, button):
        self.updateOptions(None, None)


    def _isAxisInUse(self, name):
        # for opt in self.avgSelection, self.xSelection, self.ySelection:
        for opt in self.xSelection, self.ySelection:
            if name == opt.currentText():
                return True
        return False


    def updateOptions(self, changedOption, newVal):
        """
        After changing the role of a data axis manually, we need to make
        sure this axis isn't used anywhere else.
        """

        for opt in self.xSelection, self.ySelection:
            if opt != changedOption and opt.currentText() == newVal:
                opt.setCurrentIndex(0)

        subtractAverage = None
        if self.subtractAverageByRowButton.isChecked():
            subtractAverage = 'byRow'
        elif self.subtractAverageByColumnButton.isChecked():
            subtractAverage = 'byColumn'

        self.choiceInfo = {
            'xAxis' : {
                'idx' : self.xSelection.currentIndex() - 1,
                'name' : self.xSelection.currentText(),
            },
            'yAxis' : {
                'idx' : self.ySelection.currentIndex() - 1,
                'name' : self.ySelection.currentText(),
            },
            'subtractAverage' : subtractAverage,
        }

        if self.doEmitChoiceUpdate:
            self.choiceUpdated.emit(self.choiceInfo)

    @pyqtSlot(dict)
    def setOptions(self, dataStructure):
        """
        Populates the data choice widgets initially.
        """
        self.doEmitChoiceUpdate = False
        self.axesNames = [ n for n, k in dataStructure['axes'].items() ]

        # Need an option that indicates that the choice is 'empty'
        self.noSelName = '<None>'
        while self.noSelName in self.axesNames:
            self.noSelName = '<' + self.noSelName + '>'
        self.axesNames.insert(0, self.noSelName)

        # add all options
        for opt in self.xSelection, self.ySelection:
            opt.clear()
            opt.addItems(self.axesNames)

        # see which options remain for x and y, apply the first that work
        xopts = self.axesNames.copy()
        xopts.pop(0)

        if len(xopts) > 0:
            self.xSelection.setCurrentText(xopts[0])
        if len(xopts) > 1:
            self.ySelection.setCurrentText(xopts[1])

        self.doEmitChoiceUpdate = True
        self.choiceUpdated.emit(self.choiceInfo)


class PlotData(QObject):

    dataProcessed = pyqtSignal(object, object, object, bool)

    def __init__(self, parent=None):
        super().__init__(parent)

    def setData(self, df, choiceInfo):
        self.df = df
        self.choiceInfo = choiceInfo

    def processData(self):

        try:

            xarr = dataFrameToXArray(self.df)
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
            plotData = data.squeeze(squeezeDims)
            plotData = np.ma.masked_where(np.isnan(plotData), plotData)

            if plotData.size < 1:
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
                        rowMeans = plotData.mean(0)
                        rowMeansMatrix = rowMeans[np.newaxis, :]
                        plotData = plotData - rowMeansMatrix
                    elif self.choiceInfo['subtractAverage'] == 'byColumn':
                        # columns / y axis / vertical one
                        columnMeans = plotData.mean(1)
                        columnMeansMatrix = columnMeans[:, np.newaxis]
                        plotData = plotData - columnMeansMatrix

            self.dataProcessed.emit(plotData, xVals, yVals, True)
            return

        except (ValueError, IndexError):
            logger.debug('PlotData.processData: No grid for the data.')
            logger.debug('Fall back to scatter plot')

        if self.choiceInfo['xAxis']['idx'] > -1:
            xVar = self.choiceInfo['xAxis']['name']
            xVals = self.df.index.get_level_values(xVar).values
        else:
            xVar = None
            xVals = None

        if self.choiceInfo['yAxis']['idx'] > -1:
            yVar = self.choiceInfo['yAxis']['name']
            yVals =self.df.index.get_level_values(yVar).values
        else:
            yVar = None
            yVals = None

        plotData = self.df.values.flatten()
        self.dataProcessed.emit(plotData, xVals, yVals, False)
        return


class DataAdder(QObject):

    dataUpdated = pyqtSignal(object, dict)

    def _getDataStructure(self, df):
        ds = {}
        ds['nValues'] = int(df.size)
        ds['axes'] = OrderedDict({})
        # logger.debug(ds['axes'])

        for m, lvls in zip(df.index.names, df.index.levels):
            ds['axes'][m] = {}
            ds['axes'][m]['uniqueValues'] = lvls.values
            ds['axes'][m]['nValues'] = len(lvls)

        return ds

    def setData(self, curData, curStructure, newDataDict):
        self.curData = curData
        self.curStructure = curStructure
        self.newDataDict = newDataDict

    def run(self):

        # logger.debug('step 3.2 = DataAdder.run to add queued data to existing')
        newDataFrames = dictToDataFrames(self.newDataDict)

        dataStructure = self.curStructure
        data = {}

        for df in newDataFrames:
            col_name = list(df.columns)[0]

            if self.curData == {}:
                data[col_name] = df
                dataStructure[col_name] = self._getDataStructure(df)
            elif col_name in self.curData:
                data[col_name] = appendNewData(self.curData[col_name], df)
                dataStructure[col_name] = self._getDataStructure(data[col_name])

        self.dataUpdated.emit(data, dataStructure)


class DataWindow(QMainWindow):

    dataAdded = pyqtSignal(dict)
    dataActivated = pyqtSignal(dict)
    windowClosed = pyqtSignal(str)

    def __init__(self, dataId, parent=None):
        super().__init__(parent)

        self.dataId = dataId
        self.setWindowTitle(getAppTitle() + f" ({dataId})")
        self.data = {}     # this is going to be the full dataset
        self.dataStructure = {}

        self.addingQueue = {}
        self.currentlyProcessingPlotData = False
        self.pendingPlotData = False

        # plot settings
        setMplDefaults()

        # data chosing widgets
        self.structureWidget = DataStructure()
        self.plotChoice = PlotChoice()
        chooserLayout = QVBoxLayout()
        chooserLayout.addWidget(self.structureWidget)
        chooserLayout.addWidget(self.plotChoice)

        # plot control widgets
        self.plot = MPLPlot(width=5, height=4)
        plotLayout = QVBoxLayout()
        plotLayout.addWidget(self.plot)
        plotLayout.addWidget(NavBar(self.plot, self))

        # Main layout
        self.frame = QFrame()
        mainLayout = QHBoxLayout(self.frame)
        mainLayout.addLayout(chooserLayout)
        mainLayout.addLayout(plotLayout)

        # data processing threads
        self.dataAdder = DataAdder()
        self.dataAdderThread = QThread()
        self.dataAdder.moveToThread(self.dataAdderThread)
        self.dataAdder.dataUpdated.connect(self.dataFromAdder)
        self.dataAdder.dataUpdated.connect(self.dataAdderThread.quit)
        self.dataAdderThread.started.connect(self.dataAdder.run)

        self.plotData = PlotData()
        self.plotDataThread = QThread()
        self.plotData.moveToThread(self.plotDataThread)
        self.plotData.dataProcessed.connect(self.updatePlot)
        self.plotData.dataProcessed.connect(self.plotDataThread.quit)
        self.plotDataThread.started.connect(self.plotData.processData)

        # signals/slots for data selection etc.
        self.dataAdded.connect(self.structureWidget.update)
        # self.dataAdded.connect(self.updatePlotChoices)
        self.dataAdded.connect(self.updatePlotData)

        self.structureWidget.itemSelectionChanged.connect(self.activateData)
        self.dataActivated.connect(self.plotChoice.setOptions)

        self.plotChoice.choiceUpdated.connect(self.updatePlotData)

        # activate window
        self.frame.setFocus()
        self.setCentralWidget(self.frame)
        self.activateWindow()

    @pyqtSlot()
    def activateData(self):
        item = self.structureWidget.selectedItems()[0]
        self.activeDataSet = item.text(0)
        self.dataActivated.emit(self.dataStructure[self.activeDataSet])

    @pyqtSlot()
    def updatePlotData(self):
        if self.plotDataThread.isRunning():
            self.pendingPlotData = True
        else:
            self.currentPlotChoiceInfo = self.plotChoice.choiceInfo
            self.pendingPlotData = False
            # dumpData(self.data)
            self.plotData.setData(self.data[self.activeDataSet], self.currentPlotChoiceInfo)
            self.plotDataThread.start()


    def _plot1D_line(self, x, data):

        marker = '.'
        marker_size = 4
        marker_color = 'k'
        x = x.flatten() # assume this is cool
        if (len(x)==data.shape[0]) or (len(x)==len(data)):
            self.plot.axes.plot(
                x, data,
                marker = marker,
                markerfacecolor=marker_color,
                markeredgecolor=marker_color,
                markersize=marker_size,
                )
        elif (len(x)==data.shape[1]):
            self.plot.axes.plot(
                x, data.transpose(),
                marker = marker,
                markerfacecolor=marker_color,
                markeredgecolor=marker_color,
                markersize=marker_size,
                )
        else:
            raise ValueError('Cannot find a sensible shape for _plot_1D_line')

        try:
            xmin, xmax = get_axis_lims(x)
            self.plot.axes.set_xlim(xmin, xmax)
        except:
            pass

        self.plot.axes.set_xlabel(self.currentPlotChoiceInfo['xAxis']['name'])
        self.plot.axes.set_ylabel(self.activeDataSet)

    def _plot1D_scatter(self, x, data):

        x = x.flatten() # assume this is cool
        if (len(x)==data.shape[0]) or (len(x)==len(data)):
            self.plot.axes.scatter(x, data)
        elif (len(x)==data.shape[1]):
            self.plot.axes.scatter(x, data.transpose())
        else:
            raise ValueError('Cannot find a sensible shape for _plot_1D_scatter')

        try:
            xmin, xmax = get_axis_lims(x)
            self.plot.axes.set_xlim(xmin, xmax)
        except:
            pass

        try:
            ymin, ymax = get_axis_lims(data)
            self.plot.axes.set_ylim(ymin, ymax)
        except:
            pass

        self.plot.axes.set_xlabel(self.currentPlotChoiceInfo['xAxis']['name'])
        self.plot.axes.set_ylabel(self.activeDataSet)

    def _plot2D_pcolor(self, x, y, data):

        xx, yy = pcolorgrid(x, y)

        if self.currentPlotChoiceInfo['xAxis']['idx'] < self.currentPlotChoiceInfo['yAxis']['idx']:
            im = self.plot.axes.pcolormesh(xx, yy, data.transpose())
        else:
            im = self.plot.axes.pcolormesh(xx, yy, data)

        cb = self.plot.fig.colorbar(im)
        self.plot.axes.set_xlabel(self.currentPlotChoiceInfo['xAxis']['name'])
        self.plot.axes.set_ylabel(self.currentPlotChoiceInfo['yAxis']['name'])
        cb.set_label(self.activeDataSet)

    def _plot2D_scatter(self, x, y, data):

        sc = self.plot.axes.scatter(x, y, c=data)
        try:
            xmin, xmax = get_axis_lims(x)
            self.plot.axes.set_xlim(xmin, xmax)
            ymin, ymax = get_axis_lims(y)
            self.plot.axes.set_ylim(ymin, ymax)
        except:
            pass

        cb = self.plot.fig.colorbar(sc)
        self.plot.axes.set_xlabel(self.currentPlotChoiceInfo['xAxis']['name'])
        self.plot.axes.set_ylabel(self.currentPlotChoiceInfo['yAxis']['name'])
        cb.set_label(self.activeDataSet)

    @pyqtSlot(object, object, object, bool)
    def updatePlot(self, data, x, y, grid_found):
        self.plot.clearFig()

        try:
            pdims = get_plot_dims(data, x, y)
            if pdims==0:
                raise ValueError('No data sent to DataWindow.updatePlot')
            else:
                if grid_found:
                    if pdims==1:
                        self._plot1D_line(x, data)
                    elif pdims==2:
                        try:
                            self._plot2D_pcolor(x, y, data)
                        except Exception as e:
                            logger.debug('2D plot -- {}'.format(e))
                else:
                    if pdims==1:
                        self._plot1D_scatter(x, data)
                    elif pdims==2:
                        self._plot2D_scatter(x, y, data)

            idx = self.dataId.find('# run ID') + 2
            self.plot.axes.set_title("{} [{}]".format(self.dataId[idx:], self.activeDataSet), size='x-small')

            self.plot.draw()

        except Exception as e:
            logging.debug('Could not plot selected data')
            logging.debug('Exception raised: {}'.format(e))

        if self.pendingPlotData:
            self.updatePlotData()

    def addData(self, dataDict):
        """
        Here we receive new data from the listener.
        We'll use a separate thread for processing and combining (numerics might be costly).
        If the thread is already running, we'll put the new data into a queue that will
        be resolved during the next call of addData (i.e, queue will grow until current
        adding thread is done.)
        """

        dataDict = dataDict.get('datasets', {})

        if self.dataAdderThread.isRunning():
            # logger.debug('step 2.1 = DataWindow.addData add data to queue')
            if self.addingQueue == {}:
                self.addingQueue = dataDict
            else:
                self.addingQueue = combineDicts(self.addingQueue, dataDict)
        else:
            if self.addingQueue != {}:
                dataDict = combineDicts(self.addingQueue, dataDict)

            if dataDict != {}:
                # move data to dataAdder obj and start dataAdderThread
                # logger.debug('step 2.2 = DataWindow.addData add queued and existing data in dataAdderThread')
                self.dataAdder.setData(self.data, self.dataStructure, dataDict)
                self.dataAdderThread.start()
                self.addingQueue = {}

    @pyqtSlot(object, dict)
    def dataFromAdder(self, data, dataStructure):
        self.data = data
        self.dataStructure = dataStructure

        logger.debug('step 4 = DataWindow.dataFromAdder')

        self.dataAdded.emit(self.dataStructure)

    # clean-up
    def closeEvent(self, event):
        self.windowClosed.emit(self.dataId)


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
                dataId = data['id']
                self.sendInfo.emit(f'Received data for dataset: {dataId}')
                self.sendData.emit(data)
                logger.debug(f'\n\t DataReceiver received: {data} \n')
                continue
            elif 'ping' in data.keys():
                # so this doesn't look like an error
                # when checking if the server is running
                ping = data['ping']
                self.sendInfo.emit(f'Received ping.')
                continue
            else:
                self.sendInfo.emit(f'Received invalid message (expected DataDict or ping):\n{data}')
                continue

class Logger(QPlainTextEdit):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)

    @pyqtSlot(str)
    def addMessage(self, msg):
        newMsg = "{} {}".format(getTimestamp(), msg)
        self.appendPlainText(newMsg)

class QchartMain(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle(getAppTitle())
        self.resize ( 900, 300 )
        self.activateWindow()

        # layout of basic widgets
        self.logger = Logger()
        self.frame = QFrame()
        layout = QVBoxLayout(self.frame)
        layout.addWidget(self.logger)

        self.setCentralWidget(self.frame)
        self.frame.setFocus()

        # basic setup of the data handling
        self.dataHandlers = {}

        # setting up the ZMQ thread
        self.listeningThread = QThread()
        self.listener = DataReceiver()
        self.listener.moveToThread(self.listeningThread)

        # communication with the ZMQ thread
        self.listeningThread.started.connect(self.listener.loop)
        self.listener.sendInfo.connect(self.logger.addMessage)
        self.listener.sendData.connect(self.processData)

        # go!
        self.listeningThread.start()


    @pyqtSlot(dict)
    def processData(self, data):

        dataId = data['id']

        if dataId not in self.dataHandlers:
            self.dataHandlers[dataId] = DataWindow(dataId=dataId)
            self.dataHandlers[dataId].show()
            self.logger.addMessage(f'Started new data window for {dataId}')
            self.dataHandlers[dataId].windowClosed.connect(self.dataWindowClosed)

        w = self.dataHandlers[dataId]
        w.addData(data)

    def closeEvent(self, event):
        self.listener.running = False
        self.listeningThread.quit()

        hs = [h for d, h in self.dataHandlers.items()]
        for h in hs:
            h.close()

    @pyqtSlot(str)
    def dataWindowClosed(self, dataId):
        self.dataHandlers[dataId].close()
        del self.dataHandlers[dataId]

def console_entry():
    """
    Entry point for launching the app from a console script
    """

    logger.debug('Starting shockley plot...')

    app = QApplication(sys.argv)
    main = QchartMain()
    main.show()
    sys.exit(app.exec_())
