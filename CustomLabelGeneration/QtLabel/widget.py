# This Python file uses the following encoding: utf-8
import sys
from PySide6.QtGui import QPixmap, QImage, QColor
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox, QGraphicsRectItem, QGraphicsPixmapItem, QTableWidgetItem
from PySide6 import QtWidgets, QtGui, QtCore

import os
import spectral as spy
import numpy as np
import time

def generate_file_list(dir, end, includeFolderPath = False):
    if includeFolderPath:
        list = [os.path.join(dir,f) for f in os.listdir(dir) if f.endswith(end)]
    else:
        list = [f for f in os.listdir(dir) if f.endswith(end)]
    list.sort()
    return list

def changeExtension(file, ext):
    return os.path.splitext(file)[0] + "." + ext

def optimized_linear(arr):
    a, b = np.percentile(arr, (2.5, 99))
    c = a - 0.1 * (b - a)
    d = b + 0.5 * (b - a)
    arr = (arr - c) / (d - c) * 255
    arr = np.clip(arr, 0, 255)
    return np.uint8(arr)

def percent_linear(arr, percent=2):
    arr_min, arr_max = np.percentile(arr, (percent, 100-percent))
    arr = (arr - arr_min) / (arr_max - arr_min) * 255
    arr = np.clip(arr, 0, 255)
    return np.uint8(arr)

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_Widget
class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)

        self.selectedFolder = ""
        self.selectedFile = ""
        self.selectedLabelFile = ""
        self.outputFolder = ""
        self.originalImage = None # should be a QImage object
        self.originalLabel = None # should be a numpy array
        self.newLabel = None # should be a numpy array
        self.originalLabelImage = None # should be a QImage object
        self.newLabelImage = None # should be a QImage object
        self.displayOriginalLabel = True

        self.colors = None
        self.sorted_label = None
        self.selected_label = []

        self.ui.selectFolderButton.clicked.connect(self.selectFolderClicked)
        self.ui.fileListWidget.currentRowChanged.connect(self.selectFileClicked)
        self.ui.originalGraphView.horizontalScrollBar().valueChanged.connect(self.ui.labelView.horizontalScrollBar().setValue)
        self.ui.originalGraphView.verticalScrollBar().valueChanged.connect(self.ui.labelView.verticalScrollBar().setValue)
        self.ui.selectTypeButton.clicked.connect(self.selectTypeButtonClicked)
        self.ui.switchLabelButton.clicked.connect(self.switchLabelButtonClicked)
        self.ui.removeButton.clicked.connect(self.removeButtonClicked)
        self.ui.changeExportLocationButton.clicked.connect(self.changeExportLocationButtonClicked)
        self.ui.exportButton.clicked.connect(self.exportButtonClicked)
        self.outputColors = np.array([[0,0,0],
                                      [255,0,0]], dtype=np.uint8)

    def reset(self):
        self.selectedFile = ""
        self.selectedLabelFile = ""
        self.outputFolder = ""
        self.ui.outputFolderLabel.setText("")
        self.outputImageFolder = ""
        self.originalImage = None # should be a QImage object
        self.originalLabel = None # should be a numpy array
        self.newLabel = None # should be a numpy array
        self.originalLabelImage = None # should be a QImage object
        self.newLabelImage = None # should be a QImage object
        self.displayOriginalLabel = True

        self.colors = None
        self.sorted_label = None
        self.selected_label = []

    def switchLabelButtonClicked(self):
        self.displayOriginalLabel = not self.displayOriginalLabel
        self.updateSecondImageDisplay()

    def removeButtonClicked(self):
        table = self.ui.newLabelTypeTable
        row = table.currentRow()
        # get index number
        index = int(table.item(row,0).text())
        self.selected_label.remove(index)
        # remove row
        table.removeRow(row)
        self.updateLabelTable()

    def updateLabelTable(self):
        table = self.ui.newLabelTypeTable
        table.setRowCount(len(self.selected_label))
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["ID", "Color"])
        label_num = len(self.selected_label)
        for i in range(0,label_num):
            index = self.selected_label[i]
            table.setItem(i, 0, QTableWidgetItem(str(index)))
            color_item = QTableWidgetItem()
            selectedColor = self.colors[index]
            color_item.setBackground(QColor.fromRgb(selectedColor[0], selectedColor[1], selectedColor[2]))
            table.setItem(i, 1, color_item)
        # update new label and label image
        new_label = np.where(np.isin(self.originalLabel, self.selected_label), 1, 0)
        self.newLabel = new_label
        img = self.outputColors[self.newLabel]
        h, w= self.originalLabel.shape
        self.newLabelImage = QImage(img, w, h, 3 * w, QImage.Format_RGB888)
        self.updateSecondImageDisplay()

    def selectTypeButtonClicked(self):
        rowIndex = self.ui.originalLabelTypeTable.currentRow()
        if self.sorted_label[rowIndex] not in self.selected_label:
            self.selected_label.append(self.sorted_label[rowIndex])
            self.updateLabelTable()



    # Trigger folder selection, should update folderNameLabel and folderContentList
    def selectFolderClicked(self):
        dlg = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        file_dir = str(dlg)
        self.selectedFolder = file_dir
        self.ui.folderNameLabel.setText(self.selectedFolder)
        # Find all file under the folder with hdr extension
        fileNameList = generate_file_list(self.selectedFolder, "hdr")
        self.ui.fileListWidget.addItems(fileNameList)


    # Trigger folder selection, should update folderNameLabel and folderContentList
    def changeExportLocationButtonClicked(self):
        dlg = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        file_dir = str(dlg)
        self.outputFolder = file_dir
        self.ui.outputFolderLabel.setText(self.outputFolder)

    def updateImageContent(self):
        # Set up original graph
        hsi_data = spy.open_image(os.path.join(self.selectedFolder,self.selectedFile)).load()
        hsi_data = hsi_data * 255
        rgb_data = np.ascontiguousarray(hsi_data[:,:,[41,17,3]])
        rgb_data = rgb_data.astype(np.uint8)
        for i in range(0,3):
            rgb_data[:,:,i] = optimized_linear(rgb_data[:,:,i])
        h, w, _ = rgb_data.shape
        self.originalImage = QImage(rgb_data, w, h, 3 * w, QImage.Format_RGB888)

        # Set up label and orignal label image
        label = np.load(os.path.join(self.selectedFolder,self.selectedLabelFile))
        label = label.astype(np.uint8)
        label_unique_values, label_counts =  np.unique(label, return_counts=True)
        label_num = label_unique_values.shape[0]
        colors = np.random.rand(label_num*3)*255
        colors = colors.astype(np.uint8)
        colors = colors.reshape((label_num,3))
        sorted_label_unique_values = label_unique_values[np.argsort(-label_counts)]
        self.colors = colors
        self.sorted_label = sorted_label_unique_values
        img = colors[label]
        h, w, _ = img.shape
        self.originalLabelImage = QImage(img, w, h, 3 * w, QImage.Format_RGB888)
        self.originalLabel = label

        # Set up originalLabelTable
        table = self.ui.originalLabelTypeTable
        table.setRowCount(label_num)
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["ID", "Color"])
        for i in range(0, label_num):
            index = self.sorted_label[i]
            table.setItem(i, 0, QTableWidgetItem(str(index)))
            color_item = QTableWidgetItem()
            selectedColor = self.colors[index]
            color_item.setBackground(QColor.fromRgb(selectedColor[0], selectedColor[1], selectedColor[2]))
            table.setItem(i, 1, color_item)

        # Initialize new label and Image
        new_label = np.where(np.isin(self.originalLabel, self.selected_label), 1, 0) 
        self.newLabel = new_label
        img = self.outputColors[self.newLabel]
        self.newLabelImage = QImage(img, w, h, 3 * w, QImage.Format_RGB888)


    def updateImageDisplay(self):
        # ----------------------First Image ------------------------
        originalPic = QGraphicsPixmapItem()
        originalPic.setPixmap(QPixmap.fromImage(self.originalImage))
        # initialize a new scene for the first image everytime
        originalScene = QtWidgets.QGraphicsScene()
        self.ui.originalGraphView.setScene(originalScene)
        self.ui.originalGraphView.scene().addItem(originalPic)

        # -----------------------Second Image ---------------------------
        labelPic = QGraphicsPixmapItem()
        # set up the first image, either original image or the corresponding label
        if self.displayOriginalLabel:
            # Generate the label graph
            labelPic.setPixmap(QPixmap.fromImage(self.originalLabelImage))
        else:
            labelPic.setPixmap(QPixmap.fromImage(self.newLabelImage))
        # initialize a new scene for the second image everytime
        labelScene = QtWidgets.QGraphicsScene()
        labelScene.addItem(labelPic)
        self.ui.labelView.setScene(labelScene)

    def updateSecondImageDisplay(self):

        # -----------------------Second Image ---------------------------
        labelPic = QGraphicsPixmapItem()
        # set up the first image, either original image or the corresponding label
        if self.displayOriginalLabel:
            # Generate the label graph
            labelPic.setPixmap(QPixmap.fromImage(self.originalLabelImage))
        else:
            labelPic.setPixmap(QPixmap.fromImage(self.newLabelImage))
        # initialize a new scene for the second image everytime
        labelScene = QtWidgets.QGraphicsScene()
        labelScene.addItem(labelPic)
        self.ui.labelView.setScene(labelScene)

    def selectFileClicked(self):
        # Double check we indeed want to change into a new file
        selectedFile = self.ui.fileListWidget.currentItem().text()
        reply = QtWidgets.QMessageBox()
        reply.setText("Change to another file? " + selectedFile)
        reply.setStandardButtons(QtWidgets.QMessageBox.StandardButtons.Yes |
                                 QtWidgets.QMessageBox.StandardButtons.No)
        x = reply.exec()
        if x == QtWidgets.QMessageBox.StandardButtons.Yes:
            self.reset()
            self.selectedFile = selectedFile
            self.selectedLabelFile = changeExtension(self.selectedFile, "npy")
            self.updateImageContent()
            self.updateImageDisplay()
            self.ui.fileNameLabel.setText(self.selectedFile)

    def exportButtonClicked(self):
        exportFile = os.path.join(self.outputFolder,self.selectedFile)
        exportNpyFile = changeExtension(exportFile,"npy")
        exportJpegFile = changeExtension(exportFile,"tif")
        reply = QtWidgets.QMessageBox()
        reply.setText("Export file " + exportNpyFile + " and " + exportJpegFile)
        reply.setStandardButtons(QtWidgets.QMessageBox.StandardButtons.Yes |
                                 QtWidgets.QMessageBox.StandardButtons.No)
        x = reply.exec()
        if x == QtWidgets.QMessageBox.StandardButtons.Yes:
            np.save(exportNpyFile, self.newLabel)
            self.newLabelImage.save(exportJpegFile,"tif", 100)
            self.ui.originalLabelTypeTable.clearContents()
            self.ui.originalLabelTypeTable.setRowCount(0)
            self.ui.newLabelTypeTable.clearContents()
            self.ui.newLabelTypeTable.setRowCount(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    sys.exit(app.exec())

