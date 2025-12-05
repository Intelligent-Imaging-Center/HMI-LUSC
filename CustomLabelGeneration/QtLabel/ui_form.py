# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.6.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractScrollArea, QApplication, QGraphicsView, QHeaderView,
    QLabel, QListWidget, QListWidgetItem, QPushButton,
    QSizePolicy, QTableWidget, QTableWidgetItem, QWidget)

class Ui_Widget(object):
    def setupUi(self, Widget):
        if not Widget.objectName():
            Widget.setObjectName(u"Widget")
        Widget.resize(1600, 900)
        self.label = QLabel(Widget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(60, 130, 91, 16))
        self.originalGraphView = QGraphicsView(Widget)
        self.originalGraphView.setObjectName(u"originalGraphView")
        self.originalGraphView.setGeometry(QRect(200, 150, 640, 500))
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.originalGraphView.sizePolicy().hasHeightForWidth())
        self.originalGraphView.setSizePolicy(sizePolicy)
        self.originalGraphView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.originalGraphView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.originalGraphView.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.selectFolderButton = QPushButton(Widget)
        self.selectFolderButton.setObjectName(u"selectFolderButton")
        self.selectFolderButton.setGeometry(QRect(50, 90, 101, 23))
        self.selectFolderButton.setToolTipDuration(12)
        self.labelView = QGraphicsView(Widget)
        self.labelView.setObjectName(u"labelView")
        self.labelView.setGeometry(QRect(880, 150, 640, 500))
        self.labelView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.labelView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.labelView.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.label_2 = QLabel(Widget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(320, 110, 311, 31))
        self.label_2.setAlignment(Qt.AlignCenter)
        self.label_3 = QLabel(Widget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(980, 110, 441, 20))
        self.label_3.setAlignment(Qt.AlignCenter)
        self.selectTypeButton = QPushButton(Widget)
        self.selectTypeButton.setObjectName(u"selectTypeButton")
        self.selectTypeButton.setGeometry(QRect(730, 700, 131, 71))
        self.removeButton = QPushButton(Widget)
        self.removeButton.setObjectName(u"removeButton")
        self.removeButton.setGeometry(QRect(730, 800, 131, 81))
        self.exportButton = QPushButton(Widget)
        self.exportButton.setObjectName(u"exportButton")
        self.exportButton.setGeometry(QRect(1400, 790, 131, 71))
        self.folderNameLabel = QLabel(Widget)
        self.folderNameLabel.setObjectName(u"folderNameLabel")
        self.folderNameLabel.setGeometry(QRect(30, 70, 301, 16))
        self.fileListWidget = QListWidget(Widget)
        self.fileListWidget.setObjectName(u"fileListWidget")
        self.fileListWidget.setGeometry(QRect(10, 150, 171, 311))
        self.switchLabelButton = QPushButton(Widget)
        self.switchLabelButton.setObjectName(u"switchLabelButton")
        self.switchLabelButton.setGeometry(QRect(1370, 90, 131, 51))
        self.originalLabelTypeTable = QTableWidget(Widget)
        self.originalLabelTypeTable.setObjectName(u"originalLabelTypeTable")
        self.originalLabelTypeTable.setGeometry(QRect(400, 690, 256, 192))
        self.newLabelTypeTable = QTableWidget(Widget)
        self.newLabelTypeTable.setObjectName(u"newLabelTypeTable")
        self.newLabelTypeTable.setGeometry(QRect(940, 680, 256, 192))
        self.folderNameLabel_2 = QLabel(Widget)
        self.folderNameLabel_2.setObjectName(u"folderNameLabel_2")
        self.folderNameLabel_2.setGeometry(QRect(30, 50, 641, 16))
        self.fileNameLabel = QLabel(Widget)
        self.fileNameLabel.setObjectName(u"fileNameLabel")
        self.fileNameLabel.setGeometry(QRect(1340, 690, 231, 16))
        self.label_5 = QLabel(Widget)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(1220, 690, 81, 16))
        self.label_6 = QLabel(Widget)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(1220, 720, 131, 16))
        self.outputFolderLabel = QLabel(Widget)
        self.outputFolderLabel.setObjectName(u"outputFolderLabel")
        self.outputFolderLabel.setGeometry(QRect(1340, 720, 231, 16))
        self.label_4 = QLabel(Widget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(930, 10, 521, 81))
        self.changeExportLocationButton = QPushButton(Widget)
        self.changeExportLocationButton.setObjectName(u"changeExportLocationButton")
        self.changeExportLocationButton.setGeometry(QRect(1230, 790, 151, 71))

        self.retranslateUi(Widget)

        QMetaObject.connectSlotsByName(Widget)
    # setupUi

    def retranslateUi(self, Widget):
        Widget.setWindowTitle(QCoreApplication.translate("Widget", u"Widget", None))
        self.label.setText(QCoreApplication.translate("Widget", u"Choose File", None))
        self.selectFolderButton.setText(QCoreApplication.translate("Widget", u"Select Folder", None))
        self.label_2.setText(QCoreApplication.translate("Widget", u"Original Graph", None))
        self.label_3.setText(QCoreApplication.translate("Widget", u"Original Label and New Label", None))
        self.selectTypeButton.setText(QCoreApplication.translate("Widget", u"Select", None))
        self.removeButton.setText(QCoreApplication.translate("Widget", u"Delete", None))
        self.exportButton.setText(QCoreApplication.translate("Widget", u"Export", None))
        self.folderNameLabel.setText(QCoreApplication.translate("Widget", u"No Folder Selected Yet", None))
        self.switchLabelButton.setText(QCoreApplication.translate("Widget", u"Change Mod", None))
        self.folderNameLabel_2.setText(QCoreApplication.translate("Widget", u"Input Folder\uff08must include hdr,dat and npy for Kmean/other classifications sharing same filename\uff09", None))
        self.fileNameLabel.setText("")
        self.label_5.setText(QCoreApplication.translate("Widget", u"Selected File", None))
        self.label_6.setText(QCoreApplication.translate("Widget", u"Output Folder", None))
        self.outputFolderLabel.setText("")
        self.label_4.setText(QCoreApplication.translate("Widget", u"For output files, npy is classification strating from 0, jpeg is visual graphs", None))
        self.changeExportLocationButton.setText(QCoreApplication.translate("Widget", u"Change Output Folder", None))
    # retranslateUi

