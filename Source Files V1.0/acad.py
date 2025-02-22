# Form implementation generated from reading ui file 'acad.ui'
#
# Created by: PyQt6 UI code generator 6.1.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Dialoga(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(536, 502)
        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setGeometry(QtCore.QRect(10, 70, 521, 421))
        self.textBrowser.setObjectName("textBrowser")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(10, 20, 321, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.textBrowser.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt; font-weight:600;\">Problem Statement:</span><span style=\" font-size:8pt;\"> </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">The task assigned for the PBL is to determine the needed fire flow for the Academic Block.</span><span style=\" font-size:8pt;\"> </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\"> </span><span style=\" font-size:8pt;\"> </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:18pt; font-weight:600;\">Solution:</span><span style=\" font-size:12pt;\"> </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">To estimate the amount of water required to fight a fire in an individual, non-sprinklered building, ISO</span><span style=\" font-family:\'TimesNewRoman,serif\'; font-size:12pt; color:#000000;\"> </span><span style=\" font-size:12pt;\">uses the formula:</span><span style=\" font-size:8pt;\"> </span></p>\n"
"<p align=\"center\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'TimesNewRoman,serif\'; font-size:12pt; color:#000000;\"><br />NFF</span><span style=\" font-size:8pt;\">i </span><span style=\" font-size:12pt;\">= (C</span><span style=\" font-size:8pt;\">i</span><span style=\" font-size:12pt;\">)(O</span><span style=\" font-size:8pt;\">i</span><span style=\" font-size:12pt;\">)[1.0 + (X + P)</span><span style=\" font-size:8pt;\">i</span><span style=\" font-size:12pt;\">]</span><span style=\" font-size:8pt;\"> </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'TimesNewRoman,serif\'; font-size:12pt; font-weight:600; color:#000000;\"><br /></span><span style=\" font-size:12pt;\">where</span><span style=\" font-family:\'TimesNewRoman,serif\'; font-size:12pt; color:#000000;\"><br />NFF</span><span style=\" font-size:8pt;\">i </span><span style=\" font-size:12pt;\">= the needed fire flow in gallons per minute (gpm)</span><span style=\" font-family:\'TimesNewRoman,serif\'; font-size:12pt; color:#000000;\"><br />C</span><span style=\" font-size:8pt;\">i </span><span style=\" font-size:12pt;\">= a factor related to the type of construction and effective area</span><span style=\" font-family:\'TimesNewRoman,serif\'; font-size:12pt; color:#000000;\"><br />O</span><span style=\" font-size:8pt;\">i </span><span style=\" font-size:12pt;\">= a factor related to the type of occupancy</span><span style=\" font-family:\'TimesNewRoman,serif\'; font-size:12pt; color:#000000;\"><br />X = a factor related to the exposure hazard of adjacent buildings<br />P = a factor related to the communication hazard with adjacent buildings</span><span style=\" font-size:8pt;\"> </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">For Academic Block:</span><span style=\" font-size:8pt;\"> </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Symbol\'; font-size:12pt;\">·</span><span style=\" font-family:\'Times New Roman\'; font-size:7pt;\">         </span><span style=\" font-size:12pt; font-weight:600; font-style:italic;\">Calculating C:</span><span style=\" font-size:8pt;\"> </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'CambriaMath,serif\'; font-size:12pt; color:#000000;\">C = 18F(√A ) where for academic Block Effective Area from calculation = 74356 ft</span><span style=\" font-family:\'CambriaMath,serif\'; font-size:12pt; color:#000000; vertical-align:super;\">2</span><span style=\" font-family:\'CambriaMath,serif\'; font-size:12pt; color:#000000;\"> (the effective area shall be the total square foot area of the largest floor in the building, plus  25% of the area of not exceeding the two other largest floors) and F factor = 0.6 for Construction Class 6 (Fire Resistive) </span><span style=\" font-family:\'CambriaMath,serif\'; font-size:12pt; font-weight:600; font-style:italic; color:#000000;\">C= 3000</span><span style=\" font-size:8pt;\"> </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'CambriaMath,serif\'; font-size:12pt; color:#000000;\">Thus C is in the limits 500&lt;C&lt;6000.</span><span style=\" font-size:8pt;\"> </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Symbol\'; font-size:12pt;\">·</span><span style=\" font-family:\'Times New Roman\'; font-size:7pt;\">         </span><span style=\" font-family:\'CambriaMath,serif\'; font-size:12pt; font-weight:600; font-style:italic;\">Calculating Occupancy Factor O:</span><span style=\" font-size:8pt;\"> </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'CambriaMath,serif\'; font-size:12pt;\">The occupancy factor for academic block is </span><span style=\" font-family:\'CambriaMath,serif\'; font-size:12pt; font-weight:600;\">0.85</span><span style=\" font-family:\'CambriaMath,serif\'; font-size:12pt;\"> Limited Combustibility (C-2) -</span><span style=\" font-size:12pt;\"> </span><span style=\" font-family:\'CambriaMath,serif\'; font-size:12pt;\">Because its having Merchandise or materials, including furniture, stock, or equipment, of low combustibility, with limited concentrations of combustible materials.</span><span style=\" font-size:8pt;\"> </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Symbol\'; font-size:12pt;\">·</span><span style=\" font-family:\'Times New Roman\'; font-size:7pt;\">         </span><span style=\" font-family:\'CambriaMath,serif\'; font-size:12pt; font-weight:600; font-style:italic;\">Exposure Factor:</span><span style=\" font-size:8pt;\"> </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'CambriaMath,serif\'; font-size:12pt;\">The exposure factor for academic was taken to be 0 since from Google Maps there is no building in the vicinity of 40ft.</span><span style=\" font-size:8pt;\"> </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Symbol\'; font-size:12pt;\">·</span><span style=\" font-family:\'Times New Roman\'; font-size:7pt;\">         </span><span style=\" font-family:\'CambriaMath,serif\'; font-size:12pt; font-weight:600; font-style:italic;\">Communication Factor:</span><span style=\" font-size:8pt;\"> </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'CambriaMath,serif\'; font-size:12pt;\">The communication factor for academic block was also taken to be zero.</span><span style=\" font-size:8pt;\"> </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'CambriaMath,serif\'; font-size:12pt;\"> </span><span style=\" font-size:8pt;\"> </span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'CambriaMath,serif\'; font-size:12pt;\"> </span><span style=\" font-size:8pt;\"> </span></p>\n"
"<p align=\"center\" style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'CambriaMath,serif\'; font-size:14pt; font-weight:600;\">Needed Fire Flow Calculated from the equation was 2503 Which can be rounded off to 2500.</span><span style=\" font-size:8pt;\"> </span></p></body></html>"))
        self.label.setText(_translate("Dialog", "Calculation For Academic Block"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialoga()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec())
