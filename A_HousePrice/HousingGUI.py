'''
Created on 2022. 4. 27.

@author: pc360
'''
# -*- coding: utf-8 -*-

import os
import sys
import csv
import wx.dataview
import wx.richtext
import logging
from time import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.use('WXAgg')

MY_EPOCH = 500  # 반복횟수
MY_BATCH = 64  # 일괄처리량

class RedirectText(object):
    def __init__(self,aWxTextCtrl):
        self.terminal = sys.stdout
        self.out = aWxTextCtrl

    def write(self,string):
        self.out.WriteText(string)

    def flush(self):
        self.terminal.flush()
        

class MyFrame1(wx.Frame):

    def __init__(self, parent):
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=wx.EmptyString, pos=wx.DefaultPosition,
                          size=wx.Size(1024, 768), style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)

        self.SetSizeHints(wx.DefaultSize, wx.DefaultSize)

        bSizer1 = wx.BoxSizer(wx.HORIZONTAL)

        bSizer2 = wx.BoxSizer(wx.VERTICAL)

        bSizer2.SetMinSize(wx.Size(700, -1))
        bSizer3 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_filePicker1 = wx.TextCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0)
        bSizer3.Add(self.m_filePicker1, 1, wx.ALL | wx.EXPAND, 5)

        self.m_button0 = wx.Button(self, wx.ID_ANY, u"불러오기", wx.DefaultPosition, wx.Size(70, -1), 0)
        bSizer3.Add(self.m_button0, 0, wx.ALL, 5)

        self.m_button1 = wx.Button(self, wx.ID_ANY, u"자료삭제", wx.DefaultPosition, wx.Size(70, -1), 0)
        bSizer3.Add(self.m_button1, 0, wx.ALL | wx.EXPAND, 5)

        bSizer2.Add(bSizer3, 0, wx.EXPAND, 5)

        bSizer4 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_button3 = wx.Button(self, wx.ID_ANY, u"원본 데이터 통계", wx.DefaultPosition, wx.DefaultSize, 0)
        bSizer4.Add(self.m_button3, 1, wx.ALL, 5)

        self.m_button4 = wx.Button(self, wx.ID_ANY, u"정규화 데이터 통계", wx.DefaultPosition, wx.DefaultSize, 0)
        bSizer4.Add(self.m_button4, 1, wx.ALL, 5)

        self.m_button5 = wx.Button(self, wx.ID_ANY, u"그래프 보기", wx.DefaultPosition, wx.DefaultSize, 0)
        bSizer4.Add(self.m_button5, 1, wx.ALL, 5)

        self.m_button6 = wx.Button(self, wx.ID_ANY, u"DNN 요약", wx.DefaultPosition, wx.DefaultSize, 0)
        bSizer4.Add(self.m_button6, 1, wx.ALL, 5)

        self.m_button7 = wx.Button(self, wx.ID_ANY, u"DNN 학습 시작", wx.DefaultPosition, wx.DefaultSize, 0)
        bSizer4.Add(self.m_button7, 1, wx.ALL, 5)

        self.m_button8 = wx.Button(self, wx.ID_ANY, u"결과보기", wx.DefaultPosition, wx.DefaultSize, 0)
        bSizer4.Add(self.m_button8, 1, wx.ALL, 5)

        bSizer2.Add(bSizer4, 0, wx.EXPAND, 5)

        bSizer5 = wx.BoxSizer(wx.HORIZONTAL)
        self.m_tabs = wx.Notebook(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, 0)
        bSizer5.Add(self.m_tabs, 1, wx.EXPAND | wx.ALL, 5)

        bSizer2.Add(bSizer5, 1, wx.EXPAND, 5)

        bSizer1.Add(bSizer2, 1, wx.EXPAND, 5)

        self.SetSizer(bSizer1)
        self.Layout()

        self.Centre(wx.BOTH)

        # Connect Events
        self.m_button0.Bind(wx.EVT_BUTTON, self.loadFile)
        self.m_button1.Bind(wx.EVT_BUTTON, self.clearData)
        self.m_button3.Bind(wx.EVT_BUTTON, self.O_Describe)
        self.m_button4.Bind(wx.EVT_BUTTON, self.Z_Describe)
        self.m_button5.Bind(wx.EVT_BUTTON, self.ShowPlot)
        self.m_button6.Bind(wx.EVT_BUTTON, self.SumDNN)
        self.m_button7.Bind(wx.EVT_BUTTON, self.StartStudy)
        self.m_button8.Bind(wx.EVT_BUTTON, self.ShowResult)


    def loadFile(self, event):

        self.csv_data = wx.Panel(self.m_tabs, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        bSizer7 = wx.BoxSizer(wx.VERTICAL)

        self.csv_DataViewListCtrl = wx.dataview.DataViewListCtrl(self.csv_data, wx.ID_ANY, wx.DefaultPosition,
                                                                 wx.DefaultSize, 0)
        bSizer7.Add(self.csv_DataViewListCtrl, 1, wx.ALL | wx.EXPAND, 5)

        self.csv_data.SetSizer(bSizer7)
        self.csv_data.Layout()
        bSizer7.Fit(self.csv_data)
        self.m_tabs.AddPage(self.csv_data, u"원본 데이터", True)
        global pathname; global thisfile;
        global filename; global fileexts;
        global spltname; global dirrname;
        with wx.FileDialog(self, "Open CSV file", \
            wildcard="CSV files (*.csv *.txt)|*.csv;*.txt", \
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return     
            pathname = fileDialog.GetPath()
            thisfile=os.path.basename(pathname)
            filename=thisfile.split('.')[0]
            fileexts=thisfile.split('.')[1]
            dirrname=os.path.dirname(pathname)
            spltname=os.path.splitext(pathname)
            self.m_filePicker1.Clear()
            self.m_filePicker1.AppendText(pathname)

        with open(pathname, 'r', encoding='utf-8') as f:
            raw = csv.reader(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)

            # 기존컬럼 있으면 지우기
            if (self.csv_DataViewListCtrl.GetColumnCount() > 0):
                self.csv_DataViewListCtrl.ClearColumns()
                self.csv_DataViewListCtrl.DeleteAllItems()

            # 컬럼&값 추가하기
            rownum = 0
            global COLS;            COLS = []
            for row in raw:
                if rownum == 0:
                    for col in row:
                        column = self.csv_DataViewListCtrl.AppendTextColumn(col, width=-1)
                        self.csv_DataViewListCtrl.Columns.append(column)
                        COLS.append(col)
                else:
                    for col in row:
                        col = round(float(col), 4)
                    self.csv_DataViewListCtrl.AppendItem(row)
                rownum += 1


    def clearData(self, event):
        self.m_tabs.Destroy()


    def O_Describe(self, event):
        raw = pd.read_csv(pathname)
        ori = raw.describe()
        ori = ori.transpose()
        colname = ori.columns.tolist()
        rowname = ori.index.tolist()

        self.csv_stats = wx.Panel(self.m_tabs, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        bSizer8 = wx.BoxSizer(wx.VERTICAL)

        self.csv_StatsViewListCtrl = wx.dataview.DataViewListCtrl(self.csv_stats, wx.ID_ANY, wx.DefaultPosition,
                                                                  wx.DefaultSize, 0)
        bSizer8.Add(self.csv_StatsViewListCtrl, 1, wx.ALL | wx.EXPAND, 5)

        self.csv_stats.SetSizer(bSizer8)
        self.csv_stats.Layout()
        bSizer8.Fit(self.csv_stats)
        self.m_tabs.AddPage(self.csv_stats, u"원본 통계", True)

        if (self.csv_StatsViewListCtrl.GetColumnCount() > 0):
            self.csv_StatsViewListCtrl.ClearColumns()
            self.csv_StatsViewListCtrl.DeleteAllItems()
        rowname.insert(0, '통계값')
        for col in rowname:
            column = self.csv_StatsViewListCtrl.AppendTextColumn(col, width=-1)
            self.csv_StatsViewListCtrl.Columns.append(column)
        rowname.remove('통계값')
        for col in colname:
            rowData = [col]
            for row in rowname:
                pp = str(round(ori[col][row], 4))
                rowData.append(pp)
            self.csv_StatsViewListCtrl.AppendItem(rowData)


    def Z_Describe(self, event):
        raw = pd.read_csv(pathname)
        scaler = StandardScaler()
        global Z_data
        Z_data = scaler.fit_transform(raw)
        Z_data = pd.DataFrame(Z_data, columns=raw.columns.values.tolist())
        zdd = Z_data.describe()
        zdd = zdd.transpose()
        colname = zdd.columns.tolist()
        rowname = zdd.index.tolist()

        self.z_stats = wx.Panel(self.m_tabs, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        bSizer9 = wx.BoxSizer(wx.VERTICAL)

        self.z_StatsViewListCtrl = wx.dataview.DataViewListCtrl(self.z_stats, wx.ID_ANY, wx.DefaultPosition,
                                                                wx.DefaultSize, 0)
        bSizer9.Add(self.z_StatsViewListCtrl, 1, wx.ALL | wx.EXPAND, 5)

        self.z_stats.SetSizer(bSizer9)
        self.z_stats.Layout()
        bSizer9.Fit(self.z_stats)
        self.m_tabs.AddPage(self.z_stats, u"정규화 통계", True)

        if (self.z_StatsViewListCtrl.GetColumnCount() > 0):
            self.z_StatsViewListCtrl.ClearColumns()
            self.z_StatsViewListCtrl.DeleteAllItems()

        rowname.insert(0, '통계값')
        for col in rowname:
            column = self.z_StatsViewListCtrl.AppendTextColumn(col, width=-1)
            self.z_StatsViewListCtrl.Columns.append(column)
        rowname.remove('통계값')
        for col in colname:
            rowData = [col]
            for row in rowname:
                pp = str(round(zdd[col][row], 4))
                rowData.append(str(pp))
            self.z_StatsViewListCtrl.AppendItem(rowData)


    def ShowPlot(self, event):
        plt.cla();        plt.clf()
        global X_data;          global Y_data;
        X_data = Z_data.drop('MEDV', axis=1)
        Y_data = Z_data['MEDV']
        global X_train;         global Y_train;
        global X_test;          global Y_test
        X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.3)
        sns.set(font_scale=0.8)
        sns.boxplot(data=Z_data, palette='dark')
        plt.tight_layout()
        
        plotPNG = filename+"plot.png"
        if os.path.isfile(plotPNG):
            os.remove(plotPNG)
        plt.savefig(plotPNG, bbox_inches='tight')

        self.BoxPlot1 = wx.Panel(self.m_tabs, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        bSizerBP = wx.BoxSizer(wx.VERTICAL)

        png = wx.Image(plotPNG, wx.BITMAP_TYPE_ANY).ConvertToBitmap()

        self.BoxPlotPNG = wx.StaticBitmap(self.BoxPlot1, wx.ID_ANY, png, wx.DefaultPosition,
                                          (png.GetWidth(), png.GetHeight()), 0)
        bSizerBP.Add(self.BoxPlotPNG, 1, wx.ALIGN_CENTER | wx.ALL, 5)

        self.BoxPlot1.SetSizer(bSizerBP)
        self.BoxPlot1.Layout()
        bSizerBP.Fit(self.BoxPlot1)

        self.m_tabs.AddPage(self.BoxPlot1, u"박스그래프", True)


    def SumDNN(self, event):
        global model
        model = Sequential()
        indim = X_train.shape[1]
        model.add(Dense(200, input_dim=indim, activation='relu'))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(1))

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        SumLog = filename+"modelSum.log"
        if os.path.isfile(SumLog):
            os.remove(SumLog)
        file_handler = logging.FileHandler(SumLog)
        logger.addHandler(file_handler)
        model.summary(print_fn=logger.info)
        file_handler.close()

        self.DNN_S = wx.Panel(self.m_tabs, wx.ID_ANY, wx.DefaultPosition, wx.Size(-1, 500), wx.TAB_TRAVERSAL)
        bSizerS = wx.BoxSizer(wx.VERTICAL)

        self.Sum_Text = wx.richtext.RichTextCtrl(self.DNN_S, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition,
                                                 wx.DefaultSize,
                                                 0 | wx.VSCROLL | wx.HSCROLL | wx.NO_BORDER | wx.WANTS_CHARS)
        bSizerS.Add(self.Sum_Text, 1, wx.ALL | wx.EXPAND, 5)

        self.DNN_S.SetSizer(bSizerS)
        self.DNN_S.Layout()
        bSizerS.Fit(self.DNN_S)
        self.m_tabs.AddPage(self.DNN_S, u"DNN 요약", True)

        with open(SumLog, 'r', encoding='utf-8') as f:
            rows = f.readlines()
            for row in rows:
                self.Sum_Text.AppendText(row)
        f.close()


    def StartStudy(self, event):
        self.StudyP = wx.Panel( self.m_tabs, wx.ID_ANY, wx.DefaultPosition, wx.Size( -1,500 ), wx.TAB_TRAVERSAL )
        bSizerSP = wx.BoxSizer( wx.VERTICAL )

        self.StudyLog = wx.richtext.RichTextCtrl( self.StudyP, wx.ID_ANY, '학습이 진행중입니다.....', wx.DefaultPosition, wx.DefaultSize, 0|wx.VSCROLL|wx.HSCROLL|wx.NO_BORDER|wx.WANTS_CHARS )
        bSizerSP.Add( self.StudyLog, 1, wx.ALL|wx.EXPAND, 5 )
        self.StudyP.SetSizer( bSizerSP )
        self.StudyP.Layout()
        bSizerSP.Fit( self.StudyP )
        self.m_tabs.AddPage( self.StudyP, u"DNN 학습진행", True )
        self.StudyLog.Clear()
        
        redir = RedirectText(self.StudyLog)
        sys.stdout = redir

        model.compile(optimizer='sgd', loss='mse')

        begin = time()
        model.fit(X_train, Y_train, epochs=MY_EPOCH, batch_size=MY_BATCH, verbose=2)
        end = time()
        ET = end - begin
        self.StudyLog.Newline()
        self.StudyLog.WriteText('학습이 종료되었습니다.')
        self.StudyLog.Newline()
        self.StudyLog.WriteText('\n총 학습시간 : {:.1f}초'.format(ET))

        wx.MessageBox("총 학습시간 : {:.1f}초".format(ET), "학습이 종료되었습니다")


    def ShowResult(self, event):
        plt.cla();        plt.clf()
        loss = model.evaluate(X_test, Y_test, verbose=0)
        res = 'DNN 평균 제곱 오차 (MSE) : {:.2f}'.format(loss)
        pred = model.predict(X_test)
        sns.regplot(x=Y_test, y=pred)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.tight_layout()
        msePlot = filename+"MSE.png"
        if os.path.isfile(msePlot):
            os.remove(msePlot)
        plt.savefig(msePlot, bbox_inches='tight')

        self.MSEPlot = wx.Panel(self.m_tabs, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
        bSizerMSE = wx.BoxSizer(wx.VERTICAL)

        png = wx.Image(msePlot, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        
        self.m_staticText1 = wx.StaticText( self.MSEPlot, wx.ID_ANY, res, wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText1.Wrap( -1 )
        self.m_staticText1.SetFont( wx.Font( 20, 70, 90, 92, False, wx.EmptyString))

        self.BoxPlotPNG = wx.StaticBitmap(self.MSEPlot, wx.ID_ANY, png, wx.DefaultPosition,
                                          (png.GetWidth(), png.GetHeight()), 0)
        bSizerMSE.Add(self.BoxPlotPNG, 1, wx.ALIGN_CENTER | wx.ALL, 5)
        bSizerMSE.Add( self.m_staticText1, 0, wx.ALIGN_CENTER|wx.ALL, 5 )

        self.MSEPlot.SetSizer(bSizerMSE)
        self.MSEPlot.Layout()
        bSizerMSE.Fit(self.MSEPlot)

        self.m_tabs.AddPage(self.MSEPlot, u"최종 MSE 평가", True)


if __name__ == '__main__':
    ex = wx.App()
    frame = MyFrame1(None)
    frame.SetSize(wx.Size(1024, 768))
    frame.Show()
    ex.MainLoop()