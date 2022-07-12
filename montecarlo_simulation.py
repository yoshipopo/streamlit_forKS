#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:26:25 2022

@author: harukiyoshida
"""
import streamlit as st
import requests
import datetime as dt
import pandas_datareader.data as web
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import math
import time
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections import deque
from pandas_datareader.stooq import StooqDailyReader

st.set_page_config(layout="wide")
def main():
    st.title('モンテカルロシミュレーション')
    path = '/Users/harukiyoshida/Downloads/data_j.xls'
    df_all_company_list = path_to_df_all_company_list(path)
    st.write('全銘柄')
    st.dataframe(df_all_company_list)
    
    selections = st.multiselect('銘柄を複数選択してください',
                                     df_all_company_list['コード&銘柄名'],
                                     ['8306三菱ＵＦＪフィナンシャル・グループ','8591オリックス','9020東日本旅客鉄道','9101日本郵船'],on_change=session_change)
    st.write('選択した銘柄')
    st.dataframe(selections_to_selected_company_list_and_selected_company_list_hyouji(df_all_company_list,selections)[0])
    
    selected_company_list = selections_to_selected_company_list_and_selected_company_list_hyouji(df_all_company_list,selections)[1]
    selected_company_list_hyouji = selections_to_selected_company_list_and_selected_company_list_hyouji(df_all_company_list,selections)[2]
    selected_company_list_hyouji_datenashi = selections
    
    duration = st.slider('株価取得期間は？(年)',1,10,2,
                         on_change=session_change)
    
    N = st.slider('モンテカルロ法回数は？',100,100000,10000,
                         on_change=session_change)
       
    #select_button = st.selectbox("解析設定",["設定１","設定２","設定３"],on_change=session_change)
    #st.markdown("##### ↑これの値が変わったら、再度ボタンはOFFにしたくありませんか？？")
    
    press_button = st.button("submit,csv取得")
    st.session_state["is_pressed"] = button_states()
    
    if press_button:
        st.session_state["is_pressed"].update({"pressed": True})
    
    if st.session_state["is_pressed"]["pressed"]:
        #th = st.number_input("解析開始ボタン押下後に出てくるウィジェット")
        df_price_merged = selected_company_list_to_get_df(selected_company_list,selected_company_list_hyouji,duration)[0]
        df_tourakuritu_merged = selected_company_list_to_get_df(selected_company_list,selected_company_list_hyouji,duration)[1]
        
        st.dataframe(df_price_merged)
        
        #st.dataframe(df_price_merged)
        #生株価
        a=df_price_merged
        fig = go.Figure()
        for i in range(len(selected_company_list_hyouji_datenashi)):
          fig.add_trace(go.Scatter(x=a['Date'],
                              y=a.iloc[:,i+1],
                              name=selected_company_list_hyouji_datenashi[i])
                        )
        fig.update_traces(hovertemplate='%{y}')
        fig.update_layout(hovermode='x')
        fig.update_layout(height=500,width=1500,
                          title='資産価格推移',
                          xaxis={'title': 'Date'},
                          yaxis={'title': 'price/円'})                  
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig)
        
        #100にする基準日スライダーでいじれるようにしたかったけど　断念
        #standard_date_tentative = st.slider("株価基準日は？",0,
                                             #len(df_price_merged)-1,
                                             #(int((len(df_price_merged)-1)/2), len(df_price_merged)-1),
                                             #on_change=session_change                                      
        #0日目基準日にした
        standard_date_tentative  = (0,0)
    
        standard_date_tentative2 = len(df_price_merged) -1  -standard_date_tentative[0]
        standard_date = df_price_merged.iat[standard_date_tentative2,0]
        
        df_price_100 = df_price_merged
        for i in range(len(selected_company_list_hyouji_datenashi)):
          df_price_100[selected_company_list_hyouji_datenashi[i]]=100*df_price_100[selected_company_list_hyouji_datenashi[i]]/df_price_100.at[df_price_100.index[standard_date_tentative2], selected_company_list_hyouji_datenashi[i]]

        #st.write(df_price_100)
        b=df_price_100
        fig = go.Figure()
        for i in range(len(selected_company_list_hyouji_datenashi)):
          fig.add_trace(go.Scatter(x = b['Date'],
                              y = b.iloc[:,i+1],
                              name = selected_company_list_hyouji_datenashi[i])
                        )
        fig.update_traces(hovertemplate='%{y}')
        fig.update_layout(hovermode='x')
        fig.update_layout(height=500,width=1500,
                          title='資産価格推移({}=100)'.format(standard_date),
                          xaxis={'title': 'Date'},
                          yaxis={'title': 'price'})
        fig.update_layout(showlegend=True)
        fig.add_shape(type="line",
                      x0=standard_date, y0=0,
                      x1=standard_date, y1=100,
            line=dict(color="black",width=1))
        st.plotly_chart(fig)
        
        st.dataframe(df_tourakuritu_merged)
        #騰落率
        fig = go.Figure()
        for i in range(len(selected_company_list_hyouji_datenashi)):
            fig.add_trace(go.Histogram(x=df_tourakuritu_merged.iloc[:,i+1],
                                       opacity=0.5, name='{}'.format(selected_company_list_hyouji_datenashi[i]),
                                       nbinsx=50
                                       #histnorm='probability',
                                       #hovertext='date{}'.df_tourakuritu_merged.iloc[:,i+1]
                                       ))
            fig.update_layout(height=500,width=1500,
                              title='収益率のヒストグラム',
                              xaxis={'title': '騰落率'},
                              yaxis={'title': '度数'})
        
        fig.update_layout(barmode='overlay')
        st.plotly_chart(fig)
        
        #相関係数
        fig_corr = px.imshow(df_tourakuritu_merged.corr(), text_auto=True, 
                             zmin=-1,zmax=1,
                             color_continuous_scale=['blue','white','red'])
        fig_corr.update_layout(height=500,width=1000,
                               title='収益率の相関係数'
                               )
        st.plotly_chart(fig_corr)
        
        
        
        #モンテカルロ法結果
        
        df=df_tourakuritu_merged
        df=df.drop('Date', axis=1)
        company_list_hyouji_datenashi=df.columns.values
        #st.write(company_list_hyouji_datenashi)
    
        n=len(df.columns)
        #st.write(n)
    
        def get_portfolio(array1,array2,array3):
            rp = np.sum(array1*array2)
            sigmap = array1 @ array3 @ array1
            return array1.tolist(), rp, sigmap
    
        df_vcm=df.cov()
    
        a=np.ones((n,n))
        np.fill_diagonal(a,len(df))
        np_vcm=df_vcm.values@a
    
        a=np.ones((n,n))
        np.fill_diagonal(a,len(df))
    
        df_mean=df.mean()
        np_mean=df_mean.values
        np_mean=np_mean*len(df)
    
        #N=int(st.number_input('モンテカルロシミュレーション回数は？',min_value=1000))
        x=np.random.uniform(size=(N,n))
        x/=np.sum(x, axis=1).reshape([N, 1])
    
        temp=np.identity(n)
        x=np.append(x,temp, axis=0)
    
        squares = [get_portfolio(x[i],np_mean,np_vcm) for i in range(x.shape[0])]
        df2 = pd.DataFrame(squares,columns=['投資比率','収益率', '収益率の分散'])
    
        df2['分類']='PF{}資産で構成'.format(len(company_list_hyouji_datenashi))
        for i in range(x.shape[0]-n,x.shape[0]):
          df2.iat[i, 3] = company_list_hyouji_datenashi[i-x.shape[0]]
          #print(i,company_list_hyouji_datenashi[i-x.shape[0]])
    
        st.dataframe(df2)
        
        fig = px.scatter(df2, x='収益率の分散', y='収益率',hover_name='投資比率',color='分類')
        fig.update_layout(height=500,width=1000,
                          title='モンテカルロシミュレーション結果',
                          xaxis={'title': '収益率の分散'},
                          yaxis={'title': '収益率'},
                          )
        
        st.plotly_chart(fig)
        
        """"
        legend=dict(orientation='h',
                    xanchor='right',
                    x=1,
                    yanchor='bottom',
                    y=1)
        -----
        df2['分類'] = df2['分類'].astype(str)
        fig = go.Figure(data=go.Scatter(x=df2['収益率の分散'],
                                y=df2['収益率'],
                                mode='markers',
                                marker_color=df2['分類'],
                                text=df2['投資比率'])) # hover text goes here
        fig.update_layout(height=800,width=1500,
                          title='モンテカルロシミュレーション結果',
                          xaxis={'title': '収益率の分散'},
                          yaxis={'title': '収益率'})
        st.plotly_chart(fig)
        
        """
        



@st.cache(allow_output_mutation=True)
def button_states():
    return {"pressed": None}

def session_change():
    if "is_pressed" in st.session_state:
        st.session_state["is_pressed"].update({"pressed": None})


@st.cache
def path_to_df_all_company_list(path):
    df_all_company_list = pd.read_excel(path)
    df_all_company_list = df_all_company_list.replace('-', np.nan)
    df_all_company_list['コード&銘柄名'] = df_all_company_list['コード'].astype(str)+df_all_company_list['銘柄名']
    return df_all_company_list


#####銘柄確定させる
#@st.cache(allow_output_mutation=True)
def selections_to_selected_company_list_and_selected_company_list_hyouji(df_all_company_list,selections):
    df_meigarasenntaku_temp = df_all_company_list[df_all_company_list['コード&銘柄名'].isin(selections)]
    selected_company_list = [str(i)+'.JP' for i in df_meigarasenntaku_temp['コード']]
    d = deque(selections)
    d.appendleft('Date')
    selected_company_list_hyouji = list(d)
    return df_meigarasenntaku_temp, selected_company_list, selected_company_list_hyouji

#@st.cache(allow_output_mutation=True)
#@st.cache
def selected_company_list_to_get_df(selected_company_list,selected_company_list_hyouji,duration):
    end = dt.datetime.now()
    start = end-dt.timedelta(days=duration*365)
    for i in range(len(selected_company_list)):
        code = selected_company_list[i]

        stooq = StooqDailyReader(code, start=start, end=end)
        df = stooq.read()  # pandas.core.frame.DataFrame

        df_price = df['Close']
        df_price = df_price.reset_index()

        df_tourakuritu = df['Close']
        df_tourakuritu = df_tourakuritu.pct_change(-1)
        df_tourakuritu = df_tourakuritu.reset_index()
        df_tourakuritu = df_tourakuritu.dropna()
        df_tourakuritu = df_tourakuritu.reset_index(drop=True)

        if i ==0:
          df_price_merged = df_price
          df_tourakuritu_merged = df_tourakuritu
        else:
          df_price_merged=pd.merge(df_price_merged, df_price, on='Date')
          df_tourakuritu_merged=pd.merge(df_tourakuritu_merged, df_tourakuritu, on='Date')
          
    df_price_merged = df_price_merged.set_axis(selected_company_list_hyouji, axis='columns')
    df_tourakuritu_merged = df_tourakuritu_merged.set_axis(selected_company_list_hyouji, axis='columns')
    df_price_merged['Date'] = df_price_merged['Date'].dt.round("D")
    df_tourakuritu_merged['Date'] = df_tourakuritu_merged['Date'].dt.round("D")
    return df_price_merged, df_tourakuritu_merged

if __name__ == "__main__":
    main()