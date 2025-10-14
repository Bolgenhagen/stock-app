
import yfinance as yf
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

tickers = {"Google":"GOOGL","UnitedHealth":"UNH", "META":"META", "IBM": "IBM", "Amazon": "AMZN"}

tick_dics={}

for name, keys in tickers.items():
    tick_dics[name] = yf.Ticker(keys)


hist_dic={}


for a,b in tick_dics.items():
    hist_dic[a] =  tick_dics[a].history(period="5y")

tick_dics["Google"].info


def add_parameters(company):
    hist_dic[company]["Country"] =  tick_dics[company].info.get("country", "Not Found")
    hist_dic[company]["Industry"] =  tick_dics[company].info.get("industry", "Not Found")
    hist_dic[company]["Year"] = hist_dic[company].index.year
    hist_dic[company]["Return (%)"] = (hist_dic[company]["Close"].pct_change())*100
    hist_dic[company]["Volatility (%)"] = ((hist_dic[company]["High"] - hist_dic[company]["Low"])/hist_dic[company]["Close"])*100
    hist_dic[company]["Volume Growth (%)"] = (hist_dic[company]["Volume"].pct_change())*100
    hist_dic[company]["Return_cumulative (%)"] = (1+ hist_dic[company]["Return (%)"]).cumprod() -1
    hist_dic[company]["Return_log (%)"]= np.log(hist_dic[company]["Close"]- hist_dic[company]["Close"].shift(-1))*100
    return None

for comp in tick_dics.keys():
    add_parameters(comp)


for comp in tick_dics.keys():
    hist_dic[comp].index=hist_dic[comp].index.date



for a,b in hist_dic.items():
    hist_dic[a]=b.dropna()
    hist_dic[a]=b.reset_index().rename(columns={"index":"Date"})


Rev_year={}
for a in  hist_dic.keys():
    Rev_year[a]= hist_dic[a].groupby("Year").agg(min_date=("Date", "min"),
                                                     max_date=("Date", "max")).reset_index()


min_Rev_year = {}
max_Rev_year= {}
max_return_cum={}
mean_return={}
std_return={}
years= Rev_year["Google"]["Year"].unique()
for a in tick_dics.keys():
        min_Rev_year[a] = []
        max_Rev_year[a] = []
        max_return_cum[a]= []
        mean_return[a]=[]
        std_return[a]=[]
        for b in years:
                ma=hist_dic[a]["Close"].loc[hist_dic[a]["Date"]==Rev_year[a]["max_date"].loc[Rev_year[a]["Year"]==b].iloc[0]].iloc[0]
                mi=hist_dic[a]["Close"].loc[hist_dic[a]["Date"]==Rev_year[a]["min_date"].loc[Rev_year[a]["Year"]==b].iloc[0]].iloc[0]
                min_Rev_year[a].append(mi)
                max_Rev_year[a].append(ma)
        max_return_cum[a].append(hist_dic[a]["Return_cumulative (%)"].max())
        mean_return[a].append(np.mean(hist_dic[a]["Return (%)"]))
        std_return[a].append(np.std(hist_dic[a]["Return (%)"]))

        

for a in Rev_year.keys():
    Rev_year[a]["Close_Min"]=  min_Rev_year[a]
    Rev_year[a]["Close_Max"]=  max_Rev_year[a]
    Rev_year[a]["Yearly_Return_(%)"]=  (Rev_year[a]["Close_Max"]/Rev_year[a]["Close_Min"]) -1    
Rev_year["Google"]


for a,b in hist_dic.items():
    b.set_index(hist_dic[a]["Date"], inplace=True)
    if "Date" in b.columns:
        b.drop("Date", axis=1, inplace=True) 

for a,b in Rev_year.items():
    b.set_index(Rev_year[a]["Year"], inplace=True)
    if "Year" in b.columns:
        b.drop("Year", axis=1, inplace=True) 



year_hist={}
for a,b in hist_dic.items():
    year_hist[a] = b.groupby("Year").agg(
        Close_avg=("Close","mean"),
        Volume_avg=("Volume","sum"),
        Volatility_avg=("Volatility (%)", "mean"),
        Volume_Growth_avg=("Volume Growth (%)", "mean")
    )


finance={} 
for a,b in tick_dics.items(): 
    finance[a] = tick_dics[a].financials.T[["Total Revenue", "Net Income", "Gross Profit"]]
balance={} 
for a,b in tick_dics.items(): 
    balance[a] = tick_dics[a].balance_sheet.T[["Total Liabilities Net Minority Interest", "Stockholders Equity"]]
cash={} 
for a,b in tick_dics.items(): 
    cash[a] = tick_dics[a].cashflow.T[["Operating Cash Flow", "Capital Expenditure"]]


final_finance= {}
for a in tick_dics.keys():
    df1=finance[a].merge(balance[a], left_index=True, right_index=True, how="inner")
    df2 = df1.merge(cash[a],left_index=True, right_index=True, how="inner")
    final_finance[a]=df2


def add_parameters2(company):
    final_finance[company]["Gross Margin"] = final_finance[company]["Gross Profit"]/final_finance[company]["Total Revenue"]
    final_finance[company]["Debt-to-Equity Ratio"] = final_finance[company]["Total Liabilities Net Minority Interest"]/final_finance[company]["Stockholders Equity"]
    return None

for comp in tick_dics.keys():
    add_parameters2(comp)


for c,ca in final_finance.items():
    final_finance[c] = ca.dropna()   




#TIME-SERIES with buttons
time_series = go.Figure()
for a,b in hist_dic.items():
    time_series.add_trace(go.Scatter(x=b.index, y=b["Close"],mode="lines",
        text=b.index, 
        name=a,
        hovertemplate='Date: %{text}<br>'+'Close: %{y:.2f}<extra></extra>'))




time_series.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=3, label='3y', step='year', stepmode='backward'),
                dict(step='all', label='All')
            ]),
            x=1,          
            xanchor='right',    
            y=1.11,              
            yanchor='top'

        )
    )

time_series.add_vline(x="2022-11-30", line_width=1.5, line_dash="dot", line_color="black")
time_series.add_annotation(
    x="2022-11-30",           
    y=1.10,                    
    yref="paper",              
    text="GPT-3.5",         
    showarrow=False,           
    font=dict(color="black", size=11))
time_series.add_vline(x="2023-03-14", line_width=1.5, line_dash="dot", line_color="black")
time_series.add_annotation(
    x="2023-03-14",           
    y=1.10,                    
    yref="paper",              
    text="GPT-4.0",         
    showarrow=False,           
    font=dict(color="black", size=11)
)

time_series.update_layout(
    title='Daily Close Values by Company',
    xaxis_title='Date',
    yaxis_title='Close',
    legend_title="Stocks"
)




#Bar
bar_serie = go.Figure()
for a,b in Rev_year.items():
    bar_serie.add_trace(go.Bar(x=b.index, y=b["Yearly_Return_(%)"],name=a, 
        hovertemplate='Date: %{x}<br>'+'Yearly Return: %{y:.2f}%<extra></extra>'))
bar_serie.add_annotation(
    x="2022",           
    y=1.1,                    
    yref="paper",              
    text="GPT-3.5",         
    showarrow=False,           
    font=dict(color="black", size=11)
)
bar_serie.add_annotation(
    x="2023",           
    y=1.1,                    
    yref="paper",              
    text="GPT-4",         
    showarrow=False,           
    font=dict(color="black", size=11)
)
bar_serie.update_layout(
    title=dict(text='Annual Return Fluctuations',x=0.42,
        xanchor='center'),
    xaxis_title='Year',
    yaxis_title='Yearly Return (%)',
    legend_title="Stocks"
)





final_finance["Google"].iloc[1,:3]



from sklearn.preprocessing import StandardScaler
final_finance_esc={}
for a,b in final_finance.items():
    scaler = StandardScaler()
    final_finance_esc[a] = scaler.fit_transform(b)


final_finance_esc.keys()


import plotly.graph_objects as go

companies = ['Google', 'UnitedHealth', 'META', 'IBM', 'Amazon']
years = [2024, 2023, 2022, 2021]  # row 0 = 2024, row 1 = 2023, etc.
theta = ["Total Revenue", "Net Income ", "Gross Profit","Capital Expenditure", "Debt-to-Equity Ratio"]


radar = go.Figure()
for company in companies:
    radar.add_trace(go.Scatterpolar(
        r=final_finance_esc[company][0, [0, 1, 2, 5, 8]],
        theta=["Total Revenue", "Net Income ", "Gross Profit","Capital Expenditure", "Debt-to-Equity Ratio"],
        fill="toself",
        name=company
    ))


buttons = []
for year_idx, year in enumerate(years):
    buttons.append(dict(
        label=str(year),
        method="update",
        args=[
            {"r": [final_finance_esc[company][year_idx, [0, 1, 2, 5, 8]] for company in companies]},
            {"title": f"Impact of Chat GPT on stocks of the {year}"}
        ]
    ))

radar.update_layout(
    title=dict(text=f"Company Performance Radar ({years[0]})", x=0.5,xanchor='center'),
    updatemenus=[dict(
        active=0,
        buttons=buttons,
        x=1.175,
        y=0.5
    )],
    polar=dict(radialaxis=dict(visible=True)),
    legend_title="Stocks"
)





copies={}
for a,b in final_finance.items():
    copies[a]=b
    b["Year"]=b.index.year

gross = go.Figure()

for a, b in copies.items():
    gross.add_trace(go.Bar(
        x=b["Year"],
        y=b["Gross Margin"],
        name=a,
        hovertemplate='Gross Margin: %{y:.2f}<extra></extra>'
    ))


gross.add_annotation(
    x=2022, y=1.09, yref="paper",
    text="GPT-3.5",
    showarrow=False,
    font=dict(color="black", size=11)
)
gross.add_annotation(
    x=2023, y=1.09, yref="paper",
    text="GPT-4",
    showarrow=False,
    font=dict(color="black", size=11)
)


gross.update_layout(
    title=dict(text='Gross Margin Dynamics Around ChatGPT',x=0.42,
        xanchor='center'),
    xaxis_title='Year',
    yaxis_title='Yearly Return (%)',
    legend_title="Stocks"
)






hist_list=list(hist_dic.values())
hist_years=pd.DataFrame(columns=[f"{i}" for i in hist_dic.keys()], index=hist_list[0].index)
for a,b in hist_dic.items():
    for c in hist_years.columns:
        if a==c:
            hist_years[a]=hist_dic[a]["Close"]



hist_years=hist_years.dropna()
hist_years.index = pd.to_datetime(hist_years.index)
def Month_return(year):
    hist_return=hist_years[hist_years.index.year==year]
    hist_return["month"]=hist_return.index.month
    months=hist_return["month"].unique()
    min_per_m= []
    max_per_m=[]

    for a in months:
        min_per_m.append(hist_return.index[hist_return["month"]==a].min())
        max_per_m.append(hist_return.index[hist_return["month"]==a].max())

    min_closes_m={}
    max_closes_m={}


    for a in hist_dic.keys():
        min_closes_m[a]=[]
        max_closes_m[a]=[]
        for b,c in zip(min_per_m,max_per_m):
            ma_m=round(float(hist_return[a].loc[hist_return.index==b].values),2)
            mi_m=round(float(hist_return[a].loc[hist_return.index==c].values),2)
            min_closes_m[a].append(ma_m)
            max_closes_m[a].append(mi_m)


    for a,b in hist_dic.items():
        for c in hist_return.columns:
            if a==c:
                hist_return[a]=hist_dic[a]["Close"]

    monthly_return=pd.DataFrame(columns=[f"{i}" for i in hist_dic.keys()], index=months)

    for a,b in hist_dic.items():
        for c in monthly_return.columns:
            if a==c:
                monthly_return[f"Max_{a}"]=max_closes_m[a]
                monthly_return[f"Min_{a}"]=min_closes_m[a]
                monthly_return[f"Mon_Return_{a}"] = (monthly_return[f"Max_{a}"]/monthly_return[f"Min_{a}"]) -1    



    for a in hist_dic.keys():
        for c in monthly_return.columns:
            if a==c:
                monthly_return.drop(c, axis=1,inplace=True)
                monthly_return.drop([f"Max_{a}"], axis=1,inplace=True)
                monthly_return.drop([f"Min_{a}"], axis=1,inplace=True)

    monthly_return.columns=monthly_return.columns.str.removeprefix("Mon_Return_")
    return monthly_return

df_2022=Month_return(2022)
df_2023=Month_return(2023)





app = Dash(__name__)
server = app.server 
app.layout = html.Div([

    html.H1("Impact of Chat GPT on Stock Market", style={'text-align': 'center'}),



    dcc.Graph(id='stock_time', figure=time_series ),
    html.Div([
        dcc.Graph(id='stock_bar', figure=bar_serie),
        dcc.Graph(id='stock_radar', figure=radar),

        ], style={'display': 'flex', 'justify-content': 'space-between'}),
    html.Div(
        dcc.Dropdown(id="slct_year",
                 options=[
                     {"label": "2021", "value": 2021},
                     {"label": "2022", "value": 2022},
                     {"label": "2023", "value": 2023},
                     {"label": "2024", "value": 2024}],
                 multi=False,
                 value=2021,
                 style={'width': "200px"}
                 ),style={'display': 'flex','justify-content': 'flex-end', 'margin': '10px 0'}),
    html.Div([
        dcc.Graph(id='stock_gross', figure=gross),
        dcc.Graph(id='heatmap', figure={}),

        ], style={'display': 'flex', 'justify-content': 'space-between'}),
    
])
@app.callback(
    Output(component_id='heatmap', component_property='figure'),
    Input(component_id='slct_year', component_property='value')
)
def update_graph(option_slctd):
    print(option_slctd)
    print(type(option_slctd))


    

    dff = Month_return(option_slctd)


    fig = px.imshow(dff,labels=dict(x="Companies", y="Month", color="Monthly Return"))
    fig.update_layout(
        title=dict(
            text=f"Patterns of Monthly Stock Performance ({option_slctd})",
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=16)  
        ))
    return fig


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=False)




