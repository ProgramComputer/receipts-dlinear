from DModel import DModel
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Predicting monthly scanned receipts for 2022 from 2021", page_icon="🧾", layout="wide"
)


@st.cache_data()
def get_data():
    return pd.read_csv("receipts_daily.csv")



@st.cache_resource()
def get_chart(data):
    hover = alt.selection_single(
        fields=["date"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data, height=500, title="Daily # of scanned receipts for 2021 and 2022 prediction")
        .mark_line()
        .encode(
            x=alt.X("# Date", title="Date",axis=alt.Axis(labelAngle=-75)),
            y=alt.Y("Receipt_Count", title="# of receipts", scale=alt.Scale(domain=[data["Receipt_Count"].min(), data["Receipt_Count"].max()])),
            color="type"
        )
    )

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="# Date",#yearmonthdate()
            y="Receipt_Count",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("# Date", title="Date"),
                alt.Tooltip("Receipt_Count", title="# of receipts"),
            ],
        )
        .add_selection(hover)
    )

    return (lines + points + tooltips)


st.title("Predicting monthly scanned receipts for 2022 from 2021")

st.write("created by Paul")

n_epochs = st.slider("$$Epochs$$",value=300,max_value=1000)

train_percent = st.slider("$$Training\;Split$$",value=0.80,max_value=0.9,min_value=0.1)
source = get_data()

#https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/ used as reference

timeseries = source[["Receipt_Count"]].values.astype('float32')

# train-test split for time series
train_size = int(len(timeseries) * train_percent)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.from_numpy(X), torch.from_numpy(y)

lookback = 1
X_train, y_train = create_dataset(train, lookback)
X_test, y_test = create_dataset(test, lookback)
class Config:
  def __init__(self, pred_len, seq_len,kernel_size):
    self.pred_len = pred_len
    self.seq_len = seq_len
    self.kernel_size =  kernel_size
config = Config(lookback,lookback,1)



model =  DModel(config)
optimizer = optim.Adam(model.parameters())

loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 2e-3, epochs=n_epochs, steps_per_epoch=len(loader))

for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

u = torch.unsqueeze(X_test[-1],dim=1)
last_day = model(u)
receipts_npy = np.array([], dtype = np.float32)
model.eval()
def callModel(x):
  global receipts_npy
  with torch.no_grad():
   for _ in range(365):
      x = model(x)
      val = x.detach().numpy()
      receipts_npy = np.append(receipts_npy,val[:,-1,:].flatten().item())
  return x
callModel(last_day)

source["type"] = "past"

#Predict future months
date_range = pd.date_range(start='2022-01-01', periods=365, freq='D')
future_source = pd.DataFrame({'# Date': date_range, 'Receipt_Count': receipts_npy, 'type':"future"})


concat_source = pd.concat([source,future_source],axis=0)





chart = get_chart(concat_source)

#Display Chart
st.altair_chart(chart, use_container_width=True)

#Display table

months = ["January", "February", "March", "April", "May", "June", 
          "July", "August", "September", "October", "November", "December"]

source['# Date'] = pd.to_datetime(source['# Date'], errors='coerce')
source = source.rename(columns={source.columns[0]: 'Month',"Receipt_Count":"Number of receipts"})
past_table = source.groupby(source['Month'].dt.strftime('%B'))['Number of receipts'].sum().sort_values().to_frame()

past_table = past_table.reindex(months)

future_source['# Date'] = pd.to_datetime(future_source['# Date'], errors='coerce')
future_source = future_source.rename(columns={future_source.columns[0]: 'Month',"Receipt_Count":"Number of receipts"})

future_table = future_source.groupby(source['Month'].dt.strftime('%B'))['Number of receipts'].sum().sort_values().to_frame()

future_table = future_table.reindex(months)

future_table.loc[:, "Number of receipts"] = future_table["Number of receipts"].astype('int32').map('{:,d}'.format)
past_table.loc[:, "Number of receipts"] = past_table["Number of receipts"].map('{:,d}'.format)

st.write("## Monthly scanned receipts for 2021")
st.table(past_table)
st.write("## Predicted monthly scanned receipts for 2022")
st.table(future_table)


st.write("## Code")

st.write(
    "See more in my public [GitHub"
    " repository](https://github.com/ProgramComputer/receipts-dlinear)"
)
   
