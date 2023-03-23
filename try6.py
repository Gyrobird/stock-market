#getiing the data from yahoo finance
import yfinance as yf

#matplotlib for the plotting
import matplotlib.pyplot as plt


#srtaing and ending date
start_date = '2021-01-01'
end_date = '2021-12-31'

#seting the ticker
ticker = 'AAPL'

#getting the data
data  = yf.download(ticker, start_date, end_date)

# Display data in table form using pandas
data.tail()

#ploting the data
data['Adj Close'].plot()
plt.show()  #this is to show the plot

#sucess fully plotted the graph

