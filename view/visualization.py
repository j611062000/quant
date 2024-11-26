import bokeh
import bokeh.plotting
from pandas import DataFrame
from repository.ticker import Ticker

def plot_historical_data(ticker: Ticker):
    historical_data: DataFrame = ticker.historical_data(start='2024-01-01', end='2024-11-26')
    
    # Create a figure with a datetime x-axis
    p = bokeh.plotting.figure(x_axis_type='datetime', title=f'{ticker.symbol} Stock Price',
                            width=800, height=400)

    # Calculate candlestick positions
    inc: pd.Series[bool] = historical_data['Close'] > historical_data['Open']
    dec: pd.Series[bool] = ~inc
    
    print(historical_data)
    
    # Draw candlesticks
    # Green candlesticks for price increase
    p.segment(historical_data.index, historical_data['High'], historical_data.index, historical_data['Low'], color='green')
    p.vbar(historical_data.index[inc], 0.5, historical_data['Open'][inc], historical_data['Close'][inc], 
           fill_color='green', line_color='green')

    # Red candlesticks for price decrease 
    p.segment(historical_data.index, historical_data['High'], historical_data.index, historical_data['Low'], color='red')
    p.vbar(historical_data.index[dec], 0.5, historical_data['Open'][dec], historical_data['Close'][dec],
           fill_color='red', line_color='red')
    
    # Style the plot
    p.grid.grid_line_alpha = 0.3
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Price'
    
    # Show the plot
    bokeh.plotting.show(p)

if __name__ == '__main__':
    plot_historical_data(Ticker('2312.TW'))