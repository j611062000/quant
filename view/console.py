import sys
from repository.ticker import Ticker
from repository.model import load_model
from service.backtest import Backtester, MSE, DirectionalAccuracy, WinRate, TotalReturn

def print_help():
    print("Available commands:")
    print("  load <ticker_symbol>  - Load model and data for given ticker")
    print("  predict <days>        - Make predictions for next N days") 
    print("  backtest             - Run backtest on loaded model")
    print("  help                 - Show this help message")
    print("  exit                 - Exit the program")

def main():
    model = None
    ticker = None
    metrics = [MSE(), DirectionalAccuracy(), WinRate(), TotalReturn()]
    
    print("Stock Price Prediction Console")
    print("Type 'help' for available commands")
    
    while True:
        try:
            command = input("> ").strip().split()
            if not command:
                continue
                
            if command[0] == "exit":
                break
                
            elif command[0] == "help":
                print_help()
                
            elif command[0] == "load":
                if len(command) != 2:
                    print("Usage: load <ticker_symbol>")
                    continue
                    
                ticker_symbol = command[1]
                try:
                    ticker = Ticker(ticker_symbol)
                    model = load_model(f'models/{ticker_symbol}_model.keras')
                    print(f"Loaded model for {ticker_symbol}")
                except Exception as e:
                    print(f"Error loading model: {str(e)}")
                    
            elif command[0] == "backtest":
                if not model or not ticker:
                    print("Please load a model first using 'load <ticker_symbol>'")
                    continue
                    
                backtester = Backtester(strategy=model, metrics=metrics)
                results = backtester.backtest(
                    ticker=ticker,
                    start='2024-11-01',
                    end='2024-11-26'
                )
                
                for metric, value in results.items():
                    print(f"{metric}: {value:.4f}")
                    
            elif command[0] == "predict":
                if not model or not ticker:
                    print("Please load a model first using 'load <ticker_symbol>'")
                    continue
                    
                if len(command) != 2:
                    print("Usage: predict <days>")
                    continue
                    
                try:
                    days = int(command[1])
                    if days < 1:
                        raise ValueError("Days must be positive")
                        
                    # TODO: Implement prediction logic
                    print("Prediction functionality not yet implemented")
                    
                except ValueError as e:
                    print(f"Invalid number of days: {str(e)}")
                    
            else:
                print(f"Unknown command: {command[0]}")
                print("Type 'help' for available commands")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            
    print("\nGoodbye!")

if __name__ == '__main__':
    main()
