train:
	python -m service.model

backtest:
	python -m service.backtest

plot:
	python -m view.visualization $(ticker)

console:
	python -m view.console