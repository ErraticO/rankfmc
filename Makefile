lint:
	python -m flake8 --ignore W3,E3,E5,E74 rankfmc/

test:
	python -m pytest -r Efp tests/