check:
	ruff format .
	ruff check --fix .

run:
	python main.py