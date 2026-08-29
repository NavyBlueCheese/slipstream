PY ?= python

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m compileall -q src tests scripts

demo:
	$(PY) scripts/demo_resolution_ladder.py

.PHONY: test lint demo
