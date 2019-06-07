.PHONY: help clean dev docs package test

clean:
	rm -rf dist/*

dev:
	pip install -r requirements/dev.txt
	pip install -e .

package:
	python setup.py sdist
	python setup.py bdist_wheel