
do: unittests
	pdflatex -shell-escape PhD-thesis.tex

dependencies:
	cd backtracking && wget https://raw.githubusercontent.com/massimo-nocentini/competitive-programming/master/python-libs/bits.py

unittests:
	cd backtracking && python3 -m doctest queens.py polyominoes.py
