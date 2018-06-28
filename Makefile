
do: unittests
	lualatex -shell-escape PhD-thesis.tex

dependencies:
	cd backtracking && wget https://raw.githubusercontent.com/massimo-nocentini/competitive-programming/master/python-libs/bits.py

unittests:
	cd backtracking && PYTHONPATH=../deps/competitive-programming/python-libs/ python3 -m doctest ECO.py #queens.py polyominoes.py

clean:
	rm -f *.aux *.idx *.lof *.log *.lot *.out *.pdf *.toc
