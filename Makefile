
do: #unittests
	lualatex -shell-escape PhD-thesis.tex

dependencies:
	cd deps/SCILP/ && git pull && cd ../../ # go there, git pull and come back
	cp deps/SCILP/scilp.tex deps/SCILP/scilp.bib SCILP/

unittests:
	cd backgrounds && PYTHONPATH=../deps/simulation-methods/src/ python3 -m doctest backgrounds.py
	cd backtracking && PYTHONPATH=../deps/competitive-programming/python-libs/ python3 -m doctest ECO.py #queens.py polyominoes.py

clean:
	rm -f *.aux *.idx *.lof *.log *.lot *.out *.pdf *.toc *.bbl *.blg

world: clean
	lualatex -shell-escape PhD-thesis.tex
	bibtex PhD-thesis
	lualatex -shell-escape PhD-thesis.tex
	lualatex -shell-escape PhD-thesis.tex
	lualatex -shell-escape PhD-thesis.tex

update-submodules:
	git pull
	git submodule foreach "(git checkout master; git pull)"
