.PHONY: all draft submission wordcount clean

all: draft submission

draft:
	latexmk paper.tex

submission:
	latexmk submission.tex

wordcount:
	texcount -inc -total -sum tex/document.tex

clean:
	latexmk -C paper.tex submission.tex
	rm -rf build
