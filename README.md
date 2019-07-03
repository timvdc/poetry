# Charles poetry system for French

v0.11

developed at IRIT, Toulouse

tim.vandecruys@irit.fr
www.timvandecruys.be


## Introduction

Charles is a system for automatic poetry generation, developed within
the MELODI group at IRIT, the research institute for computer science
in Toulouse. Charles has been trained on billions of words extracted
from generic web texts; its rhyming knowledge has been extracted from
the French Wiktionary, and it automatically learned an elemntary
notion of sense by looking at the context of words.

NOTE: experimental, unclean, and badly documented development
version. At the moment, the code is not very clean and badly
documented; it will undergo significant rewrites in the following
months.

Currently hard-coded for execution on the GPU.

Currently, only models for French are available; models for both
English and Dutch are equally in the pipeline.

## Installation and execution

1) Clone the git repository:

git clone https://github.com/timvdc/poetry.git

2) Create an anaconda (python3) environment with all the necessary
dependencies; an environment description is included in the
archive. The environment can be installed with the command:

conda env create -f environment.yml

3) Once installed, activate the environment, and run python. A poem
can then be written using the following commands:

import charles
p = charles.Poem()
p.write()
p.write(nmfDim=1)

## dependencies

Pytorch is the most important one; all dependencies are stipulated
environment.yml, which can be used for a suitable Anaconda
environment.