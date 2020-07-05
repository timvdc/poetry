# Automatic Poetry Generation from Prosaic Text

v1.0

developed at IRIT, Toulouse

tim.vandecruys@irit.fr

www.timvandecruys.be


## Introduction

Charles/Sylvia is a system for automatic poetry generation, developed
within the MELODI group at IRIT, the research institute for computer
science in Toulouse. The system has been trained on billions of words
extracted from generic web texts; its rhyming knowledge has been
extracted from the French Wiktionary, and it automatically learned an
elementary notion of sense by looking at the context of words.

Sylvia writes in English, while Charles is French.

## Examples

## Installation and execution

1) Clone the git repository:


`git clone https://github.com/timvdc/poetry.git`

2) Create an anaconda (python3) environment with all the necessary
dependencies; an environment description is included in the
archive. The environment can be installed with the command:

`conda env create -f environment.yml`

3) Put the required model files (not included) in directory `data`

4) Once installed and model files in place, activate the environment,
and run python. A poem can then be written using the following
commands (for French):

~~~
import charles
p = charles.Poem()
p.write()
p.write(nmfDim=1)
~~~~

For English, replace `charles` with `sylvia`.

NOTE: Currently hard-coded for execution on the GPU.

## Model files

Model files (neural network parameters, rhyme dictionary, NMF model,
n-gram model) are not included due to their large file size
(3.7GB). In order to obtain a copy, send a mail to
tim.vandecruys@irit.fr

Currently, only models for French are available; models for both
English and Dutch are equally in the pipeline.

## Dependencies

Pytorch is the most important one; all dependencies are stipulated in
the file `environment.yml`, which can be used to create a suitable
Anaconda environment. Note that the poetry generation system heavily
relies on the Pytorch version of OpenNMT
(https://github.com/OpenNMT/OpenNMT-py), which equally needs to be
installed.
