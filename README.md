# Automatic Poetry Generation from Prosaic Text

v2.0

tim.vandecruys@kuleuven.be

www.timvandecruys.be


## Introduction

Charles/Sylvia is a system for automatic poetry generation, developed
at the Faculty of Arts, KU Leuven, Belgium. The system has been trained
on billions of words extracted from generic web texts; its rhyming
knowledge has been extracted from Wiktionary, and it automatically
learned an elementary notion of sense by looking at the context of words.

Sylvia writes in English, while Charles is French.

A previous version of this system was developed at IRIT, Toulouse.

## Examples

~~~
2020-07-05 23:42:52 nmfdim 1 (tendresse, joie, bonheur)

je sentais les larmes sur son visage
je le ressens au plus profond de mon cœur
merci de ta tendresse , pour ce partage
je t' aime d' amour , c' est un vrai bonheur

la douceur de tes mots me rend malade
tu es mon coeur , j' aime le silence
tu es ma joie dans mes nuits froides
tu me rappelle des souvenirs d' enfance

                                     - Charles
~~~

~~~
2020-07-05 23:44:53 nmfdim 13 (sorrow, longing, admiration)

it seemed as though he 'd never had a heart attack
after a moment , a sudden silence filled the room
oh , dear , the man said , his voice almost black
i smiled , admiring the sight of my hands in the bathroom

for a moment , i felt a sense of great pride
taking a deep breath , i roused myself to my feet
i closed my eyes , turning my gaze to the far side
i was restless , eager to see something to eat

                                     - Sylvia

~~~

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
n-gram model) are not included due to their large file size (2.6GB for
French, 3.4GB for English). In order to obtain a copy, send a mail to
tim.vandecruys@irit.fr

## Dependencies

Pytorch is the most important one; all dependencies are stipulated in
the file `environment.yml`, which can be used to create a suitable
Anaconda environment. Note that the poetry generation system heavily
relies on the Pytorch version of OpenNMT
(https://github.com/OpenNMT/OpenNMT-py), which equally needs to be
installed.

## Reference

Tim Van de Cruys. 2020. [Automatic Poetry Generation from Prosaic
Text](https://www.aclweb.org/anthology/2020.acl-main.223.pdf). In
*Proceedings of the 58th Annual Meeting of the Association for
Computational Linguistics (ACL)*, pp. 2471-2480.

~~~
@inproceedings{vandecruys2020automatic,
    title = "Automatic Poetry Generation from Prosaic Text",
    author = "Van de Cruys, Tim",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    pages = "2471--2480",
}
~~~
