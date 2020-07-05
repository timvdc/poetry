#!/usr/bin/env python

from poembase import PoemBase

class Poem(PoemBase):

    def __init__(self, form='short', config='config/charles.json'):
        super().__init__(form=form, config=config)

