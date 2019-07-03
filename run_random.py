#!/usr/bin/env python
# -*- coding: utf-8 -*-

import charles
import random
import sys

p = charles.Poem()

while True:
    draw = random.random()
    #print(draw)
    if draw < 0.2:
        p.write(nmfDim=None)
    else:
        p.write(nmfDim='random')

