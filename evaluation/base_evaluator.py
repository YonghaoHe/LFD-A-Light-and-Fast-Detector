# -*- coding: utf-8 -*-
# author: Yonghao He
# description: base class for evaluators


class Evaluator(object):

    def update(self, results):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
