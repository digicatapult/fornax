#!/usr/bin/env python
import xmlrunner
import unittest
import os

if __name__ == '__main__':

    tests = unittest.TestLoader().discover('test')
    xmlrunner.XMLTestRunner(output=os.environ.get('CIRCLE_TEST_REPORTS','test-reports')).run(tests)