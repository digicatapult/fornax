import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install

# circleci.py version
exec(open("./fornax/version.py").read())
VERSION = __version__

with open('README.md') as fp:
    long_description = fp.read()

class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = 'verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)

setup(
    name='fornax',
    version=VERSION,
    license='Apache License 2.0',
    author='Daniel Staff',
    description='Approximate fuzzy subgraph matching in polynomial time',
    install_requires=[
        'psycopg2>=2.7.4',
        'SQLAlchemy>=1.2.8',
        'numpy>-1.14.5'
    ],
    author_email='daniel.staff@digicatapult.org.uk',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ],
    python_requires='>=3',
    cmdclass={
        'verify': VerifyVersionCommand,
    }
)
