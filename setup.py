import itertools
from pathlib import Path
from worstpractices import gather_reqs

from setuptools import setup

HERE = Path(__name__).parent
PKG_NAME = 'worstpractices'

reqs, extras = gather_reqs(HERE / PKG_NAME)

setup(
    name=PKG_NAME,
    version='0.0.1',
    description='An opinionated library of Python practices',
    long_description=(HERE / 'README.md').read_text(),
    long_description_content_type='text/markdown',
    url='https://github.com/vadim0x60/worstpractices',
    author='Vadim Liventsev',
    author_email='hello@vadim.me',
    license='MIT',
    packages=[PKG_NAME],
    install_requires=reqs,
    extras_require=extras,
    python_requires='>=3.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries'
    ],
)