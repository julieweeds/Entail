from setuptools import setup
from disttest import test

# Import this to prevent spurious error: info('process shutting down')

setup(name='tdd-workshop',
      version='0.1',
      description='Test Driven Development workshop',
      #url='http://github.com/storborg/funniest',
      author='Daoud Clarke',
      #author_email='flyingcircus@example.com',
      license='MIT',
      packages=['sep'],
      cmdclass = {'test': test},
      options = {'test' : {'test_dir':['test']}}
      #zip_safe=False
      )
