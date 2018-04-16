from distutils.core import setup

with open('README.md') as fh:
    long_description = fh.read()


setup(name='django_tensor',
      version='0.0.1',
      author='Diwanshu Shekhar',
      author_email='diwanshu@gmail.com',
      url='https://github.com/DiwanshuShekhar/django_tensor',
      description ='A framework to deploy your trained tensorflow model in Django',
      long_description=long_description,
      requires=['numpy', 'tensorflow'],
      packages=['django_tensor']
      )
