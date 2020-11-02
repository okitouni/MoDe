from setuptools import setup


setup(
    name='modeloss',
    version='0.1.2',    
    description='Moment Decorrelation for constrained NN outputs with respect to protected attributes. (Decorrelated (flat), linear, quadratic, etc.)',
    url='https://github.com/okitouni/MoDe',
    author='Ouail Kitouni',
    author_email='kitouni@gmail.com',
    license='BSD 2-clause',
    packages=['modeloss'],
    install_requires= ['numpy','scipy','torch','tensorflow'],
    python_requires='>=3.6',
    )
