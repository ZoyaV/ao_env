# Simple GYM environment for adaptive optics - ao_env

### About project
This is a simple AO gym-environment based on the soapy library (https://github.com/AOtools/soapy).

The library includes two environments with different metrics for evaluating agent behavior.

MSE-reward enviroment 
gym.make('ao_env:ao-v0')

Image clarity-reward enviroment 
gym.make('ao_env:aoebright-v0')

### Instalation
These are the steps to install the environment:

!git clone https://github.com/openai/universe.git
!cd universe
!pip install -e ./universe

!git clone https://github.com/ZoyaV/ao_env.git
!pip install -e ./ao_env

