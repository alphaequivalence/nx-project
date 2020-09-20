from skopt.space import Real, Integer, Categorical


# define the hyperparameter search space
space = [
    Integer(
        low=4,
        high=32,
        name='units_1'),
    Categorical(
        categories=['relu', 'softmax', 'tanh'],
        name='activation_1'
    ),
   Real(
        low=0.1,
        high=0.5,
        prior='log-uniform',
        name='dropout_1'),
 
    Integer(
        low=4,
        high=32,
        name='units_2'),
    Categorical(
        categories=['relu', 'softmax', 'tanh'],
        name='activation_2'
    ),
   Real(
        low=0.1,
        high=0.5,
        prior='log-uniform',
        name='dropout_2'),
 
    Integer(
        low=4,
        high=32,
        name='units_3'),
    Categorical(
        categories=['relu', 'softmax', 'tanh'],
        name='activation_1'
    ),
   Real(
        low=0.1,
        high=0.5,
        prior='log-uniform',
        name='dropout_3'),

    Real(
        low=10**-5,
        high=10**-1,
        prior='log-uniform',
        name='learning_rate')
]
