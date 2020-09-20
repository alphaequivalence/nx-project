import os
import timeit
import skopt
import hyperparameter
from config import Configuration as config
from utils import load_dataset, select_curves

from model import Model, train, eval


space = hyperparameter.space

dataset = load_dataset()
train_data = select_curves(dataset, CURVES_OF_INTEREST=['ATGPig', 'ATGcala'], TARGET_CURVE='ATG35pc')
# eval_data = select_curves(dataset, CURVES_OF_INTEREST=[], TARGET_CURVE='ATG35pc')


def objective(params):
    model = Model(params)
    train(model, train_data, config.max_training_epochs, params)
    # avg = eval(eval_data)
    return -avg


def runBayes():
    """ Launch bayesian optimization in order to tune Tensorflow model's
        hyperparameter
    """
    print('==================================================')
    print('Bayesian optimization using Gaussian processes ...')
    print('==================================================')


    start = timeit.default_timer()  # -----------------
    r = skopt.gp_minimize(
        objective,
        space,
        n_calls=config.N_CALLS,
        random_state=config.SEED,
        n_jobs=config.N_JOBS_bayes,
        verbose=True)
    stop = timeit.default_timer()   # -----------------
    print('Bayesian Optimization took')
    print(stop - start)


    # save the model to disk
    f = os.path.join(
        config.experimentsfolder,
        'bayesOptResults.sav')

    skopt.dump(r, open(f, 'wb'))

    print('OK')


if __name__ == '__main__':
    runBayes()
