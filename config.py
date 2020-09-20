import os
import datetime
from argparse import ArgumentParser


class Configuration:

    # version
    VERSION = '0.1'

    # mastering randomness
    SEED = 1

    # bayesian optimization meta-parameters
    N_CALLS = 20
    N_JOBS_bayes = -1

    # Hold-out ('ho') vs. Cross-validation ('cv')
    # VALIDATION = 'ho'
    VALIDATION = 'cv'
    # VALIDATION = 'pitfalls'

    # cross validation
    N_JOBS_cv = 1

    # debug
    DEBUG = True

    RUN = ''

    # where generated files, e.g. .mmap, slurm*, *.sav files are stored
    experimentsfolder = os.path.join('generated', VERSION)
    experiment_persistence = experimentsfolder
    bo_run_persistence = experiment_persistence

    # where extracted data live
    datafolder = os.path.join('generated', 'tmp')

    batch_size = 50

    max_training_epochs = 1000 
    min_training_epochs = 12
    max_subsequent_epochs = 8

    # hyperparameters for testing individual instantiations
    learning_rate = 0
    num_filters = 0
    kernel_sizes_list = []
    overlap = 0
    num_units_dense_layer = 0
    dropout = 0

    # metasegmented cross-validation parameters
    xval_nfolds = 5
    xval_metasegmentlength = 10

    @classmethod
    def parse_commandline(self, is_testing=False):
        """
        Synopsis
         Alter class attributes defined above with command line arguments
        """

        parser = ArgumentParser(description='')

        parser.add_argument(
            '--run',
            metavar='run',
            default='bayesopt',
            required=True
        )

        parser.add_argument(
            '--validation'
            , metavar='validation'
            , choices=(
                'ho'
                , 'cv'
                , 'pitfalls'
                , 'one-day-out'
                , 'one-user-out'
                , 'meta-segmented-cv'
            )
            , default='meta-segmented-cv'
            , required=False
        )

        parser.add_argument(
            '--batch-size'
            , metavar='batch_size'
            , type=int
            , default=50
            , required=False
        )

        # Hyperparamets instantiation by hand
        parser.add_argument(
            '--learning_rate'
            , metavar='learning_rate'
            , type=float
            , default=0.1
            # , required=is_testing
            , required=False
        )

        parser.add_argument(
            '--num_filters'
            , metavar='num_filters'
            , type=int
            , default=28
            # , required=is_testing
            , required=False
        )

        parser.add_argument(
            '--kernel_sizes_list'
            , metavar='kernel_sizes_list'
            , nargs='+'
            , type=int
            , default=[15,9,9, 9,15,9, 13,15,9, 10,14,12, 15,9,10, 9,9,12, 15,15,9]
            # , required=is_testing
            , required=False
        )

        parser.add_argument(
            '--overlap'
            , metavar='overlap'
            , type=float
            , default=0.6
            # , required=is_testing
            , required=False
        )

        parser.add_argument(
            '--num_units_dense_layer'
            , metavar='num_units_dense_layer'
            , type=int
            , default=2048
            # , required=is_testing
            , required=False
        )

        parser.add_argument(
            '--dropout'
            , metavar='dropout'
            , type=float
            , default=0.5
            # , required=is_testing
            , required=False
        )

        parser.add_argument(
            '--xval-nfolds'
            , metavar='xval_nfolds'
            , type=int
            , default=5
            , required=False
        )

        parser.add_argument(
            '--xval-metasegmentlength'
            , metavar='xval_metasegmentlength'
            , type=int
            , default=10
            , required=False
        )

        args = parser.parse_args()
        print('Args = %', args)

        self.RUN = args.run
        self.VALIDATION = args.validation
        self.batch_size = args.batch_size

        # hyper-parameters
        self.learning_rate = args.learning_rate
        self.num_filters = args.num_filters
        self.kernel_sizes_list = args.kernel_sizes_list
        self.overlap = args.overlap
        self.num_units_dense_layer = args.num_units_dense_layer
        self.dropout = args.dropout

        # metasegmented cross-validation parameters
        self.xval_nfolds = args.xval_nfolds
        self.xval_metasegmentlength = args.xval_metasegmentlength

        self.cmd_args = args

    @classmethod
    def __str__(cls):
        return ', '.join(
            '{}: {}\n'.format(k, v)
            for (k, v) in cls.__dict__.items()  # if k.startswith('_')
        )

    @classmethod
    def new_experiment(self):
        self.experiment_persistence = os.path.join(
            self.experiment_persistence, '{}'.format(datetime.datetime.now()))
        self.bo_run_persistence = self.experiment_persistence
        assert not os.path.exists(self.experiment_persistence)
        os.makedirs(self.experiment_persistence)
        print('results of this experiment can be found in %s', self.experiment_persistence)

    @classmethod
    def new_BO_run(self):
        self.bo_run_persistence = os.path.join(
            self.experiment_persistence, '{}'.format(datetime.datetime.now()))
        assert not os.path.exists(self.bo_run_persistence)
        os.makedirs(self.bo_run_persistence)