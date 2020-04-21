"""
Launches simulations on HPC machines.

Run parameters are inherited from the master test case. However, here all
the parameters can take multiple values. All combinations of parameter values are
executed:

Example:

python submitRuns.py -r coarse medium -dt 25 50

will excecute runs
python mytestcase.py -r coarse -dt 25
python mytestcase.py -r coarse -dt 50
python mytestcase.py -r medium -dt 25
python mytestcase.py -r medium -dt 50

In addition to the options defined in the test case itself this script defines
options:
--test    - print job submission scripts without launching them
--dev     - run simulation in development queue for a short time for debugging
--verbose - print job submission scirpts and debug information

Depends on hpclauncher package: https://github.com/tkarna/hpclauncher
"""
import lockExchange as thetistestcase
from collections import OrderedDict
import argparse
import itertools
import hpclauncher


def get_timerequest(options):
    """
    User defined function that will return time allocation based on run options
    """
    default = hpclauncher.TimeRequest(2, 0, 0)
    devel = hpclauncher.TimeRequest(0, 20, 0)
    timedef = {
        'coarse': hpclauncher.TimeRequest(4, 0, 0),
        'coarse2': hpclauncher.TimeRequest(4, 0, 0),
        'medium': hpclauncher.TimeRequest(8, 0, 0),
        'medium2': hpclauncher.TimeRequest(8, 0, 0),
        'fine': hpclauncher.TimeRequest(24, 0, 0),
        'ilicak': hpclauncher.TimeRequest(8, 0, 0),
        '2000-1': hpclauncher.TimeRequest(4, 0, 0),
        '1000-1': hpclauncher.TimeRequest(4, 0, 0),
        '1000-2': hpclauncher.TimeRequest(4, 0, 0),
        '500-0.5': hpclauncher.TimeRequest(16, 0, 0),
        '500-1': hpclauncher.TimeRequest(10, 0, 0),
        '500-2': hpclauncher.TimeRequest(8, 0, 0),
        '500-5': hpclauncher.TimeRequest(8, 0, 0),
        '250-0.5': hpclauncher.TimeRequest(20, 0, 0),
        '250-1': hpclauncher.TimeRequest(16, 0, 0),
    }
    if 'dev' in options and options['dev'] is True:
        return devel
    return timedef.get(options['reso_str'], default)


def launch_run(scriptname, options, option_strings):

    timereqstr = options.pop('timereq', None)
    if timereqstr is not None:
        hh, mm, ss = timereqstr.split(':')
        t = hpclauncher.TimeRequest(int(hh), int(mm), int(ss))
    else:
        t = get_timerequest(options)

    # strip additional options
    nproc = options.pop('nproc', 1)
    queue = options.pop('queue', None)
    dev_test = options.pop('dev', False)
    test_only = options.pop('test', False)
    verbose = options.pop('verbose', False)
    # generate command for running python script
    optstrs = [' '.join([option_strings[k], str(v)]).strip() for
               k, v in options.items()]
    cmd = ['{mpiexec}', 'python', scriptname] + optstrs
    cmd = ' '.join(cmd)

    # generate HPC job submission script
    if queue is None:
        queue = 'normal'
    job_name = ''.join([scriptname.split('.')[0]] + [s.replace(' ', '') for
                                                     s in optstrs])
    if dev_test:
        # do shorter run in development queue instead
        queue = 'development'
        nproc = 1  # need to run in serial to ensure all compiles go through
        job_name = 'dev_'+job_name

    j = hpclauncher.BatchJob(jobname=job_name, queue=queue,
                             nproc=nproc, timereq=t, logfile='log_'+job_name)

    j.append_new_task(cmd)
    # submit to queue manager
    hpclauncher.submit_jobs(j, testonly=test_only, verbose=verbose)


def parse_options():
    """
    Parses all options that are inherited from the master python module.

    Adds a new option for number of cores to use
    """
    parser = thetistestcase.get_argparser()

    optionals = parser._option_string_actions
    store_opts = set()
    store_false_opts = set()
    store_true_opts = set()

    for k in optionals:
        action = optionals[k]
        label = action.dest
        if isinstance(action, argparse._StoreAction):
            store_opts.add((label, action))
        elif isinstance(action, argparse._StoreFalseAction):
            store_false_opts.add((label, action))
        elif isinstance(action, argparse._StoreTrueAction):
            store_true_opts.add((label, action))

    store_opts = dict(store_opts)
    store_true_opts = dict(store_true_opts)
    store_false_opts = dict(store_false_opts)

    # make a new parser for defining multiple runs
    batchparser = argparse.ArgumentParser()
    batchparser.add_argument('-nproc', type=int,
                             help='Set number of cores to use',
                             default=1,
                             nargs='+')
    batchparser.add_argument('--timereq', type=str,
                             help='force time request hh:mm:ss')
    batchparser.add_argument('--queue', type=str,
                             help='force submitting to given job queue')
    batchparser.add_argument('--dev', action='store_true',
                             help='Do a short run in development queue')
    batchparser.add_argument('--test', action='store_true',
                             help='Print generated job script but do not lauch it')
    batchparser.add_argument('--verbose', action='store_true',
                             help='Print generated job script and debug info')
    # copy conventional arguments, but enable setting lists
    for label, action in store_opts.items():
        batchparser.add_argument(*action.option_strings, type=action.type,
                                 help=action.help,
                                 dest=action.dest,
                                 default=action.default,
                                 nargs='+',
                                 choices=action.choices)
    # store true/false arguments will be morphed into list of 0|1
    for label, action in store_true_opts.items():
        batchparser.add_argument(*action.option_strings, type=int,
                                 help=action.help,
                                 dest=action.dest,
                                 default=0,
                                 nargs='+',
                                 choices=[0, 1])
    for label, action in store_false_opts.items():
        batchparser.add_argument(*action.option_strings, type=int,
                                 help=action.help,
                                 dest=action.dest,
                                 default=0,
                                 nargs='+',
                                 choices=[0, 1])

    args = batchparser.parse_args()
    args_dict = vars(args)

    def listify(a):
        if isinstance(a, list):
            return a
        return [a]

    label_to_optstr = dict((action.dest, action.option_strings[0]) for
                           action in batchparser._actions)

    # convert all arguments to [(label, key), ...] list
    all_args = []
    for action in batchparser._actions:
        label = action.dest
        if label == 'help' or args_dict[label] is None:
            continue
        value_list = listify(args_dict[label])
        value_list = list(OrderedDict.fromkeys(value_list))  # rm duplicates
        nargs = len(value_list)
        if label in store_true_opts.keys() | store_false_opts.keys():
            # if 1 use (label, ''), if 0 use ('', '')
            label_list = [label if i == 1 else '' for i in value_list]
            value_list = ['']*nargs
        else:
            label_list = [label]*nargs
        pairs = list(zip(label_list, value_list))
        all_args.append(listify(pairs))
    # get all combinations of options
    combinations_tuples = itertools.product(*all_args)
    # turn each combination into an option dict
    combinations = []
    for comb in combinations_tuples:
        comb_dict = OrderedDict()
        for k, v in comb:
            if len(k) > 0:
                comb_dict[k] = v
        if comb_dict:
            combinations.append(comb_dict)

    scriptname = thetistestcase.__name__ + '.py'
    for comb in combinations:
        launch_run(scriptname, comb, label_to_optstr)


if __name__ == '__main__':
    parse_options()
