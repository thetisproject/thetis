"""
Submits runs to queue manager.
Will execute command with user defined arguments
'{mpi_exec} python run_test_suite.py {reso} -Re {Re} -p {p} {lim_arg}'

Usage:

# launch a single run with parameters
python submit_runs.py coarse -Re 0.1 -l

# test only (print script content on stdout)
python submit_runs.py coarse -Re 0.1 -l -t

# multiple values, run all combinations as independent jobs
python submit_runs.py coarse,medium -Re 0.1,1.0,2.5 -j 1.0
python submit_tests.py coarse -Re 0.1,1.0,2.5 -j 1.0,2.0

"""
from batch_script_lib import *

# cluster_params.initialize_from_file('tkarna_stampede.yaml')

duration = {
    'coarse': time_request(4, 0, 0),
    'coarse2': time_request(4, 0, 0),
    'medium': time_request(8, 0, 0),
    'medium2': time_request(8, 0, 0),
    'fine': time_request(24, 0, 0),
    'ilicak': time_request(8, 0, 0),
}

processes = {
    'coarse': 2,
    'coarse2': 4,
    'medium': 8,
    'medium2': 16,
    'fine': 32,
    'ilicak': 8,
}


def process_args(reso, reynolds_number, use_limiter, jump_diff_factor, poly_order,
                 mimetic, dev_test=False, test_only=False, verbose=False):
    """
    runs the following command
    """
    cmd = '{mpi_exec} python run_test_suite.py {reso} -Re {Re} {mim_arg} -p {p} {lim_arg}'

    if use_limiter:
        lim_arg = '-l'
    else:
        # use jump diffusion
        lim_arg = '-j {:}'.format(jump_diff_factor)
    mimetic_arg = '-m' if mimetic else ''
    space_str = 'RT' if mimetic else 'DG'

    t = duration[reso]
    nproc = processes[reso]
    queue = 'normal'
    limiter_str = 'lim' if use_limiter else 'j_dif'+str(jump_diff_factor)
    job_name = 'lck_{:}_p{:}{:}_Re{:}_{:}'.format(reso, poly_order, space_str, reynolds_number,
                                                  limiter_str)
    if dev_test:
        # do shorter run in development queue instead
        t = time_request(0, 20, 0)
        queue = 'development'
        nproc = 1  # need to run in serial to ensure all compiles go through
        job_name = 'dev_'+job_name

    j = batch_job(job_name=job_name, queue=queue,
                  nproc=nproc, time_req=t, log_file='log_'+job_name)

    j.append_new_task(cmd, reso=reso, Re=reynolds_number,
                      p=poly_order, lim_arg=lim_arg, mim_arg=mimetic_arg)
    # submit to queue manager
    submit_jobs(j, test_only=test_only, verbose=verbose)


def parse_command_line():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('reso_str', type=str,
                        help='resolution string (coarse, medium, fine)')
    parser.add_argument('-j', '--jump_diff_factor', default=1.0,
                        help='factor for jump diff')
    parser.add_argument('-l', '--use_limiter', action='store_true',
                        help='use slope limiter for tracers instead of diffusion')
    parser.add_argument('-p', '--poly_order', default=1,
                        help='order of finite element space (0|1)')
    parser.add_argument('-m', '--mimetic', action='store_true',
                        help='use mimetic elements for velocity')
    parser.add_argument('-Re', '--reynolds_number', default=2.0,
                        help='mesh Reynolds number for Smagorinsky scheme')
    parser.add_argument('--dev', action='store_true', default=False,
                        help='run short test in development queue to check integrity')
    parser.add_argument('-t', '--test_only', action='store_true', default=False,
                        help=('do not launch anything, just print submission '
                              'script on stdout'))
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='print submission script on stdout')
    args = parser.parse_args()

    def get_arg_list(a):
        if isinstance(a, str):
            return [s.strip() for s in a.split(',')]
        return [a]
    # parse lists (if any)
    reso_list = get_arg_list(args.reso_str)
    re_list = get_arg_list(args.reynolds_number)
    jd_list = get_arg_list(args.jump_diff_factor)
    p_list = get_arg_list(args.poly_order)

    # loop over all parameter combinations, launcing a job for each case
    for reynolds_number in re_list:
        for reso in reso_list:
            for jump_diff_factor in jd_list:
                for poly_order in p_list:
                    process_args(reso, reynolds_number,
                                 args.use_limiter, jump_diff_factor,
                                 poly_order, args.mimetic, dev_test=args.dev,
                                 test_only=args.test_only, verbose=args.verbose)


if __name__ == '__main__':
    parse_command_line()
