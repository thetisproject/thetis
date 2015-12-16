"""
Submits runs to queue manager.
Will execute command with user defined arguments
'{mpiExec} python runTestSuite.py {reso} -Re {Re} -p {p} {limArg}'

Usage:

# launch a single run with parameters
python submitRuns.py coarse -Re 0.1 -l

# test only (print script content on stdout)
python submitRuns.py coarse -Re 0.1 -l -t

# multiple values, run all combinations as independent jobs
python submitRuns.py coarse,medium -Re 0.1,1.0,2.5 -j 1.0
python submitTests.py coarse -Re 0.1,1.0,2.5 -j 1.0,2.0

"""
from batchScriptLib import *

# clusterParams.initializeFromFile('tkarna_stampede.yaml')

duration = {
    'coarse': timeRequest(4, 0, 0),
    'coarse2': timeRequest(4, 0, 0),
    'medium': timeRequest(8, 0, 0),
    'medium2': timeRequest(8, 0, 0),
    'fine': timeRequest(24, 0, 0),
    'ilicak': timeRequest(8, 0, 0),
}

processes = {
    'coarse': 2,
    'coarse2': 4,
    'medium': 8,
    'medium2': 16,
    'fine': 32,
    'ilicak': 8,
}


def processArgs(reso, reynoldsNumber, useLimiter, jumpDiffFactor, polyOrder,
                mimetic, devTest=False, testOnly=False, verbose=False):
    """
    runs the following command
    """
    cmd = '{mpiExec} python runTestSuite.py {reso} -Re {Re} {mimArg} -p {p} {limArg}'

    if useLimiter:
        limArg = '-l'
    else:
        # use jump diffusion
        limArg = '-j {:}'.format(jumpDiffFactor)
    mimeticArg = '-m' if mimetic else ''
    spaceStr = 'RT' if mimetic else 'DG'

    t = duration[reso]
    nproc = processes[reso]
    queue = 'normal'
    limiterStr = 'lim' if useLimiter else 'jDif'+str(jumpDiffFactor)
    jobName = 'lck_{:}_p{:}{:}_Re{:}_{:}'.format(reso, polyOrder, spaceStr, reynoldsNumber,
                                                 limiterStr)
    if devTest:
        # do shorter run in development queue instead
        t = timeRequest(0, 20, 0)
        queue = 'development'
        nproc = 1  # need to run in serial to ensure all compiles go through
        jobName = 'dev_'+jobName

    j = batchJob(jobName=jobName, queue=queue,
                 nproc=nproc, timeReq=t, logFile='log_'+jobName)

    j.appendNewTask(cmd, reso=reso, Re=reynoldsNumber,
                    p=polyOrder, limArg=limArg, mimArg=mimeticArg)
    # submit to queue manager
    submitJobs(j, testOnly=testOnly, verbose=verbose)


def parseCommandLine():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('reso_str', type=str,
                        help='resolution string (coarse, medium, fine)')
    parser.add_argument('-j', '--jumpDiffFactor', default=1.0,
                        help='factor for jump diff')
    parser.add_argument('-l', '--useLimiter', action='store_true',
                        help='use slope limiter for tracers instead of diffusion')
    parser.add_argument('-p', '--polyOrder', default=1,
                        help='order of finite element space (0|1)')
    parser.add_argument('-m', '--mimetic', action='store_true',
                        help='use mimetic elements for velocity')
    parser.add_argument('-Re', '--reynoldsNumber', default=2.0,
                        help='mesh Reynolds number for Smagorinsky scheme')
    parser.add_argument('--dev', action='store_true', default=False,
                        help='run short test in development queue to check integrity')
    parser.add_argument('-t', '--testOnly', action='store_true', default=False,
                        help=('do not launch anything, just print submission '
                              'script on stdout'))
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='print submission script on stdout')
    args = parser.parse_args()

    def getArgList(a):
        if isinstance(a, str):
            return [s.strip() for s in a.split(',')]
        return [a]
    # parse lists (if any)
    ResoList = getArgList(args.reso_str)
    ReList = getArgList(args.reynoldsNumber)
    jdList = getArgList(args.jumpDiffFactor)
    pList = getArgList(args.polyOrder)

    # loop over all parameter combinations, launcing a job for each case
    for reynoldsNumber in ReList:
        for reso in ResoList:
            for jumpDiffFactor in jdList:
                for polyOrder in pList:
                    processArgs(reso, reynoldsNumber,
                                args.useLimiter, jumpDiffFactor,
                                polyOrder, args.mimetic, devTest=args.dev,
                                testOnly=args.testOnly, verbose=args.verbose)


if __name__ == '__main__':
    parseCommandLine()
