pipeline {
    agent {
      docker {
        image 'firedrakeproject/firedrake-vanilla:latest'
        label 'firedrakeproject'
        args '-v /var/run/docker.sock:/var/run/docker.sock'
        alwaysPull true
      }
    }
    environment {
        PATH = "/usr/local/bin:/usr/bin:/bin"
        PETSC_CONFIGURE_OPTIONS="--download-mumps --download-scalapack --download-parmetis --download-metis"
        CC = "mpicc"
    }
    stages {
        stage('Clean') {
            steps {
                dir('build') {
                    deleteDir()
                }
            }
        }
        stage('Install Pyadjoint') {
            steps {
                sh 'mkdir build'
                dir('build') {
                    timestamps {
                        sh 'sudo chown -R jenkins /home/firedrake/firedrake/bin'
                        sh '''
. /home/firedrake/firedrake/bin/activate
firedrake-update --install pyadjoint
'''
                    }
                }
            }
        }
        stage('Install Thetis') {
            steps {
                timestamps {
                    sh '''
. /home/firedrake/firedrake/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
'''
                }
            }
        }
        stage('Lint') {
            steps {
                timestamps {
                    sh '''
. /home/firedrake/firedrake/bin/activate
make lint
'''
                }
            }
        }
        stage('Test') {
            steps {
                timestamps {
                    sh '''
. /home/firedrake/firedrake/bin/activate
python $(which firedrake-clean)
python -mpytest -v test/ -n 12
'''
                }
            }
        }
        stage('Test Adjoint') {
            steps {
                timestamps {
                    sh '''
. /home/firedrake/firedrake/bin/activate
python $(which firedrake-clean)
python -mpytest -v test_adjoint/
'''
                }
            }
        }
    }
}
