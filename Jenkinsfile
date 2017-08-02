pipeline {
    agent {
        label 'linux'
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
        stage('Install Firedrake') {
            steps {
                sh 'mkdir build'
                dir('build') {
                    timestamps {
                        sh 'pip2 install virtualenv'
                        sh 'curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install'
                        sh 'python3 ./firedrake-install --disable-ssh --minimal-petsc --adjoint'
                        sh '$HOME/.local/bin/virtualenv --relocatable firedrake'
                    }
                }
            }
        }
        stage('Install Thetis') {
            steps {
                timestamps {
                    sh '''
. build/firedrake/bin/activate
pip install -r requirements.txt
pip install -e .
'''
                }
            }
        }
        stage('Lint') {
            steps {
                timestamps {
                    sh '''
. build/firedrake/bin/activate
make lint
'''
                }
            }
        }
        stage('Test') {
            steps {
                timestamps {
                    sh '''
. build/firedrake/bin/activate
export PYOP2_CACHE_DIR=${VIRTUAL_ENV}/pyop2_cache
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=${VIRTUAL_ENV}/tsfc_cache
firedrake-clean
py.test -v test/ -n 4
'''
                }
            }
        }
        stage('Test Adjoint') {
            steps {
                timestamps {
                    sh '''
. build/firedrake/bin/activate
export PYOP2_CACHE_DIR=${VIRTUAL_ENV}/pyop2_cache
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=${VIRTUAL_ENV}/tsfc_cache
firedrake-clean
py.test -v test_adjoint/
'''
                }
            }
        }
    }
}
