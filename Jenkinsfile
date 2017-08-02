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
                        sh 'curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install || (cat firedrake-install.log && /bin/false)'
                        sh 'python3 ./firedrake-install --disable-ssh --minimal-petsc --adjoint'
                    }
                }
            }
        }
        stage('Install Thetis') {
            steps {
                timestamps {
                    sh '''
. build/firedrake/bin/activate
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
python $(which firedrake-clean)
python -mpytest -v test/ -n 4
'''
                }
            }
        }
        stage('Test Adjoint') {
            steps {
                timestamps {
                    sh '''
. build/firedrake/bin/activate
python $(which firedrake-clean)
python -mpytest -v test_adjoint/
'''
                }
            }
        }
    }
}
