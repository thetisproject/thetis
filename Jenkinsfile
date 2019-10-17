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
        PYOP2_CACHE_DIR = "/home/jenkins/.firedrake_cache/pyop2"
        FIREDRAKE_TSFC_KERNEL_CACHE_DIR = "/home/jenkins/.firedrake_cache/tsfc"
    }
    stages {
        stage('Clean') {
            steps {
                dir('build') {
                    deleteDir()
                }
                sh 'ls -l /home'
                sh 'ls -la /home/jenkins'
            }
        }
        stage('Install Pyadjoint') {
            steps {
                sh 'mkdir build'
                dir('build') {
                    timestamps {
                        sh '''
sudo -u firedrake /bin/bash << Here
whoami
cd /home/firedrake
. /home/firedrake/firedrake/bin/activate
firedrake-update --install pyadjoint || (cat firedrake-update.log && /bin/false)
chmod a+rwx /home/firedrake/firedrake/lib/python*/site-packages
chmod a+rwx /home/firedrake/firedrake/lib/python*/site-packages/easy-install.pth
chmod a+rwx /home/firedrake/firedrake/bin
install -d /home/firedrake/firedrake/.cache
chmod -R a+rwx /home/firedrake/firedrake/.cache
firedrake-status
Here
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
        stage('Test') {
            parallel {
                stage('Forward Test') {
                    steps {
                        timestamps {
                            sh '''
. /home/firedrake/firedrake/bin/activate
python -mpytest -v test/ -n 11
'''
                       }
                    }
                }
                stage('Test Adjoint') {
                    steps {
                        timestamps {
                            sh '''
. /home/firedrake/firedrake/bin/activate
python -mpytest -v test_adjoint/
'''
                        }
                    }
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
    }
}
