#!/usr/bin/env bash

# Configuration initially used from conda-forge/conda-smithy (BSD 3-clause)

ROOT_DIR=$(cd "$(dirname "$0")/.."; pwd;)

docker info

config=$(cat <<CONDARC

channels:
 - conda-forge
 - defaults # As we need conda-build

show_channel_urls: true

conda-build:
 root-dir: /root_dir/build_artefacts

CONDARC
)

ENV_PATH=/root/pandas-test
REQUIREMENTS="numpy cython cmake boost arrow-cpp pytz python-dateutil"

cat << EOF | docker run -i \
                    -v ${ROOT_DIR}:/root_dir \
                    -a stdin -a stdout -a stderr \
                    condaforge/linux-anvil \
                    bash || exit $?

export PYTHONUNBUFFERED=1

set -ex

echo "$config" > ~/.condarc
# A lock sometimes occurs with incomplete builds. The lock file is stored in build_artefacts.
conda clean --lock

conda create -y -q -p $ENV_PATH python=3.5
source activate $ENV_PATH

conda install --yes --quiet conda-forge-build-setup
source run_conda_forge_build_setup

conda install -y -q $REQUIREMENTS

mkdir test-build
cd test-build

export ARROW_HOME=$ENV_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ENV_PATH/lib

cmake -DPANDAS_BUILD_CYTHON=off /root_dir || exit 1
make -j4 || exit 1
ctest || exit 1

EOF
