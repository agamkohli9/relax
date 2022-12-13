<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# <img src=https://raw.githubusercontent.com/apache/tvm-site/main/images/logo/tvm-logo-small.png width=128/> Open Deep Learning Compiler Stack

[Documentation](https://tvm.apache.org/docs) |
[Contributors](CONTRIBUTORS.md) |
[Community](https://tvm.apache.org/community) |
[Release Notes](NEWS.md)

[![Build Status](https://ci.tlcpack.ai/buildStatus/icon?job=tvm/main)](https://ci.tlcpack.ai/job/tvm/job/main/)
[![WinMacBuild](https://github.com/apache/tvm/workflows/WinMacBuild/badge.svg)](https://github.com/apache/tvm/actions?query=workflow%3AWinMacBuild)

Apache TVM is a compiler stack for deep learning systems. It is designed to close the gap between the
productivity-focused deep learning frameworks, and the performance- and efficiency-focused hardware backends.
TVM works with deep learning frameworks to provide end to end compilation to different backends.

_(See the [original README](https://github.com/apache/tvm/blob/main/README.md) for more information)_

## Setup

The following instructions have been adapted from [TVM's official "Installing TVM from source" guide](https://tvm.apache.org/docs/install/from_source.html#developers-get-source-from-github)

**Getting started**

Start by cloning our TVM fork.

```bash
git clone --recursive https://github.com/agamkohli9/relax
```

Update submodules

```bash
cd ./relax
git submodule init
git submodule update
```

**Build TVM**

```bash
# Setup build directory
mkdir build
cp cmake/config.cmake build

# Build TVM
make -C ./build
```

**Python package setup**

Update your python path to include your local build of TVM.

```bash
export TVM_HOME=/path/to/relax
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```

**Python scripting environment**
We recommend using a virtual environment. See the [official documentation here](https://docs.python.org/3/library/venv.html).

Create a virtual environment and install dependencies.

```bash
cd ./demo

# Create and activate a virtual environment
python -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```


## Project Structure

```
├── definitions.py
├── gen_lib.py                          <- generate compiled libs for benchmarking
├── gen_plots.py                        <- generate plots from compiled libs
├── gen_relax_ir.py                     <- generate optimized Relax IR from inputs
├── logger.py
├── plot.png
├── relax
│   ├── __init__py
│   ├── modules.py
│   └── optimize.py
├── relax-ir
│   ├── ModuleBasic-opt.relax
│   ├── ModuleBasic-raw.relax
|   ...
├── relay
│   ├── __init__.py
│   ├── config.py
│   └── lib.py
└── requirements.txt
```


## Optimizing models

For all of the following, ensure you are in the `./demo` directory with a python virtual environment active and all dependencies installed..

**Testing optimizations**

Run an optimization pass defined in `demo/relax/optimize.py` for all
example programs included in `demo/relax/modules.py`.

```bash
python3 ./gen_relax_ir.py
```

This produces two `.relax` files for each input (one containing the origininal, unmodified IR, and one containing the optimized IR). All generated `.relax` files are saved to `demo/relax-ir`.

_Example:_

<table>
<tr>
<th>Input</th>
<th>Output</th>
</tr>
<tr>
<td>

```bash
# ModuleBasic-raw.relax

@R.function
def foo() -> R.Tensor(None, dtype="int32", ndim=0):
    # block 0
    res: R.Tensor((), dtype="int32") = R.add(30, 40)
    return res
```

</td>
<td>

```bash
# ModuleBasic-opt.relax

@R.function
def foo() -> R.Tensor(None, dtype="int32", ndim=0):
    # block 0
    res: R.Tensor((), dtype="int32") = 70
    return 70
```

</td>
</tr>

