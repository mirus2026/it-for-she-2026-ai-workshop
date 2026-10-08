# Tk / tkinter Troubleshooting (Linux, ParrotOS / Debian)

This document captures everything learned while getting the drawing UI in
`mnist_ui_local_model.py` to run on Linux. The UI uses Python's `tkinter`
module; if Python was built without Tk support you will hit a
`ModuleNotFoundError: No module named '_tkinter'` and the app won't start.

---

## 1. The symptom

Running the UI script fails at the `import tkinter` line:

```
Traceback (most recent call last):
  File ".../mnist_ui_local_model.py", line 39, in <module>
    import tkinter as tk
  File ".../3.12.10/lib/python3.12/tkinter/__init__.py", line 38, in <module>
    import _tkinter # If this fails your Python may not be configured for Tk
ModuleNotFoundError: No module named '_tkinter'
```

### What it means

`tkinter` is a thin Python wrapper around `_tkinter`, a C extension that is
compiled into the Python build. `_tkinter` only gets built if the Tcl/Tk
**development headers** (`tk-dev`, `tcl-dev`) are present *at the time Python is
compiled*. A pyenv-built interpreter created without those headers will have no
`_tkinter`, hence the error.

Key fact: `tkinter` is part of the Python standard library. It is **NOT** a pip
package and cannot be installed with `pip install tkinter` or listed in
`requirements.txt`.

---

## 2. How to check whether Tk is available

### 2a. Does *your* Python have Tk support? (the test that matters most)

Run with the exact interpreter you will use to run the app:

```bash
python -c "import tkinter; print(tkinter.TkVersion)"
```

- Prints a version (e.g. `8.6`)  -> Tk works in that Python.
- `ModuleNotFoundError: No module named '_tkinter'` -> that Python was built
  without Tk.

Visual self-test (opens a small Tk window):

```bash
python -c "import tkinter; tkinter._test()"
```

Underlying Tcl/Tk patch level:

```bash
python -c "import tkinter; r=tkinter.Tk(); print('Tcl', r.tk.call('info','patchlevel'))"
```

Because Tk support is per-build, test each interpreter you care about:

```bash
/usr/bin/python3 -c "import tkinter; print(tkinter.TkVersion)"   # system python
python -c "import tkinter; print(tkinter.TkVersion)"             # pyenv / active venv
which python && python --version
```

### 2b. Is the Python Tk package installed? (Debian / ParrotOS)

```bash
dpkg -l | grep -i python3-tk
dpkg -s python3-tk            # detailed status; look for "Status: install ok installed"
```

Fedora / RHEL equivalent:

```bash
rpm -q python3-tkinter
```

### 2c. Are the system Tcl/Tk runtime libraries present?

```bash
dpkg -l | grep -iE "libtk|libtcl"      # Debian / ParrotOS
ldconfig -p | grep -iE "libtk|libtcl"  # shared libs known to the linker
tclsh <<< 'puts [info patchlevel]'     # Tcl interpreter version, if tclsh installed
```

### 2d. Are the dev headers installed? (needed to build Python with Tk, e.g. pyenv)

```bash
dpkg -l | grep -iE "tk-dev|tcl-dev"
dpkg -s tk-dev tcl-dev 2>/dev/null | grep -E "Package|Status"
ls /usr/include/tk.h /usr/include/tcl.h 2>/dev/null   # header files on disk
```

Which check to use:
- "Will my app's `import tkinter` work?"  -> 2a with the exact interpreter.
- Debugging a pyenv build lacking `_tkinter`  -> 2d (headers must exist BEFORE
  building), then rebuild.
- Confirming OS packages landed after `apt install`  -> 2b and 2d.

---

## 3. Installing Tk support on ParrotOS / Debian

### 3a. The straightforward case

If you only need Tk for the **system** Python, this is usually enough:

```bash
sudo apt update
sudo apt install -y python3-tk
```

On this project's machine `python3-tk` was already installed, which means the
**system Python 3.11 already had working tkinter**. The `_tkinter` error only
appeared because the app was being run with a pyenv 3.12.10 build that lacked Tk.

### 3b. The dev headers (needed to rebuild pyenv Python with Tk)

```bash
sudo apt install -y tk-dev tcl-dev
```

---

## 4. The APT dependency conflict we hit (and the fix)

Installing the `-dev` packages failed with held/broken packages:

```
The following packages have unmet dependencies:
 uuid-dev : Depends: libuuid1 (= 2.38.1-5+deb12u1) but 2.38.1-5+deb12u3 is to be installed
E: Unable to correct problems, you have held broken packages.
```

### Root cause

The `-dev` chain pulled in `uuid-dev`, which demands an **exact version match**
of `libuuid1`. The installed `libuuid1` was a *newer* security point release
(`+deb12u3`) than the `uuid-dev` candidate expected (`+deb12u1`). The `-dev`
package in the enabled repos had not caught up to the newer point release, so
APT could not satisfy the `=` version pin. (The earlier messages blaming
`libfontconfig1-dev` / `libxft-dev` were misleading; those versions actually
matched fine. `uuid-dev` was the real blocker.)

### The fix that worked

Pin `uuid-dev` to the exact version of the installed `libuuid1`, then install
the Tk dev packages:

```bash
sudo apt install -y uuid-dev=2.38.1-5+deb12u3
sudo apt install -y libfontconfig1-dev libxft-dev tk-dev tcl-dev
```

> Adjust the version string to whatever `apt policy libuuid1` reports as
> *Installed* on your machine. Check with:
>
> ```bash
> apt policy uuid-dev libuuid1
> ```

### Other approaches that can help with similar held-package situations

```bash
sudo apt --fix-broken install        # repair partially-configured packages
sudo dpkg --configure -a             # finish interrupted installs
apt-mark showhold                    # list held packages (unhold with apt-mark unhold <pkg>)

# Let aptitude propose resolutions (better solver than apt for conflicts):
sudo apt install -y aptitude
sudo aptitude install tk-dev tcl-dev
# Prefer solutions that UPGRADE the conflicting packages.
# Reject any that remove core packages (util-linux, login, mount, etc.).
```

---

## 5. Making an existing pyenv Python pick up Tk

Installing `tk-dev` gives the system the Tk **headers**, but it does NOT
retroactively add `_tkinter` to a Python that was already built without it. You
must rebuild the interpreter so its build process finds the newly installed
headers:

```bash
pyenv uninstall 3.12.10
pyenv install 3.12.10
```

Then recreate the virtualenv and verify:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import tkinter; tkinter._test()"   # a small Tk window should appear
```

After that:

```bash
python mnist_ui_local_model.py
```

> Alternative: skip the pyenv rebuild entirely and run the project with the
> system Python 3.11 that already has Tk:
>
> ```bash
> /usr/bin/python3 -m venv .venv
> source .venv/bin/activate
> pip install -r requirements.txt
> python mnist_ui_local_model.py
> ```
>
> TensorFlow 2.21 provides wheels for Python 3.11, so this works for this
> project.

---

## 6. Per-platform tkinter install reference

| Platform                     | How to get tkinter                                   |
| ---------------------------- | ---------------------------------------------------- |
| Windows                      | Included with the official python.org installer      |
| Debian / Ubuntu / ParrotOS   | `sudo apt install python3-tk`                         |
| Fedora                       | `sudo dnf install python3-tkinter`                    |
| macOS                        | python.org installer; or `brew install python-tk`    |
| pyenv (any OS)               | Install `tk-dev` / `tcl-dev` BEFORE building, then `pyenv install <version>` |

Verify on any platform:

```bash
python -c "import tkinter; tkinter._test()"
```

---

## 7. Unrelated noise: the CUDA / absl warnings

When running the TensorFlow code you may also see:

```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:... cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
```

These are **informational, not errors**. TensorFlow prints them at import time
to say no NVIDIA GPU/CUDA was found, so it runs on CPU. For this workshop, CPU
is fine and the model still trains and runs.

### Why `TF_CPP_MIN_LOG_LEVEL=3` doesn't hide them

The `cudart_stub` line is emitted by TensorFlow's C++ layer during dynamic
library load, **before** absl's logger is initialized (that is exactly what the
first line says). So it is printed before `TF_CPP_MIN_LOG_LEVEL` or
`tf.get_logger().setLevel(...)` can take effect.

### Ways to suppress them (set BEFORE `import tensorflow`)

Recommended for CPU-only use — removes the root cause by hiding the GPU so TF
never probes for CUDA:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # TF won't look for any GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
```

Alternative — initialize absl early and raise its stderr threshold:

```python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.FATAL)
absl_logging.set_stderrthreshold("fatal")

import tensorflow as tf
```

Guaranteed but blunt — swallow file-descriptor-level stderr during the import
only (also hides any real import error, so keep it scoped tightly):

```python
import os, sys, contextlib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

@contextlib.contextmanager
def _silence_stderr():
    stderr_fd = sys.stderr.fileno()
    saved = os.dup(stderr_fd)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), stderr_fd)
        try:
            yield
        finally:
            os.dup2(saved, stderr_fd)
            os.close(saved)

with _silence_stderr():
    import tensorflow as tf
    from tensorflow.keras.models import load_model
```

---

## 8. Quick reference (TL;DR)

```bash
# 1. Check if the interpreter you'll use has Tk:
python -c "import tkinter; tkinter._test()"

# 2. If missing on Debian/ParrotOS system Python:
sudo apt install -y python3-tk

# 3. If building Python with pyenv, install headers FIRST:
sudo apt install -y uuid-dev=2.38.1-5+deb12u3        # match your libuuid1 version
sudo apt install -y libfontconfig1-dev libxft-dev tk-dev tcl-dev
pyenv uninstall 3.12.10 && pyenv install 3.12.10

# 4. Recreate venv and verify:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import tkinter; tkinter._test()"
```
