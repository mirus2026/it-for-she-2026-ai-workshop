# IT For SHE Woman in Tech 2026 - AI/ML workshop

Input material for **Handwritten digit recognition using neural network** workshop.

A convolutional neural network (Keras/TensorFlow) that classifies handwritten digits (0-9) from the MNIST-style CSV dataset in `datasets/Train.csv`.
MNIST stands for Modified National Institute of Standards and Technology.

The project has two parts:

- `handwritten-digit-recognition-en.ipynb` - trains the model on `datasets/Train.csv` and saves it to `mnist_model.keras`.
- `mnist_ui_local_model.py` - a Tkinter paint app to draw a 28x28 digit and evaluate against local model.


## ML pipeline (Steps 1-10)

Handwritten digit recognition ML is organized into ten numbered steps that take the raw CSV all the way to a trained, saved model plus sample predictions.
The diagram below shows that flow end to end:

```mermaid
flowchart LR
    S1["Step 1: Import libraries<br/>numpy, pandas, sklearn, tensorflow, matplotlib"]
    S2["Step 2: Load and explore the dataset<br/>read datasets/Train.csv, split into X (pixels) and y (labels)"]
    S3["Step 3: Preprocess the data<br/>to numeric, fill missing, normalize /255, reshape to 28x28x1"]
    S4["Step 4: One-Hot encode the labels<br/>y to 10-class vectors via to_categorical"]
    S5["Step 5: Split the data<br/>train_test_split -> 80% train / 20% validation"]
    S6["Step 6: Build the neural network model<br/>Conv2D 32 -> Pool -> Conv2D 64 -> Pool -> Flatten -> Dense 128 -> Dropout -> Dense 10 Softmax"]
    S7["Step 7: Train the model<br/>model.fit, 10 epochs, batch size 32"]
    S8["Step 8: Evaluate the model<br/>evaluate on validation set, plot accuracy"]
    S9["Step 9: Save the trained model<br/>model.save('mnist_model.keras')"]
    S10["Step 10: Make predictions<br/>load datasets/test.csv, predict, show 5 samples"]

    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8 --> S9 --> S10
```

## Neural Network model configuration

The model is a Convolutional Neural Betwork (CNN) defined. 
A 28x28x1 grayscale image passes through two Conv2D + MaxPooling blocks that learn local spatial features, is flattened, 
then goes through a 128-unit ReLU dense layer with dropout and a 10-unit softmax output (one probability per digit 0-9).
It is compiled with the Adam optimizer and categorical cross-entropy loss.

```mermaid
flowchart LR
    IN["Input<br/>28 x 28 x 1 image"] --> C1["Conv2D 32<br/>3x3, ReLU"]
    C1 --> P1["MaxPooling2D<br/>2x2"]
    P1 --> C2["Conv2D 64<br/>3x3, ReLU"]
    C2 --> P2["MaxPooling2D<br/>2x2"]
    P2 --> FL["Flatten<br/>1600 units"]
    FL --> H1["Dense 128<br/>ReLU"]
    H1 --> DR["Dropout<br/>rate 0.5"]
    DR --> OUT["Dense 10<br/>Softmax"]
    OUT --> PRED["Predicted digit<br/>argmax over 0-9"]
```

Layer summary:

| Layer             | Output shape  | Params  | Activation |
| ----------------- | ------------- | ------- | ---------- |
| Input             | (28, 28, 1)   | 0       | -          |
| Conv2D (32, 3x3)  | (26, 26, 32)  | 320     | ReLU       |
| MaxPooling2D 2x2  | (13, 13, 32)  | 0       | -          |
| Conv2D (64, 3x3)  | (11, 11, 64)  | 18496   | ReLU       |
| MaxPooling2D 2x2  | (5, 5, 64)    | 0       | -          |
| Flatten           | (1600,)       | 0       | -          |
| Dense (hidden)    | (128,)        | 204928  | ReLU       |
| Dropout (0.5)     | (128,)        | 0       | -          |
| Dense (output)    | (10,)         | 1290    | Softmax    |

Total trainable parameters: 225034. Optimizer: Adam. Loss: categorical cross-entropy.

### Tensor-shape view (how the image flows through the CNN)

The diagram below traces how the data tensor changes shape as it moves through each
layer. Unlike a fully-connected network, a CNN keeps the 2D image structure through
the convolutional part, so it is clearer to show the tensor shape at each stage than
to draw individual neuron-to-neuron weights.

- `Conv2D` layers apply small learnable 3x3 filters (kernels) across the image,
  producing one feature map per filter (32 then 64). The spatial size shrinks by 2 in
  each direction because no padding is used (26 = 28 - 2, 11 = 13 - 2).
- `MaxPooling2D 2x2` halves the height and width, keeping the strongest activation in
  each 2x2 window.
- `Flatten` turns the final 5 x 5 x 64 feature maps into a single 1600-value vector.
- `Dense 128` + `Dropout(0.5)` learn combinations of those features (dropout is active
  only during training, to reduce overfitting).
- `Dense 10 (softmax)` outputs one probability per digit; `argmax` picks the winner.

```mermaid
flowchart LR
    IN["Input<br/>(28, 28, 1)"] -->|"Conv2D 32, 3x3, ReLU"| C1["Feature maps<br/>(26, 26, 32)"]
    C1 -->|"MaxPooling2D 2x2"| P1["(13, 13, 32)"]
    P1 -->|"Conv2D 64, 3x3, ReLU"| C2["Feature maps<br/>(11, 11, 64)"]
    C2 -->|"MaxPooling2D 2x2"| P2["(5, 5, 64)"]
    P2 -->|"Flatten"| F["Vector<br/>(1600,)"]
    F -->|"Dense 128, ReLU"| H["(128,)"]
    H -->|"Dropout 0.5"| D["(128,)"]
    D -->|"Dense 10, Softmax"| O["(10,)<br/>digit probabilities"]
    O -->|"argmax"| PRED["Predicted digit 0-9"]
```

## Requirements

- **Python 3.12 (64-bit)** is required. This project is pinned to Python **3.12.10**.
- Dependencies are pinned in [`requirements.txt`](requirements.txt):
  `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, plus `jupyterlab` / `ipykernel` for notebook use.
- `mnist_ui_local_model.py` uses only the Python standard library (`tkinter`), so it needs no dependencies and can run on any Python that ships Tk.

## Key notes

Here's some extra information:
* You need 2.5 GB free space on hard disk
* You should not have running application which uses TCP port 8888
* If you have additional anti-virus software please make sure that it will not block URL:  http://localhost:8888/lab
* On Microsoft Windows operating system Tk packages are available out of the box after Python 3.12 installation. For Linux operating system please install below OS packages:
```
sudo apt install -y libfontconfig1-dev libxft-dev tk-dev tcl-dev
```

## Prepare environment for AI/ML workshop

### Prepare environment for Microsoft Windows operating system

Fresh installation on Microsoft Windows 11 (no Python installation)

Open Terminal window. Type Terminal in Search textbox on taskbar:

![Windows Taskbar](docs/img/win-taskbar.png)

You will get:

![Terminal application](docs/img/TerminalApp.png)

Click icon or Open hyperlink:

![Terminal window](docs/img/TerminalWindow.png)

Type below command (or copy from here and paste to the Windows PowerShell terminal window):
```powershell
winget install Python.Python.3.12
```
and press Enter key:

![Winget command](docs/img/WingetPythonCmd.png)

Download started (it takes 1-2 minutes), soon installation starts:

![Python installation progress](docs/img/PythonInstallationProgress.png)

Installation takes few minutes:

![Python installation completed](docs/img/PythonInstallationCompleted.png)

Close Terminal window (click X or type exit and press Enter key).
Open it again:

![Terminal application](docs/img/TerminalWindow.png)

Type below command (or copy from here and paste to the Windows PowerShell terminal window):

```powershell
py -0p
```
![List of Python installations](docs/img/ListOfPythonVersions.png)

You should see line with version: 3.12.
Download git repository from GitHub web site.
Extract content of ZIP file.
Type below commands (or copy from here and paste to the Windows PowerShell terminal window):

```powershell
$zip = "$env:USERPROFILE\Downloads\it-for-she-2026-ai-workshop.zip"
Invoke-WebRequest -Uri "https://github.com/mirus2026/it-for-she-2026-ai-workshop/archive/refs/heads/main.zip" -OutFile $zip
Expand-Archive -Path $zip -DestinationPath "$env:USERPROFILE\Downloads" -Force
```

![Download and extract git repo](docs/img/DownloadExtractGitRepo.png)

Remark: You might type Enter twice to execute all commands.

Go to folder where files were extracted.
Type below command (or copy from here and paste to the Windows PowerShell terminal window):

```powershell
cd .\Downloads\it-for-she-2026-ai-workshop-main\
```

List all files and folders in extracted folder.
Type below command (or copy from here and paste to the Windows PowerShell terminal window):

```powershell
ls
```
You should see content like below:

![Extracted git repository](docs/img/ExtractedFolderContent.png)

Create virtual environment.
Type below command (or copy from here and paste to the Windows PowerShell terminal window):

```powershell
py -3.12 -m venv .venv
```
If you get nothing than command was successfully executed:

![Virtual environment created](docs/img/VirtualEnvCreated.png)

Activate virtual environment.
Type below command (or copy from here and paste to the Windows PowerShell terminal window):

```powershell
.\.venv\Scripts\Activate.ps1
```
You should see: (.venv) text:

![Virtual environment activated](docs/img/VirtEnvActivated.png)

Upgrade pip command to latest version.
Type below command (or copy from here and paste to the Windows PowerShell terminal window):

```powershell
python -m pip install --upgrade pip
```
![pip upgrade](docs/img/PipUpgrade.png)

Install python modules required for Machine Learning.
Type below command (or copy from here and paste to the Windows PowerShell terminal window):

```powershell
python -m pip install -r requirements.txt
```
Installation will take approx. 5-10 minutes:

![Python modules installed](docs/img/PythonModulesInstalled.png)

Now, you are prepared for workshop!

### Prepare environment for MacOS / Linux

Install pyenv first (https://github.com/pyenv/pyenv#installation)

```bash
curl -fsSL https://pyenv.run | bash
```

Add below lines to: $USER/.bashrc file:

export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

Restart your shell for the changes to take effect (logout / login).

Use pyenv to install Python 3.12.10 without touching the system Python:

```bash
pyenv install 3.12.10
```
Download and installation will take several minutes.
You should see:

```text
Downloading Python-3.12.10.tar.xz...
-> https://www.python.org/ftp/python/3.12.10/Python-3.12.10.tar.xz
Installing Python-3.12.10...
Installed Python-3.12.10 to /home/mirek/.pyenv/versions/3.12.10
```
You should select Python 3.12.10:

```bash
pyenv local 3.12.10
```

No output from command execution means that it was executed properly.

```bash
python --version
```

Last command should print: Python 3.12.10

Download git repository:

```bash
curl -L "https://github.com/mirus2026/it-for-she-2026-ai-workshop/archive/refs/heads/main.zip" -o it-for-she-2026-ai-workshop.zip
```

Extract content of ZIP file:

```bash
unzip it-for-she-2026-ai-workshop.zip
```

Go to folder where files were extracted:

```bash
cd it-for-she-2026-ai-workshop-main
```

Create virtual environment:

```bash
python -m venv .venv
```

Active virtual environment:

```bash
source .venv/bin/activate
```

Upgrade pip module:

```bash
python -m pip install --upgrade pip
```

Install Python modules:

```bash
python -m pip install -r requirements.txt
```

## Run the notebook interactively (JupyterLab)

`handwritten-digit-recognition-en.ipynb` is the step-by-step notebook version of the
training script. To run it interactively:

1. Start JupyterLab from the project root (it opens your browser at
   `http://localhost:8888/lab`; the terminal also prints the URL with an access token):

   On Microsoft Windows:

   ```powershell
   .\.venv\Scripts\jupyter lab
   ```
   On MacOS / Linux:

   ```bash
   .venv/bin/jupyter lab
   ```

You should see output like (example for Microsoft Windows operating system):

![Jupyter Lab console](docs/img/JupyterLabWindows.png)

Default web browser will be open with page:

![Jupyter Lab Launcher](docs/img/JupyterLabLauncher.png)

2. In the left file browser, double-click `handwritten-digit-recognition-en.ipynb`

![Jupyter Lab notebook opened](docs/img/JupyterLabNotebookOpened.png)


3. Confirm the kernel (top-right of the notebook) is **Python (handwritten-nn)** so the
   venv's packages are used. If it shows a different kernel, click it and switch.
4. Run cells in order (they depend on each other):
   - `Shift+Enter` runs the current cell and moves to the next.
   - `Ctrl+Enter` runs the current cell and stays.
   - Or use the "Run" menu -> "Run All Cells" to execute the whole notebook.

   Step 7 (training) is the slow cell - it runs the full training and may take a few
   minutes on CPU; a `[*]` next to the cell means it is still running. Plots render
   inline, no separate windows.

To stop the server, press `Ctrl+C` twice in the terminal where you launched it.

Note: running Steps 7 and 9 retrains the model and **overwrites `mnist_model.keras`**.
Back up that file first if you want to keep the current model.

### Clear notebook outputs

To strip all cell outputs (printed text, plots, execution counts) from the notebook
- for a clean file before committing - run:

```powershell
.\.venv\Scripts\jupyter nbconvert --clear-output --inplace handwritten-digit-recognition-en.ipynb
.\.venv\Scripts\jupyter nbconvert --clear-output --inplace handwritten-digit-recognition-pl.ipynb
```

## Run MNIST Paint UI demo application using local model

Execute below command:

```powershell
python mnist_ui_local_model.py
```

It will take couple of seconds to start because local model must be loaded to memory.

![MNIST Paint application](docs/img/MNIST-Paint-UI.png)

Draw digit in left panel and click Evaluate button:

![MNIST Paint 6 digit drawn](docs/img/MNIST-Paint-6digit.png)

You should see probability of predicted digit.


## Reference material

* [Handwritten Digit Recognition using Neural Network](https://www.geeksforgeeks.org/machine-learning/handwritten-digit-recognition-using-neural-network/)
