## Setup

### For Windows
#### Arrange folders:
```
<folder>/
├── .vscode/
|    ├── c_cpp_properties.json
|    └── tasks.json
└── <filename>.cu
```

#### Terminal
To set up terminal run (__IN CMD TERMINAL__):
```cmd
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" 
```
To compile programs:
```cl
nvcc main.cu -o main.exe
```
To run:
```cl
main.exe
```

### For Linux
- Cuda toolkit installed - reopening folder in WSL works too

To compile programs:
```cl
nvcc main.cu -o main.exe
```