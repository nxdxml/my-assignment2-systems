triton安装问题，需要手动挡
https://blog.csdn.net/yyywxk/article/details/144868136
https://hf-mirror.com/madbuda/triton-windows-builds 


$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:INCLUDE = "$env:CUDA_PATH\include;$env:INCLUDE"
$env:LIB = "$env:CUDA_PATH\lib\x64;$env:LIB"
$env:CC = "cl.exe"
