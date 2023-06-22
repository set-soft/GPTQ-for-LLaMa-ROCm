import random
random.seed(0)
import numpy as np
np.random.seed(0)

import torch
import torch.nn as nn
import torch.nn.functional as F

import quant_cuda
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def compute_err(name, ref, res):
    # Compute Mean Absolute Error (MAE)
    mae = F.l1_loss(res, ref)
    # Compute Root Mean Square Error (RMSE)
    mse = F.mse_loss(res, ref)
    rmse = torch.sqrt(mse)
    max_abs_error = (res - ref).abs().max().item()
    print(f'- {name} version: max abs {max_abs_error:.2e}, Mean Absolute Error (MAE): {mae:.2e} Root Mean Square Error (RMSE): {rmse:.2e}')


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0)

print('Benchmarking LLaMa-7B FC2 matvec ...')

DEV = torch.device('cuda:0')

B = 5
L = 128
M = 4096
N = 11008

DTYPE = torch.half
mat = torch.randn((M, N), device=DEV, dtype=DTYPE)
vec = torch.randn((B, M), device=DEV, dtype=DTYPE)
mul = torch.zeros((B, N), device=DEV, dtype=DTYPE)

COUNT = 1000
import time
tick = time.time()
for _ in range(COUNT):
    torch.matmul(vec, mat, out=mul) 
    torch.cuda.synchronize()
tref = (time.time() - tick) / COUNT
print('FP32:', tref)

DTYPE = torch.float
mat = mat.to(DTYPE)
vec = vec.to(DTYPE)
mul = mul.to(DTYPE)

mat = torch.randint(-1000000000, 1000000000, (M // 256 * 32, N), device=DEV, dtype=torch.int)
scales = torch.randn(N, device=DEV, dtype=DTYPE)
zeros = torch.randint(-1000000000, 1000000000, (1, N // 256 * 32), device=DEV, dtype=torch.int)

COUNT = 1000
import time
vec = vec.float()
tick = time.time()
for _ in range(COUNT):
    quant_cuda.vecquant2matmul(vec, mat, mul, scales, zeros, M)
    torch.cuda.synchronize()
tres = (time.time() - tick) / COUNT
print(f'2bit FP32: {(1-tres/tref)*100:5.2f}% {tres}')

vec = vec.half()
tick = time.time()
for _ in range(COUNT):
    quant_cuda.vecquant2matmul_faster(vec, mat, mul, scales, zeros, M, M//2)
    torch.cuda.synchronize()
tres = (time.time() - tick) / COUNT
print(f'2bit FP16: {(1-tres/tref)*100:5.2f}% {tres}')

vec = vec.float()
tick = time.time()
for _ in range(COUNT):
    quant_cuda.vecquant3matmul(vec, mat, mul, scales, zeros, M)
    torch.cuda.synchronize()
tres = (time.time() - tick) / COUNT
print(f'3bit FP32: {(1-tres/tref)*100:5.2f}% {tres}')

vec = vec.half()
tick = time.time()
for _ in range(COUNT):
    quant_cuda.vecquant3matmul_faster(vec, mat, mul, scales, zeros, M, M//2)
    torch.cuda.synchronize()
tres = (time.time() - tick) / COUNT
print(f'3bit FP16: {(1-tres/tref)*100:5.2f}% {tres}')

vec = vec.float()
tick = time.time()
for _ in range(COUNT):
    quant_cuda.vecquant4matmul(vec, mat, mul, scales, zeros, M)
    torch.cuda.synchronize()
tres = (time.time() - tick) / COUNT
print(f'4bit FP32: {(1-tres/tref)*100:5.2f}% {tres}')

vec = vec.half()
tick = time.time()
for _ in range(COUNT):
    quant_cuda.vecquant4matmul_faster(vec, mat, mul, scales, zeros, M, M//2)
    torch.cuda.synchronize()
tres = (time.time() - tick) / COUNT
print(f'4bit FP16: {(1-tres/tref)*100:5.2f}% {tres}')

vec = vec.float()
tick = time.time()
for _ in range(COUNT):
    quant_cuda.vecquant8matmul(vec, mat, mul, scales, zeros, M)
    torch.cuda.synchronize()
tres = (time.time() - tick) / COUNT
print(f'8bit FP32: {(1-tres/tref)*100:5.2f}% {tres}')
print('Verifiying kernel correctness ...')

M = 4096
N = 11008

from quant import *

layer = nn.Linear(M, N)
vec = torch.randn(B,L,M).to(DEV)

quantizer = Quantizer()
quantizer.configure(2, perchannel=True, sym=False, mse=False)
quantizer.find_params(layer.weight.data, weight=True)
layer.weight.data = quantize(
    layer.weight.data, quantizer.scale, quantizer.zero, quantizer.maxq
)

qlayer = QuantLinear(2, -1, layer.in_features, layer.out_features, kernel_switch_threshold = False)
qlayer.pack(layer, quantizer.scale, quantizer.zero)

qlayer = qlayer.to(DEV)
layer = layer.to(DEV)

print('2 bits:')
with torch.no_grad():
    ref = layer(vec)
    res = qlayer(vec)
    qlayer.faster = True
    res_half = qlayer(vec.half())
    compute_err('Normal', ref, res)
    compute_err('Faster', ref, res_half)

layer = nn.Linear(M, N)
vec = torch.randn(B,L,M).to(DEV)

quantizer = Quantizer()
quantizer.configure(3, perchannel=True, sym=False, mse=False)
quantizer.find_params(layer.weight.data, weight=True)
layer.weight.data = quantize(
    layer.weight.data, quantizer.scale, quantizer.zero, quantizer.maxq
)

qlayer = QuantLinear(3, -1, layer.in_features, layer.out_features, kernel_switch_threshold = False)
qlayer.pack(layer, quantizer.scale, quantizer.zero)

qlayer = qlayer.to(DEV)
layer = layer.to(DEV)

print('3 bits:')
with torch.no_grad():
    ref = layer(vec)
    res = qlayer(vec)
    qlayer.faster = True
    res_half = qlayer(vec.half())
    compute_err('Normal', ref, res)
    compute_err('Faster', ref, res_half)

layer = nn.Linear(M, N)
vec = torch.randn(B,L,M).to(DEV)

quantizer = Quantizer()
quantizer.configure(4, perchannel=True, sym=False, mse=False)
quantizer.find_params(layer.weight.data, weight=True)
layer.weight.data = quantize(
    layer.weight.data, quantizer.scale, quantizer.zero, quantizer.maxq
)

qlayer = QuantLinear(4, -1, layer.in_features, layer.out_features, kernel_switch_threshold = False)
qlayer.pack(layer, quantizer.scale, quantizer.zero)

qlayer = qlayer.to(DEV)
layer = layer.to(DEV) 

print('4 bits:')
with torch.no_grad():
    ref = layer(vec)
    res = qlayer(vec)
    qlayer.faster = True
    res_half = qlayer(vec.half())
    compute_err('Normal', ref, res)
    compute_err('Faster', ref, res_half)

layer = nn.Linear(M, N)
vec = torch.randn(B,L,M).to(DEV)

quantizer = Quantizer()
quantizer.configure(8, perchannel=True, sym=False, mse=False)
quantizer.find_params(layer.weight.data, weight=True)
layer.weight.data = quantize(
    layer.weight.data, quantizer.scale, quantizer.zero, quantizer.maxq
)

qlayer = QuantLinear(8, -1, layer.in_features, layer.out_features, kernel_switch_threshold = False)
qlayer.pack(layer, quantizer.scale, quantizer.zero)

qlayer = qlayer.to(DEV)
layer = layer.to(DEV)

print('8 bits:')
with torch.no_grad():
    ref = layer(vec)
    res = qlayer(vec)
    compute_err('Normal', ref, res)
