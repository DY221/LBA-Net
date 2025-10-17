=== Model Complexity ===
Params: 1.98 M
GFLOPs: 3.34 G

=== Performance Benchmark ===
GPU: Tesla T4
Baseline (FP32, batch=1): 138.39 FPS
FP16 (autocast, batch=1): 117.66 FPS
W1005 12:45:49.891000 5057 torch/_inductor/utils.py:1436] [0/0] Not enough SMs to use max_autotune_gemm mode
torch.compile (FP32, batch=1): 253.16 FPS
torch.compile + FP16 (batch=1): 424.60 FPS
torch.compile + FP16 (batch=4): 529.40 FPS → Throughput: 529.40 samples/sec

=== CPU Benchmark ===
CPU FPS: 8.61

=== Model Complexity ===
Params: 1.98 M
GFLOPs: 3.34 G

=== Performance Benchmark ===
GPU: NVIDIA L4
Baseline (FP32, batch=1): 127.81 FPS
FP16 (autocast, batch=1): 117.21 FPS
W1005 12:50:35.454000 1409 torch/_inductor/utils.py:1436] [0/0] Not enough SMs to use max_autotune_gemm mode
torch.compile (FP32, batch=1): 811.58 FPS
torch.compile + FP16 (batch=1): 1046.89 FPS
torch.compile + FP16 (batch=4): 1103.32 FPS → Throughput: 1103.32 samples/sec

=== CPU Benchmark ===
CPU FPS: 9.86

=== Model Complexity ===
Params: 1.98 M
GFLOPs: 3.34 G

=== Performance Benchmark ===
GPU: NVIDIA A100-SXM4-80GB
Baseline (FP32, batch=1): 126.34 FPS
FP16 (autocast, batch=1): 115.58 FPS
torch.compile (FP32, batch=1): 769.24 FPS
torch.compile + FP16 (batch=1): 778.59 FPS
torch.compile + FP16 (batch=4): 1943.44 FPS → Throughput: 1943.44 samples/sec

=== CPU Benchmark ===
CPU FPS: 8.58
