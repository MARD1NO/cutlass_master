### 1. Compile cutlass

cmake option: 

```shell
export CUDACXX=/usr/local/cuda-11.6/bin/nvcc

# because I use A100, change to your arch. 
cmake .. -DCUTLASS_NVCC_ARCHS=80 
```

here we can only make multihead_attention target: 
```cpp
make 47_fused_multi_head_attention_col_mask -j32
```

### 2. Run example: 
Enter in `cutlass/build` directory. 

```shell
./examples/47_fused_multi_head_attention_col_mask/47_fused_multi_head_attention_fixed_seqlen_col_mask \
    --head_number=12 \
    --batch_size=1 \
    --head_size=64 \
    --head_size_v=64 \
    --seq_length=512 \
    --seq_length_kv=512 \
    --causal=true \
    --mask_offset=128
```
