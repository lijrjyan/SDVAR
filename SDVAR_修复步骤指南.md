# SDVAR Parallel v1.0 修复步骤指南

## 🎯 目标
修复`sdvar_autoregressive_infer_cfg_parallel_v1`函数，确保它能够正确运行并实现预期的并行验证功能。

---

## 📋 修复优先级

### 🔴 优先级1: 关键bug修复 (立即修复)
1. **`_build_combined_query`函数** - 位置编码和CFG处理错误
2. **`_split_logits_by_stage`函数** - CFG维度分割错误  
3. **状态更新逻辑** - 避免f_hat双重更新

### 🟡 优先级2: 重要改进 (后续修复)
1. **`_get_attention_mask`函数** - 实现基础块级掩码
2. **错误处理改进** - 添加边界检查
3. **KV Cache优化** - 明确缓存管理

---

## 🛠️ 第一步：修复`_build_combined_query`函数

### 当前问题
```python
# 当前代码的问题位置
current_pos = sum(p**2 for p in state.patch_nums[:state.current_stage])
```

### 修复方案
在`models/var.py`中找到`_build_combined_query`函数（约第1224行），替换为：

```python
def _build_combined_query(self, draft_tokens: List[torch.Tensor], 
                         state, B: int) -> torch.Tensor:
    """修复版本的combined query构建"""
    if not draft_tokens:
        return torch.empty(2 * B, 0, 1024, device=state.draft_f_hat.device)
    
    if verbose := True:  # 调试开关
        print(f"[SDVAR] Building combined query for {len(draft_tokens)} stages")
    
    # 1. 正确计算位置偏移
    base_pos = sum(p**2 for p in state.patch_nums[:state.current_stage])
    if verbose:
        print(f"[SDVAR] Base position: {base_pos}, current_stage: {state.current_stage}")
    
    # 2. 构建完整输入序列
    all_embeddings = []
    
    # 3. 如果是第一阶段，需要添加first_token_map
    if state.current_stage == 0:
        # 获取第一层的token map
        first_l = self.target_model.first_l
        sos = state.target_cond_BD[:B]  # 只取前B个，不要CFG doubling
        
        first_token_map = (
            sos.unsqueeze(1).expand(B, first_l, -1) +
            self.target_model.pos_start.expand(B, first_l, -1) +
            state.target_lvl_pos[:1, :first_l].expand(B, -1, -1)
        )
        all_embeddings.append(first_token_map)
        current_pos = first_l
        
        if verbose:
            print(f"[SDVAR] Added first_token_map: {first_token_map.shape}")
    else:
        current_pos = base_pos
        # TODO: 添加之前已接受的tokens (Week 2功能)
        if verbose:
            print(f"[SDVAR] Skipping first layer, starting from position: {current_pos}")
    
    # 4. 处理每个draft token stage
    for stage_idx, tokens in enumerate(draft_tokens):
        current_stage = state.current_stage + stage_idx
        pn = state.patch_nums[current_stage]
        
        if verbose:
            print(f"[SDVAR] Processing stage {current_stage}, tokens: {tokens.shape}, pn: {pn}")
        
        # 正确的embedding路径：tokens -> VAE embedding -> word embedding
        vae_embedding = self.target_model.vae_quant_proxy[0].embedding(tokens)  # (B, pn*pn, Cvae)
        stage_embedding = self.target_model.word_embed(vae_embedding)  # (B, pn*pn, C)
        
        # 添加位置编码
        pos_embed = state.target_lvl_pos[:1, current_pos:current_pos + pn*pn].expand(B, -1, -1)
        stage_embedding = stage_embedding + pos_embed
        
        all_embeddings.append(stage_embedding)
        current_pos += pn * pn
        
        if verbose:
            print(f"[SDVAR] Stage {current_stage} embedding: {stage_embedding.shape}")
    
    # 5. 拼接所有embeddings并进行CFG doubling
    combined = torch.cat(all_embeddings, dim=1)  # B, total_tokens, C
    combined = combined.repeat(2, 1, 1)  # CFG doubling -> 2B, total_tokens, C
    
    if verbose:
        print(f"[SDVAR] Final combined query: {combined.shape}")
    
    return combined
```

### 测试第一步修复
在修复后，运行测试：

```python
# 在Colab中测试
from sdvar_test_framework import SDVARParallelV1Tester
tester = SDVARParallelV1Tester(sdvar_model)

# 只测试这个函数
result = tester.test_build_combined_query(B=2, gamma=2)
print("修复结果:", result)
```

---

## 🛠️ 第二步：修复`_split_logits_by_stage`函数

### 修复方案
找到`_split_logits_by_stage`函数（约第1288行），替换为：

```python
def _split_logits_by_stage(self, target_logits: torch.Tensor, 
                          draft_tokens: List[torch.Tensor], B: int, state) -> List[torch.Tensor]:
    """修复版本的logits分割"""
    if not draft_tokens:
        return []
    
    verbose = True  # 调试开关
    if verbose:
        print(f"[SDVAR] Splitting logits: {target_logits.shape}")
    
    # 1. 验证CFG维度
    expected_batch = 2 * B
    if target_logits.shape[0] != expected_batch:
        raise ValueError(f"Expected batch size {expected_batch}, got {target_logits.shape[0]}")
    
    # 2. 计算正确的起始位置
    base_pos = 0
    if state.current_stage == 0:
        base_pos = self.target_model.first_l  # 跳过第一层
    else:
        base_pos = sum(p**2 for p in state.patch_nums[:state.current_stage])
    
    if verbose:
        print(f"[SDVAR] Base position: {base_pos}")
    
    # 3. 分割每个stage的logits
    logits_per_stage = []
    current_pos = base_pos
    
    for stage_idx, tokens in enumerate(draft_tokens):
        current_stage = state.current_stage + stage_idx
        pn = state.patch_nums[current_stage]
        stage_length = pn * pn
        
        # 检查边界
        if current_pos + stage_length > target_logits.shape[1]:
            if verbose:
                print(f"[SDVAR] Warning: Stage {current_stage} exceeds logits length")
            break
        
        # 提取这个stage的logits
        stage_logits = target_logits[:, current_pos:current_pos + stage_length, :]
        logits_per_stage.append(stage_logits)
        
        if verbose:
            print(f"[SDVAR] Stage {current_stage} logits: {stage_logits.shape}")
        
        current_pos += stage_length
    
    return logits_per_stage
```

### 测试第二步修复
```python
# 测试logits分割
result = tester.test_split_logits_by_stage(B=2, gamma=2)
print("分割修复结果:", result)
```

---

## 🛠️ 第三步：修复状态更新逻辑

### 问题分析
当前`update_state_with_accepted_tokens`函数和主循环都在更新f_hat，导致不一致。

### 修复方案
修改主函数`sdvar_autoregressive_infer_cfg_parallel_v1`中的状态更新部分：

```python
# 在主函数中找到这个部分（约第1520行附近）
# 4. 更新推理状态
if accept_length > 0:
    # 注释掉这行，避免双重更新
    # self.update_state_with_accepted_tokens(draft_tokens, accept_length, state, B)
    
    # 直接在这里更新状态
    for stage_idx in range(accept_length):
        current_stage = state.current_stage + stage_idx
        pn = state.patch_nums[current_stage]
        
        # 获取这一层的tokens
        stage_tokens = draft_tokens[stage_idx]
        
        # 转换并更新f_hat
        h_BChw = self.draft_model.vae_quant_proxy[0].embedding(stage_tokens)
        h_BChw = h_BChw.transpose(1, 2).reshape(B, self.draft_model.Cvae, pn, pn)
        
        # 同时更新draft和target的f_hat
        state.draft_f_hat, _ = self.draft_model.vae_quant_proxy[0].get_next_autoregressive_input(
            current_stage, state.total_stages, state.draft_f_hat, h_BChw
        )
        state.target_f_hat = state.draft_f_hat.clone()  # 保持同步
    
    if verbose:
        print(f"[SDVAR] Successfully processed {accept_length} stages")
```

---

## 🛠️ 第四步：完整功能测试

### 运行完整测试
```python
# 运行所有测试
results = tester.run_all_tests(B=2, gamma=2)

# 分析结果
passed_tests = [r for r in results if r.passed]
failed_tests = [r for r in results if not r.passed]

print(f"\n📊 测试结果:")
print(f"✅ 通过: {len(passed_tests)}")
print(f"❌ 失败: {len(failed_tests)}")

if failed_tests:
    print(f"\n🔍 失败的测试:")
    for test in failed_tests:
        print(f"- {test.test_name}: {test.message}")
```

### 端到端测试
```python
# 测试实际推理
try:
    print("🧪 端到端推理测试")
    result_img = sdvar_model.sdvar_autoregressive_infer_cfg_parallel_v1(
        B=1,
        label_B=torch.tensor([980]),  # volcano class
        gamma=2,
        cfg=1.5,
        verbose=True
    )
    print(f"✅ 推理成功! 输出形状: {result_img.shape}")
    
    # 保存结果图像
    from torchvision.utils import save_image
    save_image(result_img, 'sdvar_parallel_v1_test.png')
    print("🖼️ 图像已保存为 sdvar_parallel_v1_test.png")
    
except Exception as e:
    print(f"❌ 端到端测试失败: {str(e)}")
    import traceback
    traceback.print_exc()
```

---

## 🔧 调试技巧

### 1. 启用详细日志
```python
# 在测试时启用verbose模式
result_img = sdvar_model.sdvar_autoregressive_infer_cfg_parallel_v1(
    B=1, label_B=torch.tensor([980]), gamma=2, verbose=True
)
```

### 2. 形状检查
```python
# 在关键位置添加形状检查
def debug_shapes(tensor, name):
    print(f"🔍 {name}: {tensor.shape} | device: {tensor.device} | dtype: {tensor.dtype}")

# 在函数中使用
debug_shapes(combined_query, "combined_query")
debug_shapes(target_logits, "target_logits")
```

### 3. 比较with test3
```python
# 对比新旧实现的结果
old_result = sdvar_model.sdvar_autoregressive_infer_cfg_sd_test3(
    B=1, label_B=torch.tensor([980]), entry_num=5, sd_mask=0
)
new_result = sdvar_model.sdvar_autoregressive_infer_cfg_parallel_v1(
    B=1, label_B=torch.tensor([980]), gamma=2
)

# 计算差异
diff = torch.abs(old_result - new_result).mean()
print(f"平均像素差异: {diff.item():.6f}")
```

---

## ✅ 成功标准

修复完成后，你应该能看到：

1. **测试通过**: 所有单元测试都通过
2. **推理成功**: 能够生成完整的图像
3. **性能提升**: Target调用次数减少到2-5次
4. **质量保持**: 与test3函数结果相似（差异<0.01）

---

## 🚀 下一步计划

完成基础修复后，Week 2可以实现：
- 动态γ调整策略
- 高级token匹配算法
- 更复杂的注意力掩码
- 性能优化和内存管理

---

**💡 提示**: 一次只修复一个函数，每次修复后都要测试，确保没有引入新的问题！ 

## 🛠️ 修复VAR对比测试

请将Cell 9.5中的代码修改为：

```python
# ===================== CELL 9.5: VAR基线对比测试 =====================
print("🔍 VAR基线对比测试")
print("="*50)

try:
    print("🧪 运行普通VAR推理...")
    
    # 使用sdvar_model中的target_model进行标准VAR推理
    var_result = sdvar_model.target_model.autoregressive_infer_cfg(
        B=1,
        label_B=torch.tensor([980]).to('cuda'),  # 相同的volcano class，确保在正确设备上
        cfg=1.5,
        g_seed=42  # 固定随机种子以便对比
    )
    
    print(f"✅ VAR推理成功!")
    print(f"📊 VAR输出形状: {var_result.shape}")
    print(f"📊 VAR数值范围: [{var_result.min():.3f}, {var_result.max():.3f}]")
    
    # 保存VAR基线结果
    from torchvision.utils import save_image
    save_image(var_result, 'var_baseline_test.png')
    print("🖼️ VAR基线图像已保存为 var_baseline_test.png")
    
    # 同时显示两个结果进行对比
    from PIL import Image
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # VAR基线结果
    var_img = Image.open('var_baseline_test.png')
    axes[0].imshow(var_img)
    axes[0].set_title('VAR基线 (标准推理)', fontsize=14)
    axes[0].axis('off')
    
    # SDVAR结果
    try:
        sdvar_img = Image.open('sdvar_parallel_v1_test.png')
        axes[1].imshow(sdvar_img)
        axes[1].set_title('SDVAR Parallel v1.0', fontsize=14)
        axes[1].axis('off')
    except FileNotFoundError:
        axes[1].text(0.5, 0.5, 'SDVAR图像未找到', ha='center', va='center')
        axes[1].set_title('SDVAR Parallel v1.0 (未生成)', fontsize=14)
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n📊 对比分析:")
    print(f"VAR基线:  形状{var_result.shape}, 范围[{var_result.min():.3f}, {var_result.max():.3f}]")
    
    # 如果SDVAR结果存在，进行数值对比
    try:
        # 假设sdvar_result是之前存储的结果
        if 'result_img' in locals():
            print(f"SDVAR v1:  形状{result_img.shape}, 范围[{result_img.min():.3f}, {result_img.max():.3f}]")
            
            # 计算L2距离
            l2_distance = torch.norm(var_result - result_img).item()
            print(f"🔍 L2距离: {l2_distance:.3f}")
            
            # 计算相关性
            correlation = torch.corrcoef(torch.stack([
                var_result.flatten(), 
                result_img.flatten()
            ]))[0, 1].item()
            print(f"🔍 像素相关性: {correlation:.3f}")
        else:
            print("SDVAR结果不可用，无法进行数值对比")
    except Exception as e:
        print(f"数值对比失败: {e}")
    
except Exception as e:
    print(f"❌ VAR对比测试失败: {str(e)}")
    import traceback
    print("详细错误信息:")
    traceback.print_exc()
    
    print("\n💡 修复建议:")
    print("1. 确保sdvar_model已正确加载")
    print("2. 检查target_model是否可访问: sdvar_model.target_model")
    print("3. 确保标签tensor在正确的设备上")
```

## 🚨 根本原因分析

从代码可以看出，我们在target验证时开启了KV cache：

```python
# 确保KV cache开启
for blk in self.target_model.blocks:
    blk.attn.kv_caching(True)
```

**KV Cache的工作机制**：
- 在自回归生成中，KV cache会缓存之前计算的key和value
- 当cache不为空时，模型可能只对新的token计算logits
- 这就解释了为什么6个token的输入只产生了5个token的输出！

## 🛠️ 诊断和修复方案

请在Colab中添加以下诊断代码来验证我的分析：

```python
# ===================== 诊断代码 =====================
print("🔍 KV Cache诊断")
print("="*50)

# 检查target模型的KV cache状态
print("Target模型KV Cache状态检查:")
for i, blk in enumerate(sdvar_model.target_model.blocks[:3]):  # 只检查前3个blocks
    if hasattr(blk.attn, 'k_cache') and blk.attn.k_cache is not None:
        print(f"  Block {i}: KV cache存在, k_cache shape: {blk.attn.k_cache.shape}")
    else:
        print(f"  Block {i}: KV cache为空")

# 关闭KV cache后重新测试
print("\n🔧 强制清空KV cache后重新测试...")
for blk in sdvar_model.target_model.blocks:
    blk.attn.kv_caching(False)
    # 强制清空cache
    if hasattr(blk.attn, 'k_cache'):
        blk.attn.k_cache = None
    if hasattr(blk.attn, 'v_cache'):
        blk.attn.v_cache = None

# 重新运行测试
from sdvar_test_framework import SDVARParallelV1Tester
tester = SDVARParallelV1Tester(sdvar_model)
result = tester.test_split_logits_by_stage(B=1, gamma=2)
print("清空cache后的测试结果:", result)
```

## 🛠️ 永久修复方案

问题在于我们需要在每次target验证前**完全重置KV cache**。请修改`target_verify_batch`函数：

```python
def target_verify_batch(self, draft_tokens: List[torch.Tensor], 
                       state, B: int) -> Tuple[List[torch.Tensor], int]:
    """target模型批量验证draft生成的tokens"""
    if not draft_tokens:
        return [], 0
    
    gamma = len(draft_tokens)
    print(f"[SDVAR] Target verifying batch: gamma={gamma}, current_stage={state.current_stage}")
    
    # 构建联合查询序列
    combined_query = self._build_combined_query(draft_tokens, state, B)
    print(f"[SDVAR] Combined query shape: {combined_query.shape}")
    
    # 计算适当的注意力掩码
    mask_length = combined_query.shape[1]
    attn_bias = self._get_attention_mask(mask_length, state.current_stage, gamma)
    
    # 🔧 关键修复：完全重置KV cache
    for blk in self.target_model.blocks:
        blk.attn.kv_caching(False)  # 先关闭
        # 强制清空已有的cache
        if hasattr(blk.attn, 'k_cache'):
            blk.attn.k_cache = None
        if hasattr(blk.attn, 'v_cache'):
            blk.attn.v_cache = None
        # 再重新开启（这样确保是干净的状态）
        blk.attn.kv_caching(True)
    
    # target前向计算
    state.target_calls += 1  # 统计调用次数
    print(f"[SDVAR] Target forward call #{state.target_calls}")
    
    x = combined_query
    for blk in self.target_model.blocks:
        x = blk(x=x, cond_BD=state.target_cond_BD, attn_bias=attn_bias)
    
    target_logits = self.target_model.get_logits(x, state.target_cond_BD)
    print(f"[SDVAR] After target forward: logits shape {target_logits.shape}")
    
    # 分割logits回对应的层
    logits_per_stage = self._split_logits_by_stage(target_logits, draft_tokens, B, state)
    
    # 应用CFG
    cfg_logits = []
    for stage_idx, stage_logits in enumerate(logits_per_stage):
        current_stage = state.current_stage + stage_idx
        ratio = current_stage / self.target_model.num_stages_minus_1
        t = state.cfg * ratio
        cfg_stage_logits = (1 + t) * stage_logits[:B] - t * stage_logits[B:]
        cfg_logits.append(cfg_stage_logits)
    
    print(f"[SDVAR] Target verification completed, generated {len(cfg_logits)} stage logits")
    return cfg_logits, gamma
```

请先运行诊断代码验证我的分析是否正确，然后我们可以应用这个修复方案。 

## 🔍 KV Cache诊断和修复方案

请先运行诊断代码验证我的分析是否正确，然后我们可以应用这个修复方案。

## 🔍 详细CFG诊断

从最新的错误信息看，问题可能在于CFG处理过程中的tensor维度不匹配。请运行以下详细诊断：

```python
# ===================== 详细CFG诊断代码 =====================
print("🔍 CFG和Tensor维度诊断")
print("="*50)

# 手动构建combined query来追踪问题
from sdvar_test_framework import SDVARParallelV1Tester
tester = SDVARParallelV1Tester(sdvar_model)

# 初始化state
state = sdvar_model._initialize_inference_state(B=1, label_B=torch.tensor([980]), g_seed=42, cfg=1.5, gamma=2)
print(f"初始化完成，state.target_cond_BD shape: {state.target_cond_BD.shape}")

# 生成draft tokens  
draft_tokens = sdvar_model.draft_generate_batch(state, B=1, verbose=False)
print(f"Draft tokens生成完成: {len(draft_tokens)} stages")
for i, tokens in enumerate(draft_tokens):
    print(f"  Stage {i}: {tokens.shape}")

# 手动构建combined query并跟踪每一步
print("\n🔍 手动构建Combined Query过程:")
try:
    # 调用_build_combined_query并捕获可能的错误
    combined_query = sdvar_model._build_combined_query(draft_tokens, state, B=1)
    print(f"✅ Combined query构建成功: {combined_query.shape}")
    
    # 检查state中相关变量的形状
    print(f"state.target_cond_BD shape: {state.target_cond_BD.shape}")
    print(f"state.target_lvl_pos shape: {state.target_lvl_pos.shape}")
    
    # 手动执行target forward过程
    print("\n🔍 Target Forward过程诊断:")
    
    # 重置KV cache
    for blk in sdvar_model.target_model.blocks:
        blk.attn.kv_caching(False)
        blk.attn.kv_caching(True)
    
    print(f"输入到target模型: {combined_query.shape}")
    
    # 执行target forward
    x = combined_query
    print(f"初始x: {x.shape}")
    
    # 逐个block检查
    for i, blk in enumerate(sdvar_model.target_model.blocks[:3]):  # 只检查前3个
        try:
            x_before = x.shape
            x = blk(x=x, cond_BD=state.target_cond_BD, attn_bias=None)
            print(f"Block {i}: {x_before} -> {x.shape}")
        except Exception as e:
            print(f"❌ Block {i} 失败: {str(e)}")
            break
    
    # 检查get_logits
    try:
        logits = sdvar_model.target_model.get_logits(x, state.target_cond_BD)
        print(f"✅ get_logits成功: {logits.shape}")
    except Exception as e:
        print(f"❌ get_logits失败: {str(e)}")
        print(f"x shape: {x.shape}")
        print(f"cond_BD shape: {state.target_cond_BD.shape}")
        
except Exception as e:
    print(f"❌ Combined query构建失败: {str(e)}")
    import traceback
    traceback.print_exc()
```

## 🛠️ CFG维度修复方案

基于诊断结果，问题可能在于`target_cond_BD`的维度处理。请检查以下修复：

```python
def _build_combined_query(self, draft_tokens: List[torch.Tensor], 
                        state, B: int) -> torch.Tensor:
    """修复版本的combined query构建 - 处理CFG维度问题"""
    if not draft_tokens:
        return torch.empty(2 * B, 0, 1920, device=state.draft_f_hat.device)
    
    verbose = True  # 调试开关
    if verbose:
        print(f"[SDVAR] Building combined query for {len(draft_tokens)} stages")
        print(f"[SDVAR] Input B={B}, state.target_cond_BD.shape={state.target_cond_BD.shape}")
    
    # 1. 正确计算位置偏移
    base_pos = sum(p**2 for p in state.patch_nums[:state.current_stage])
    if verbose:
        print(f"[SDVAR] Base position: {base_pos}, current_stage: {state.current_stage}")
    
    # 2. 构建完整输入序列
    all_embeddings = []
    
    # 3. 如果是第一阶段，需要添加first_token_map
    if state.current_stage == 0:
        # 获取第一层的token map
        first_l = self.target_model.first_l
        
        # 🔧 关键修复：确保sos只取前B个，避免CFG doubling问题
        sos = state.target_cond_BD[:B]  # 只取前B个，不要CFG doubling的部分
        if verbose:
            print(f"[SDVAR] sos shape after slicing: {sos.shape}")
        
        first_token_map = (
            sos.unsqueeze(1).expand(B, first_l, -1) +
            self.target_model.pos_start.expand(B, first_l, -1) +
            state.target_lvl_pos[:1, :first_l].expand(B, -1, -1)
        )
        all_embeddings.append(first_token_map)
        current_pos = first_l
        
        if verbose:
            print(f"[SDVAR] Added first_token_map: {first_token_map.shape}")
    else:
        current_pos = base_pos
        # TODO: 添加之前已接受的tokens (Week 2功能)
        if verbose:
            print(f"[SDVAR] Skipping first layer, starting from position: {current_pos}")
    
    # 4. 处理每个draft token stage
    for stage_idx, tokens in enumerate(draft_tokens):
        current_stage = state.current_stage + stage_idx
        pn = state.patch_nums[current_stage]
        
        if verbose:
            print(f"[SDVAR] Processing stage {current_stage}, tokens: {tokens.shape}, pn: {pn}")
        
        # 正确的embedding路径：tokens -> VAE embedding -> word embedding
        vae_embedding = self.target_model.vae_quant_proxy[0].embedding(tokens)  # (B, pn*pn, Cvae)
        stage_embedding = self.target_model.word_embed(vae_embedding)  # (B, pn*pn, C)
        
        # 添加位置编码
        pos_embed = state.target_lvl_pos[:1, current_pos:current_pos + pn*pn].expand(B, -1, -1)
        stage_embedding = stage_embedding + pos_embed
        
        all_embeddings.append(stage_embedding)
        current_pos += pn * pn
        
        if verbose:
            print(f"[SDVAR] Stage {current_stage} embedding: {stage_embedding.shape}")
    
    # 5. 拼接所有embeddings并进行CFG doubling
    combined = torch.cat(all_embeddings, dim=1)  # B, total_tokens, C
    combined = combined.repeat(2, 1, 1)  # CFG doubling -> 2B, total_tokens, C
    
    if verbose:
        print(f"[SDVAR] Final combined query: {combined.shape}")
    
    return combined
``` 