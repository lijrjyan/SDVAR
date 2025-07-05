#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDVAR并行验证 v1.0 测试框架
用于验证修复过程中的每个步骤
"""

# 环境检查
def check_environment():
    """检查运行环境"""
    try:
        import torch
        import numpy as np
        return True, "环境检查通过"
    except ImportError as e:
        return False, f"环境检查失败: {str(e)}"

from typing import List, Tuple, Optional, Dict, Any
import time
import traceback
from dataclasses import dataclass


@dataclass
class TestResult:
    """测试结果数据类"""
    test_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0


class SDVARTestFramework:
    """SDVAR测试框架 - 专门用于验证parallel_v1函数"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.test_results = []
        
    def log(self, message: str, level: str = "INFO"):
        """统一日志输出"""
        if self.verbose:
            prefix = f"[{level}]" if level else ""
            print(f"{prefix} {message}")
            
    def run_test(self, test_func, test_name: str, *args, **kwargs) -> TestResult:
        """运行单个测试"""
        self.log(f"🧪 运行测试: {test_name}")
        
        start_time = time.time()
        try:
            result = test_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if isinstance(result, bool):
                passed = result
                message = "测试通过" if passed else "测试失败"
                details = None
            elif isinstance(result, tuple) and len(result) >= 2:
                passed, message = result[:2]
                details = result[2] if len(result) > 2 else None
            else:
                passed = True
                message = str(result)
                details = None
                
            test_result = TestResult(
                test_name=test_name,
                passed=passed,
                message=message,
                details=details,
                execution_time=execution_time
            )
            
            self.test_results.append(test_result)
            status = "✅" if passed else "❌"
            self.log(f"{status} {test_name}: {message} ({execution_time:.3f}s)")
            
            return test_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            test_result = TestResult(
                test_name=test_name,
                passed=False,
                message=f"异常: {str(e)}",
                details={"traceback": traceback.format_exc()},
                execution_time=execution_time
            )
            
            self.test_results.append(test_result)
            self.log(f"❌ {test_name}: 异常 - {str(e)}")
            
            return test_result

    def print_summary(self):
        """打印测试摘要"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)
        
        print("\n" + "="*60)
        print(f"📊 测试摘要: {passed_tests}/{total_tests} 通过")
        print("="*60)
        
        for result in self.test_results:
            status = "✅" if result.passed else "❌"
            print(f"{status} {result.test_name}: {result.message}")
            
        if passed_tests == total_tests:
            print("🎉 所有测试通过！")
        else:
            print("⚠️  有测试失败，请检查详细信息")


class SDVARParallelV1Tester:
    """SDVAR parallel_v1 专用测试类"""
    
    def __init__(self, sdvar_model=None):
        self.sdvar_model = sdvar_model
        self.test_framework = SDVARTestFramework()
        
    def test_environment_check(self) -> Tuple[bool, str, Dict[str, Any]]:
        """测试环境检查"""
        env_ok, env_msg = check_environment()
        return env_ok, env_msg, {"environment": env_ok}
        
    def test_state_initialization(self, B: int = 2, gamma: int = 2) -> Tuple[bool, str, Dict[str, Any]]:
        """测试状态初始化"""
        try:
            if self.sdvar_model is None:
                return False, "SDVAR模型未初始化", {}
                
            env_ok, _ = check_environment()
            if not env_ok:
                return False, "环境检查失败", {}
                
            # 模拟状态初始化
            state = self.sdvar_model._initialize_inference_state(
                B=B, label_B=None, g_seed=42, cfg=1.5, gamma=gamma
            )
            
            # 验证关键属性
            checks = {
                "current_stage": state.current_stage == 0,
                "gamma": state.gamma == gamma,
                "total_stages": state.total_stages == 10,
                "patch_nums": len(state.patch_nums) == 10,
                "draft_f_hat_shape": state.draft_f_hat.shape == (B, 32, 16, 16),
                "target_f_hat_shape": state.target_f_hat.shape == (B, 32, 16, 16),
            }
            
            failed_checks = [k for k, v in checks.items() if not v]
            
            if failed_checks:
                return False, f"检查失败: {failed_checks}", checks
            else:
                return True, "状态初始化正确", checks
                
        except Exception as e:
            return False, f"初始化异常: {str(e)}", {}
    
    def test_draft_generate_batch(self, B: int = 2, gamma: int = 2) -> Tuple[bool, str, Dict[str, Any]]:
        """测试draft批量生成"""
        try:
            if self.sdvar_model is None:
                return False, "SDVAR模型未初始化", {}
                
            env_ok, _ = check_environment()
            if not env_ok:
                return False, "环境检查失败", {}
                
            # 需要导入torch来创建tensor
            import torch
            
            # 初始化状态
            state = self.sdvar_model._initialize_inference_state(
                B=B, label_B=torch.tensor([980, 437]), g_seed=42, cfg=1.5, gamma=gamma
            )
            
            # 调用draft生成
            draft_tokens = self.sdvar_model.draft_generate_batch(state, B, verbose=False)
            
            # 验证结果
            checks = {
                "返回列表": isinstance(draft_tokens, list),
                "长度正确": len(draft_tokens) <= gamma,
                "token形状": all(isinstance(t, torch.Tensor) for t in draft_tokens),
            }
            
            if draft_tokens:
                first_token = draft_tokens[0]
                pn = state.patch_nums[0]
                checks["首个token形状"] = first_token.shape == (B, pn * pn)
                
            failed_checks = [k for k, v in checks.items() if not v]
            
            if failed_checks:
                return False, f"检查失败: {failed_checks}", checks
            else:
                return True, f"生成{len(draft_tokens)}个stage的tokens", checks
                
        except Exception as e:
            return False, f"生成异常: {str(e)}", {}
    
    def test_build_combined_query(self, B: int = 2, gamma: int = 2) -> Tuple[bool, str, Dict[str, Any]]:
        """测试combined query构建"""
        try:
            if self.sdvar_model is None:
                return False, "SDVAR模型未初始化", {}
                
            env_ok, _ = check_environment()
            if not env_ok:
                return False, "环境检查失败", {}
                
            import torch
            
            # 创建模拟的draft tokens
            state = self.sdvar_model._initialize_inference_state(
                B=B, label_B=torch.tensor([980, 437]), g_seed=42, cfg=1.5, gamma=gamma
            )
            
            # 生成测试用的draft tokens
            draft_tokens = []
            for i in range(gamma):
                stage_idx = state.current_stage + i
                if stage_idx >= len(state.patch_nums):
                    break
                pn = state.patch_nums[stage_idx]
                # 创建随机tokens
                tokens = torch.randint(0, 4096, (B, pn * pn), device=state.draft_f_hat.device)
                draft_tokens.append(tokens)
            
            # 调用函数
            combined_query = self.sdvar_model._build_combined_query(draft_tokens, state, B)
            
            # 验证结果
            expected_batch_size = 2 * B  # CFG doubling
            total_tokens = sum(state.patch_nums[i] ** 2 for i in range(len(draft_tokens)))
            
            checks = {
                "返回tensor": isinstance(combined_query, torch.Tensor),
                "batch维度": combined_query.shape[0] == expected_batch_size,
                "token维度": combined_query.shape[1] == total_tokens,
                "特征维度": combined_query.shape[2] == 1024,  # embed_dim
                "无NaN值": not torch.isnan(combined_query).any(),
                "无Inf值": not torch.isinf(combined_query).any(),
            }
            
            failed_checks = [k for k, v in checks.items() if not v]
            
            if failed_checks:
                return False, f"检查失败: {failed_checks}", checks
            else:
                return True, f"构建成功 {combined_query.shape}", checks
                
        except Exception as e:
            return False, f"构建异常: {str(e)}", {}
    
    def test_split_logits_by_stage(self, B: int = 2, gamma: int = 2) -> Tuple[bool, str, Dict[str, Any]]:
        """测试logits分割"""
        try:
            if self.sdvar_model is None:
                return False, "SDVAR模型未初始化", {}
                
            env_ok, _ = check_environment()
            if not env_ok:
                return False, "环境检查失败", {}
                
            import torch
            
            # 创建模拟数据
            state = self.sdvar_model._initialize_inference_state(
                B=B, label_B=torch.tensor([980, 437]), g_seed=42, cfg=1.5, gamma=gamma
            )
            
            # 模拟draft tokens
            draft_tokens = []
            total_tokens = 0
            for i in range(gamma):
                stage_idx = state.current_stage + i
                if stage_idx >= len(state.patch_nums):
                    break
                pn = state.patch_nums[stage_idx]
                tokens = torch.randint(0, 4096, (B, pn * pn), device=state.draft_f_hat.device)
                draft_tokens.append(tokens)
                total_tokens += pn * pn
            
            # 模拟target logits
            target_logits = torch.randn(2 * B, total_tokens, 4096, device=state.draft_f_hat.device)
            
            # 调用函数
            logits_per_stage = self.sdvar_model._split_logits_by_stage(
                target_logits, draft_tokens, B, state
            )
            
            # 验证结果
            checks = {
                "返回列表": isinstance(logits_per_stage, list),
                "长度匹配": len(logits_per_stage) == len(draft_tokens),
                "形状正确": True,  # 稍后详细检查
            }
            
            # 检查每个stage的logits形状
            for i, (stage_logits, draft_stage) in enumerate(zip(logits_per_stage, draft_tokens)):
                expected_tokens = draft_stage.shape[1]  # pn * pn
                if stage_logits.shape != (2 * B, expected_tokens, 4096):
                    checks["形状正确"] = False
                    break
            
            failed_checks = [k for k, v in checks.items() if not v]
            
            if failed_checks:
                return False, f"检查失败: {failed_checks}", checks
            else:
                return True, f"分割成功 {len(logits_per_stage)} stages", checks
                
        except Exception as e:
            return False, f"分割异常: {str(e)}", {}
    
    def test_basic_token_matching(self, B: int = 2, gamma: int = 2) -> Tuple[bool, str, Dict[str, Any]]:
        """测试基础token匹配"""
        try:
            if self.sdvar_model is None:
                return False, "SDVAR模型未初始化", {}
                
            env_ok, _ = check_environment()
            if not env_ok:
                return False, "环境检查失败", {}
                
            import torch
            
            # 创建模拟数据
            state = self.sdvar_model._initialize_inference_state(
                B=B, label_B=torch.tensor([980, 437]), g_seed=42, cfg=1.5, gamma=gamma
            )
            
            # 模拟perfect matching场景
            draft_tokens = []
            target_logits = []
            
            for i in range(gamma):
                stage_idx = state.current_stage + i
                if stage_idx >= len(state.patch_nums):
                    break
                pn = state.patch_nums[stage_idx]
                
                # 创建draft tokens
                tokens = torch.randint(0, 4096, (B, pn * pn), device=state.draft_f_hat.device)
                draft_tokens.append(tokens)
                
                # 创建完美匹配的target logits
                stage_logits = torch.randn(B, pn * pn, 4096, device=state.draft_f_hat.device)
                # 确保argmax匹配draft tokens
                stage_logits.scatter_(dim=2, index=tokens.unsqueeze(2), value=100.0)
                target_logits.append(stage_logits)
            
            # 调用函数
            accept_length = self.sdvar_model.basic_token_matching(
                draft_tokens, target_logits, state, B, similarity_threshold=0.5, verbose=False
            )
            
            # 验证结果
            checks = {
                "返回整数": isinstance(accept_length, int),
                "范围正确": 0 <= accept_length <= len(draft_tokens),
                "完美匹配": accept_length == len(draft_tokens),  # 我们设计的是完美匹配
            }
            
            failed_checks = [k for k, v in checks.items() if not v]
            
            if failed_checks:
                return False, f"检查失败: {failed_checks}", checks
            else:
                return True, f"匹配成功 {accept_length}/{len(draft_tokens)} stages", checks
                
        except Exception as e:
            return False, f"匹配异常: {str(e)}", {}
    
    def run_all_tests(self, B: int = 2, gamma: int = 2):
        """运行所有测试"""
        print("🚀 开始SDVAR parallel_v1 测试")
        print(f"📊 测试参数: B={B}, gamma={gamma}")
        print("="*60)
        
        # 按顺序运行所有测试
        tests = [
            (self.test_environment_check, "环境检查"),
            (self.test_state_initialization, "状态初始化测试"),
            (self.test_draft_generate_batch, "Draft批量生成测试"),
            (self.test_build_combined_query, "Combined Query构建测试"),
            (self.test_split_logits_by_stage, "Logits分割测试"),
            (self.test_basic_token_matching, "基础Token匹配测试"),
        ]
        
        for test_func, test_name in tests:
            if test_name == "环境检查":
                self.test_framework.run_test(test_func, test_name)
            else:
                self.test_framework.run_test(test_func, test_name, B, gamma)
            
        # 打印摘要
        self.test_framework.print_summary()
        
        return self.test_framework.test_results


def create_usage_example():
    """创建使用示例"""
    print("🔧 SDVAR Parallel v1.0 测试框架使用指南")
    print("="*60)
    print()
    print("1. 在Google Colab中使用:")
    print("```python")
    print("# 导入测试框架")
    print("from sdvar_test_framework import SDVARParallelV1Tester")
    print("from models.var import SDVAR")
    print()
    print("# 构建你的SDVAR模型")
    print("sdvar_model = SDVAR(draft_model, target_model)")
    print()
    print("# 运行测试")
    print("tester = SDVARParallelV1Tester(sdvar_model)")
    print("results = tester.run_all_tests(B=2, gamma=2)")
    print()
    print("# 检查结果")
    print("passed_tests = [r for r in results if r.passed]")
    print("failed_tests = [r for r in results if not r.passed]")
    print("print(f'通过: {len(passed_tests)}, 失败: {len(failed_tests)}')")
    print("```")
    print()
    print("2. 单独测试某个函数:")
    print("```python")
    print("# 只测试状态初始化")
    print("result = tester.test_state_initialization(B=1, gamma=1)")
    print("print(result)")
    print()
    print("# 只测试combined query构建")
    print("result = tester.test_build_combined_query(B=2, gamma=2)")
    print("print(result)")
    print("```")
    print()
    print("3. 调试失败的测试:")
    print("```python")
    print("for result in results:")
    print("    if not result.passed:")
    print("        print(f'失败测试: {result.test_name}')")
    print("        print(f'错误信息: {result.message}')")
    print("        if result.details:")
    print("            print('详细信息:', result.details)")
    print("```")


if __name__ == "__main__":
    create_usage_example() 