import ast
import subprocess
from typing import Tuple

from verl.workers.agent.tools.base_tool import BaseTool


class PythonTool(BaseTool):
    """Python代码执行工具，使用本地conda环境"""
    
    def __init__(self, conda_path: str, conda_env: str):
        """
        初始化Python工具
        
        Args:
            conda_path: conda安装路径
            conda_env: conda环境名称
        """
        self.conda_path = conda_path
        self.conda_env = conda_env
        self.python_path = f"{conda_path}/envs/{conda_env}/bin/python"

    @property
    def name(self) -> str:
        return "python_interpreter"
    
    @property
    def trigger_tag(self) -> str:
        return "python"
    
    def execute(self, code: str, timeout: int = 120) -> str:
        """执行Python代码并返回结果"""
        result, report = self._run_code(code, timeout)
        
        if report == "Done":
            return result
        else:
            return report
    
    def _run_code(self, code: str, timeout: int) -> Tuple[str, str]:
        """在conda环境中运行Python代码并返回结果和状态"""
        # 处理交互式代码
        code = self._preprocess_code(code)
        
        try:
            # 使用 subprocess.run 同步执行命令
            process = subprocess.run(
                [self.python_path, '-c', code],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False 
            )

            if process.returncode == 0:
                return process.stdout.strip(), "Done"
            else:
                return "", process.stderr.strip()

        except subprocess.TimeoutExpired:
            return "", f"执行超时（超过 {timeout} 秒）"
        except Exception as e:
            return "", f"执行异常: {str(e)}"
    
    def _preprocess_code(self, code: str) -> str:
        """
        预处理Python代码，处理交互式代码
        将最后一个表达式转换为print语句（如果不是print）
        """
        try:
            tree = ast.parse(code)
            if tree.body:
                last_expr = tree.body[-1]
                if isinstance(last_expr, ast.Expr):
                    # 仅当最后一个表达式不是print调用时才转换
                    if not (isinstance(last_expr.value, ast.Call) 
                            and isinstance(last_expr.value.func, ast.Name) 
                            and last_expr.value.func.id == 'print'):
                        print_call = ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id='print', ctx=ast.Load()),
                                args=[last_expr.value],
                                keywords=[]
                            )
                        )
                        tree.body[-1] = print_call
                        code = ast.unparse(tree)
        except:
            pass  # 保持原代码不变
        
        return code

def _test():
    batch_code = [
        """
# 创建符号变量
x = sympy.symbols('x')
y = sympy.symbols('y')

# 创建一个表达式
expr = x**2 + 2*x*y + y**2

print(f"Expression: {expr}")

# 求导
derivative = sympy.diff(expr, x)
print(f"Derivative with respect to x: {derivative}")

# 代入具体值
result = expr.subs([(x, 1), (y, 2)])
print(f"Value at x=1, y=2: {result}")
        """,
        """
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        print(np.array([1, 2, 3]))
        """
    ]
    
    async def run_test():
        # 创建Python工具实例
        python_tool = PythonTool(
            conda_path="/mmu_nlp_ssd/makai05/miniconda3/",  # 请根据实际conda安装路径修改
            conda_env="verl",              # 请根据实际环境名称修改
            max_concurrent=64
        )
        
        # 执行每个代码片段
        for i, code in enumerate(batch_code):
            print(f"\n--- 执行代码片段 {i+1} ---")
            result = await python_tool.execute(code)
            print(f"结果:\n{result}")
    
    # 运行测试
    asyncio.run(run_test())

if __name__ == "__main__":
    _test()


