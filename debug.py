import sys
import os

print("1. 当前 Python 位置:")
print(sys.executable)

print("\n2. Python 搜索路径:")
for p in sys.path:
    print(p)

print("\n3. 尝试导入:")
try:
    import langchain
    print(f"LangChain 位置: {langchain.__file__}")
    from langchain.chains import create_retrieval_chain
    print("成功导入 create_retrieval_chain！")
except ImportError as e:
    print(f"导入失败: {e}")
except Exception as e:
    print(f"其他错误: {e}")