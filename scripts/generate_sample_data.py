"""
生成示例训练数据
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

output_dir = Path("data/stage2")
output_dir.mkdir(parents=True, exist_ok=True)

# 代码数据
code_samples = [
    "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + [pivot] + quick_sort(right)",
    "class BinarySearchTree:\n    def __init__(self):\n        self.root = None\n    def insert(self, val):\n        if not self.root:\n            self.root = TreeNode(val)",
    "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
] * 200  # 200 份

with open(output_dir / "code_sample.txt", 'w', encoding='utf-8') as f:
    for sample in code_samples:
        f.write(sample + '\n\n')

print(f"✓ 代码数据: {len(code_samples)} 条")

# 写作数据
writing_samples = [
    "清晨的阳光透过薄雾洒落在湖面上，波光粼粼。远山如黛，近水含烟，构成一幅宁静致远的山水画卷。",
    "秋日的午后，我独自坐在窗前，看着落叶随风起舞。每一片叶子都承载着季节的故事，轻轻诉说着时光的流逝。",
    "雨后的森林格外清新，泥土的芬芳混合着青草的气息。偶尔传来几声鸟鸣，更增添了几分幽静与祥和。",
] * 200

with open(output_dir / "writing_sample.txt", 'w', encoding='utf-8') as f:
    for sample in writing_samples:
        f.write(sample + '\n\n')

print(f"✓ 写作数据: {len(writing_samples)} 条")

# 数学数据
math_samples = [
    "求证: 对于任意正整数 n，有 1 + 2 + ... + n = n(n+1)/2",
    "微积分基本定理: ∫[a,b] f(x)dx = F(b) - F(a)，其中 F 是 f 的原函数",
    "拉格朗日中值定理: f'(c) = (f(b)-f(a))/(b-a)",
] * 200

with open(output_dir / "math_sample.txt", 'w', encoding='utf-8') as f:
    for sample in math_samples:
        f.write(sample + '\n\n')

print(f"✓ 数学数据: {len(math_samples)} 条")

# 知识数据
knowledge_samples = [
    "量子力学是描述微观物质世界的物理学理论，与相对论一起构成现代物理学的理论基础。",
    "DNA（脱氧核糖核酸）是生物体内储存遗传信息的生物大分子。",
    "光合作用是植物、藻类利用光能将二氧化碳和水转化为有机物并释放氧气的过程。",
] * 200

with open(output_dir / "knowledge_sample.txt", 'w', encoding='utf-8') as f:
    for sample in knowledge_samples:
        f.write(sample + '\n\n')

print(f"✓ 知识数据: {len(knowledge_samples)} 条")
print(f"\n总计: {len(code_samples) + len(writing_samples) + len(math_samples) + len(knowledge_samples)} 条")
