import ast
from collections import defaultdict

FILE = "./Neurogram.py"

with open(FILE, "r", encoding="utf-8") as f:
    source = f.read()

tree = ast.parse(source)

class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.classes = {}

    def visit_ClassDef(self, node):
        methods = []
        attributes = set()

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = {
                    "name": item.name,
                    "lines": item.end_lineno - item.lineno,
                    "calls": set(),
                    "attributes_used": set(),
                }

                for sub in ast.walk(item):

                    # self.foo
                    if (
                        isinstance(sub, ast.Attribute)
                        and isinstance(sub.value, ast.Name)
                        and sub.value.id == "self"
                    ):
                        method_info["attributes_used"].add(sub.attr)

                    # self.bar()
                    if (
                        isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id == "self"
                    ):
                        method_info["calls"].add(sub.func.attr)

                attributes.update(method_info["attributes_used"])
                methods.append(method_info)

        self.classes[node.name] = {
            "methods": methods,
            "attributes": sorted(attributes),
        }

analyzer = Analyzer()
analyzer.visit(tree)

for cls, info in analyzer.classes.items():
    print("=" * 80)
    print(f"CLASS: {cls}")
    print()

    print("ATTRIBUTES:")
    for attr in info["attributes"]:
        print(f"  - {attr}")

    print("\nMETHODS:")
    for m in sorted(info["methods"], key=lambda x: -x["lines"]):
        print(f"\n{m['name']} ({m['lines']} lines)")
        print(f"  calls: {sorted(m['calls'])}")
        print(f"  attrs: {sorted(m['attributes_used'])}")