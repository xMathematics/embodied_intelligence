# 5.5 工具使用

## 目录

- [1. 引言](#1-引言)
- [2. 工具使用概述](#2-工具使用概述)
- [3. 工具类型](#3-工具类型)
- [4. 工具调用框架](#4-工具调用框架)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 工具使用的重要性

**工具使用**是指大语言模型调用外部工具来完成任务的能力。这可以显著扩展模型的能力，使其能够访问实时数据、进行计算、调用API等。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **信息检索** | 获取实时信息 | 搜索最新新闻 |
| **数学计算** | 进行精确计算 | 复杂数学运算 |
| **API调用** | 调用外部API | 天气查询、股票行情 |
| **数据库操作** | 查询数据库 | 数据查询和分析 |

---

## 2. 工具使用概述

### 2.1 定义

**工具使用**：大语言模型根据任务需求，选择并调用合适的外部工具来完成任务。

**形式化表达**：
```
ToolUse(model, tools, task) → result
```

### 2.2 工具使用的特点

| 特点 | 描述 |
|------|------|
| **扩展性** | 扩展模型能力 |
| **准确性** | 提供精确结果 |
| **实时性** | 访问实时数据 |
| **复杂性** | 处理复杂任务 |

---

## 3. 工具类型

### 3.1 计算器

**用途**：进行数学计算

**示例**：
```python
def calculator(expression):
    """
    计算器工具
    
    参数:
        expression: 数学表达式
    
    返回:
        计算结果
    """
    try:
        return eval(expression)
    except Exception as e:
        return f"错误: {e}"

# 使用
print(calculator("2 + 3 * 4"))  # 14
```

### 3.2 搜索引擎

**用途**：搜索信息

**示例**：
```python
def search(query):
    """
    搜索工具
    
    参数:
        query: 搜索查询
    
    返回:
        搜索结果
    """
    # 模拟搜索结果
    results = [
        {"title": "搜索结果1", "summary": "这是第一个搜索结果"},
        {"title": "搜索结果2", "summary": "这是第二个搜索结果"}
    ]
    return results

# 使用
print(search("人工智能最新进展"))
```

### 3.3 API调用

**用途**：调用外部API

**示例**：
```python
import requests

def get_weather(city):
    """
    获取天气信息
    
    参数:
        city: 城市名称
    
    返回:
        天气信息
    """
    # 模拟API调用
    weather_data = {
        "city": city,
        "temperature": 25,
        "condition": "晴朗",
        "humidity": 60
    }
    return weather_data

# 使用
print(get_weather("北京"))
```

---

## 4. 工具调用框架

### 4.1 工具选择机制

**方法**：
1. 分析任务需求
2. 选择合适的工具
3. 生成工具调用参数
4. 执行工具调用
5. 处理工具返回结果

```python
class ToolSelector:
    def __init__(self, tools):
        """
        初始化工具选择器
        
        参数:
            tools: 可用工具列表
        """
        self.tools = tools
    
    def select_tool(self, task):
        """
        根据任务选择工具
        
        参数:
            task: 任务描述
        
        返回:
            选择的工具
        """
        # 简单的规则匹配
        if "计算" in task or "数学" in task or "等于" in task:
            return self.tools.get('calculator')
        elif "搜索" in task or "查找" in task:
            return self.tools.get('search')
        elif "天气" in task:
            return self.tools.get('weather')
        
        return None
```

### 4.2 工具调用流程

```python
class ToolCaller:
    def __init__(self, tools):
        self.tools = tools
    
    def call_tool(self, tool_name, **kwargs):
        """
        调用工具
        
        参数:
            tool_name: 工具名称
            kwargs: 工具参数
        
        返回:
            工具执行结果
        """
        if tool_name not in self.tools:
            return f"未知工具: {tool_name}"
        
        tool = self.tools[tool_name]
        try:
            return tool(**kwargs)
        except Exception as e:
            return f"工具调用失败: {e}"

# 注册工具
tools = {
    'calculator': lambda expression: eval(expression),
    'search': lambda query: f"搜索结果: {query}",
    'weather': lambda city: f"{city}天气晴朗"
}

caller = ToolCaller(tools)
print(caller.call_tool('calculator', expression='2+3'))  # 5
print(caller.call_tool('weather', city='北京'))         # 北京天气晴朗
```

### 4.3 结果整合

**方法**：将工具返回结果整合到回答中

```python
def integrate_results(task, tool_results):
    """
    整合工具结果
    
    参数:
        task: 原始任务
        tool_results: 工具结果列表
    
    返回:
        最终回答
    """
    answer = f"关于'{task}'的答案：\n"
    
    for i, result in enumerate(tool_results, 1):
        answer += f"{i}. {result}\n"
    
    return answer

# 使用
results = ["计算结果: 42", "搜索结果: 相关信息"]
print(integrate_results("2+2等于多少", results))
```

---

## 5. 实践练习

### 练习1：实现工具调用系统

```python
class ToolSystem:
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, name, func, description):
        """
        注册工具
        
        参数:
            name: 工具名称
            func: 工具函数
            description: 工具描述
        """
        self.tools[name] = {
            'function': func,
            'description': description
        }
    
    def list_tools(self):
        """列出所有可用工具"""
        return {name: info['description'] for name, info in self.tools.items()}
    
    def call_tool(self, name, **kwargs):
        """
        调用工具
        
        参数:
            name: 工具名称
            kwargs: 工具参数
        
        返回:
            结果
        """
        if name not in self.tools:
            return f"未知工具: {name}"
        
        try:
            return self.tools[name]['function'](**kwargs)
        except Exception as e:
            return f"调用失败: {e}"

# 注册工具
system = ToolSystem()
system.register_tool(
    'calculator',
    lambda expr: eval(expr),
    '数学计算器，支持基本算术运算'
)
system.register_tool(
    'weather',
    lambda city: f"{city}的天气是晴朗，温度25°C",
    '查询城市天气'
)
system.register_tool(
    'time',
    lambda: f"当前时间: 2024年1月15日 10:30",
    '获取当前时间'
)

# 测试
print("可用工具:", system.list_tools())
print(system.call_tool('calculator', expr='2*3+5'))  # 11
print(system.call_tool('weather', city='上海'))     # 上海的天气是晴朗，温度25°C
print(system.call_tool('time'))                     # 当前时间: 2024年1月15日 10:30
```

### 练习2：实现工具选择器

```python
class SmartToolSelector:
    def __init__(self, tool_system):
        self.tool_system = tool_system
    
    def analyze_task(self, task):
        """
        分析任务并选择工具
        
        参数:
            task: 任务描述
        
        返回:
            工具名称和参数
        """
        tools = self.tool_system.list_tools()
        
        # 基于关键词匹配
        if any(keyword in task for keyword in ['计算', '等于', '多少', '数学', '求和']):
            # 提取数学表达式
            import re
            match = re.search(r'[\d+\-*/^(). ]+', task)
            if match:
                return ('calculator', {'expr': match.group().strip()})
        
        elif any(keyword in task for keyword in ['天气', '温度', '下雨']):
            # 提取城市名称
            cities = ['北京', '上海', '广州', '深圳', '杭州']
            for city in cities:
                if city in task:
                    return ('weather', {'city': city})
        
        elif any(keyword in task for keyword in ['时间', '几点', '现在']):
            return ('time', {})
        
        return (None, None)

# 测试
selector = SmartToolSelector(system)
print(selector.analyze_task("2+2等于多少"))          # ('calculator', {'expr': '2+2'})
print(selector.analyze_task("北京今天天气怎么样"))    # ('weather', {'city': '北京'})
print(selector.analyze_task("现在几点了"))            # ('time', {})
```

### 练习3：实现完整的工具调用流程

```python
class Agent:
    def __init__(self, tool_system):
        self.tool_system = tool_system
        self.selector = SmartToolSelector(tool_system)
    
    def solve_task(self, task):
        """
        解决任务
        
        参数:
            task: 任务描述
        
        返回:
            结果
        """
        # 步骤1：分析任务
        tool_name, params = self.selector.analyze_task(task)
        
        if tool_name is None:
            return f"无法识别的任务: {task}"
        
        # 步骤2：调用工具
        result = self.tool_system.call_tool(tool_name, **params)
        
        # 步骤3：整合结果
        return f"任务: {task}\n结果: {result}"

# 测试
agent = Agent(system)
print(agent.solve_task("计算100乘以50"))              # 任务: 计算100乘以50\n结果: 5000
print(agent.solve_task("上海的天气怎么样"))            # 任务: 上海的天气怎么样\n结果: 上海的天气是晴朗，温度25°C
print(agent.solve_task("现在几点了"))                  # 任务: 现在几点了\n结果: 当前时间: 2024年1月15日 10:30
```

---

**返回**：[推理能力](01-reasoning-capabilities.md)

---

## 参考文献

1. ReAct: Synergizing Reasoning and Acting in Language Models. (2022).
2. Toolformer: Language Models Can Teach Themselves to Use Tools. (2023).
3. OpenAI Function Calling. (2023).
