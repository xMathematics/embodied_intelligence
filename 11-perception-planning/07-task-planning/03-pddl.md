# 7.3 规划域定义语言 (PDDL)

## 目录

- [1. 引言](#1-引言)
- [2. PDDL语法](#2-pddl语法)
- [3. PDDL解析器](#3-pddl解析器)
- [4. 示例](#4-示例)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 PDDL简介

PDDL (Planning Domain Definition Language) 是AI规划的标准语言。

```python
import re
```

---

## 2. PDDL语法

### 2.1 域文件结构

```python
class PDDLDomain:
    """PDDL域"""
    
    def __init__(self, name):
        self.name = name
        self.predicates = []
        self.actions = []
    
    def __repr__(self):
        return f"Domain({self.name}, {len(self.predicates)} predicates, {len(self.actions)} actions)"


class PDDLProblem:
    """PDDL问题"""
    
    def __init__(self, name, domain_name):
        self.name = name
        self.domain_name = domain_name
        self.objects = []
        self.init = []
        self.goal = []
    
    def __repr__(self):
        return f"Problem({self.name}, domain={self.domain_name})"


class PDDLAction:
    """PDDL动作"""
    
    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters
        self.precondition = None
        self.effect = None
    
    def __repr__(self):
        return f"Action({self.name}, {len(self.parameters)} params)"
```

---

## 3. PDDL解析器

### 3.1 简单解析器

```python
class SimplePDDLParser:
    """简单的PDDL解析器"""
    
    def __init__(self):
        pass
    
    def tokenize(self, text):
        """分词"""
        # 简化
        tokens = re.findall(r'\(|\)|[^\s()]+', text)
        return tokens
    
    def parse_domain(self, text):
        """解析域"""
        tokens = self.tokenize(text)
        domain = PDDLDomain("parsed")
        return domain
    
    def parse_problem(self, text):
        """解析问题"""
        tokens = self.tokenize(text)
        problem = PDDLProblem("parsed", "domain")
        return problem
```

---

## 4. 示例

### 4.1 积木世界PDDL

```python
def blocks_world_pddl():
    """积木世界PDDL示例"""
    
    domain_text = """
(define (domain blocks)
  (:predicates (on ?x ?y)
               (ontable ?x)
               (clear ?x)
               (handempty)
               (holding ?x))
  
  (:action pick-up
    :parameters (?x)
    :precondition (and (clear ?x) (ontable ?x) (handempty))
    :effect (and (not (ontable ?x))
                 (not (clear ?x))
                 (not (handempty))
                 (holding ?x)))
  
  (:action put-down
    :parameters (?x)
    :precondition (holding ?x)
    :effect (and (not (holding ?x))
                 (clear ?x)
                 (handempty)
                 (ontable ?x)))
  
  (:action stack
    :parameters (?x ?y)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (not (holding ?x))
                 (not (clear ?y))
                 (handempty)
                 (on ?x ?y)
                 (clear ?x)))
  
  (:action unstack
    :parameters (?x ?y)
    :precondition (and (on ?x ?y) (clear ?x) (handempty))
    :effect (and (holding ?x)
                 (not (on ?x ?y))
                 (not (clear ?x))
                 (not (handempty))
                 (clear ?y)))
)
    """
    
    problem_text = """
(define (problem blocks-problem)
  (:domain blocks)
  (:objects a b c - block)
  (:init (ontable a) (ontable b) (on c a) 
         (clear b) (clear c) (handempty))
  (:goal (and (on a b) (on b c) (ontable c)))
)
    """
    
    return domain_text, problem_text
```

### 4.2 物流PDDL

```python
def logistics_pddl():
    """物流PDDL示例"""
    
    domain_text = """
(define (domain logistics)
  (:predicates (at ?obj ?loc)
               (in ?pkg ?truck)
               (connected ?from ?to))
  
  (:action drive
    :parameters (?truck ?from ?to)
    :precondition (and (at ?truck ?from) (connected ?from ?to))
    :effect (and (not (at ?truck ?from))
                 (at ?truck ?to)))
  
  (:action load
    :parameters (?pkg ?truck ?loc)
    :precondition (and (at ?pkg ?loc) (at ?truck ?loc))
    :effect (and (not (at ?pkg ?loc))
                 (in ?pkg ?truck)))
  
  (:action unload
    :parameters (?pkg ?truck ?loc)
    :precondition (and (in ?pkg ?truck) (at ?truck ?loc))
    :effect (and (not (in ?pkg ?truck))
                 (at ?pkg ?loc)))
)
    """
    
    return domain_text, ""
```

---

## 5. 实践练习

### 练习1：PDDL示例展示

```python
def exercise_pddl():
    """PDDL练习"""
    print("=== 规划域定义语言 (PDDL) ===\n")
    
    domain, problem = blocks_world_pddl()
    
    print("=== 域文件 (Blocks World) ===")
    print(domain)
    print("\n=== 问题文件 ===")
    print(problem)
    
    print("\n\n=== 物流示例 ===")
    logistics_domain, _ = logistics_pddl()
    print(logistics_domain)
    
    print("\n\n常用PDDL规划器:")
    print("  - Fast Downward")
    print("  - LAMA")
    print("  - FF")
    print("  - Metric-FF")

# exercise_pddl()
```

---

**下一节**：[学习型规划](04-learning-planning.md)

---

## 参考文献

1. McDermott, D., et al. (1998). PDDL - The Planning Domain Definition Language.
2. Fox, M., & Long, D. (2003). PDDL2.1: An Extension to PDDL for Expressing Temporal Planning Domains.
3. Helmert, M. (2006). The Fast Downward Planning System.
4. Gerevini, A. E., et al. (2003). Planning in PDDL with LPG.
5. Hoffmann, J., & Nebel, B. (2001). The FF Planning System: Fast Plan Generation Through Heuristic Search.
