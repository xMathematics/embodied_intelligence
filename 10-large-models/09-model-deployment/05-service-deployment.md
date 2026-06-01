# 9.5 服务部署

## 目录

- [1. 引言](#1-引言)
- [2. 服务部署概述](#2-服务部署概述)
- [3. 部署架构](#3-部署架构)
- [4. 部署工具](#4-部署工具)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 服务部署的重要性

**服务部署**是将训练好的模型部署为可对外提供服务的过程。这是将AI模型应用到实际生产环境的关键步骤。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **API服务** | 提供RESTful API接口 | 云端AI服务 |
| **边缘部署** | 在边缘设备上运行 | 手机、嵌入式设备 |
| **私有化部署** | 在企业内部部署 | 企业内部AI系统 |
| **容器化部署** | 使用容器技术部署 | Docker、Kubernetes |

---

## 2. 服务部署概述

### 2.1 定义

**服务部署**：将AI模型打包并部署为可对外提供服务的过程。

**形式化表达**：
```
Deploy(Model, Infrastructure) → Service
```

### 2.2 部署目标

| 目标 | 描述 | 衡量指标 |
|------|------|---------|
| **可用性** | 服务持续可用 | 可用性百分比 |
| **可扩展性** | 支持扩展 | 水平扩展能力 |
| **安全性** | 保护数据和模型 | 安全审计结果 |
| **可监控性** | 可监控服务状态 | 监控覆盖率 |

---

## 3. 部署架构

### 3.1 单体架构

```python
class MonolithService:
    def __init__(self, model):
        self.model = model
        self.app = None
    
    def build(self):
        """构建单体服务"""
        from flask import Flask, request, jsonify
        
        app = Flask(__name__)
        
        @app.route('/infer', methods=['POST'])
        def infer():
            data = request.get_json()
            inputs = torch.tensor(data['inputs'])
            
            with torch.no_grad():
                outputs = self.model(inputs)
            
            return jsonify({'outputs': outputs.tolist()})
        
        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'healthy'})
        
        self.app = app
    
    def run(self, host='0.0.0.0', port=8080):
        """运行服务"""
        if self.app is None:
            self.build()
        
        self.app.run(host=host, port=port)

# 测试
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

service = MonolithService(model)
# service.run()  # 启动服务
```

### 3.2 微服务架构

```python
class MicroserviceManager:
    def __init__(self):
        self.services = {}
    
    def add_service(self, name, service):
        """
        添加微服务
        
        参数:
            name: 服务名称
            service: 服务实例
        """
        self.services[name] = service
    
    def start_all(self):
        """启动所有服务"""
        for name, service in self.services.items():
            print(f"启动服务: {name}")
            service.start()
    
    def stop_all(self):
        """停止所有服务"""
        for name, service in self.services.items():
            print(f"停止服务: {name}")
            service.stop()

# 示例微服务
class InferenceService:
    def __init__(self, model, port=8080):
        self.model = model
        self.port = port
    
    def start(self):
        print(f"推理服务启动在端口 {self.port}")
    
    def stop(self):
        print(f"推理服务停止")

class MonitoringService:
    def __init__(self, port=9090):
        self.port = port
    
    def start(self):
        print(f"监控服务启动在端口 {self.port}")
    
    def stop(self):
        print(f"监控服务停止")

# 测试
manager = MicroserviceManager()
manager.add_service('inference', InferenceService(model))
manager.add_service('monitoring', MonitoringService())

manager.start_all()
# manager.stop_all()
```

### 3.3 边缘架构

```python
class EdgeDeployment:
    def __init__(self, model, devices):
        self.model = model
        self.devices = devices
        self.deployed_models = {}
    
    def deploy(self):
        """部署到边缘设备"""
        for device in self.devices:
            # 优化模型
            optimized_model = self._optimize_for_edge(self.model)
            
            # 部署到设备
            self._deploy_to_device(optimized_model, device)
            
            self.deployed_models[device] = optimized_model
        
        print(f"已部署到 {len(self.devices)} 个设备")
    
    def _optimize_for_edge(self, model):
        """优化模型用于边缘设备"""
        # 量化
        quantized_model = self._quantize_model(model)
        
        # 剪枝
        pruned_model = self._prune_model(quantized_model)
        
        return pruned_model
    
    def _quantize_model(self, model):
        """量化模型"""
        for param in model.parameters():
            param.data = param.data.to(torch.float16)
        return model
    
    def _prune_model(self, model, rate=0.3):
        """剪枝模型"""
        for name, param in model.named_parameters():
            if 'weight' in name:
                mask = torch.rand_like(param.data) > rate
                param.data *= mask.float()
        return model
    
    def _deploy_to_device(self, model, device):
        """部署到设备"""
        print(f"部署到设备: {device}")

# 测试
devices = ['edge-device-1', 'edge-device-2', 'edge-device-3']
deployment = EdgeDeployment(model, devices)
deployment.deploy()

print(f"已部署设备: {list(deployment.deployed_models.keys())}")
```

---

## 4. 部署工具

### 4.1 Docker部署

```python
class DockerDeployer:
    def __init__(self, model):
        self.model = model
    
    def create_dockerfile(self, output_path='./'):
        """
        创建Dockerfile
        
        参数:
            output_path: 输出路径
        """
        dockerfile = f"""
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.pth .
COPY app.py .

EXPOSE 8080

CMD ["python", "app.py"]
        """
        
        with open(output_path + 'Dockerfile', 'w') as f:
            f.write(dockerfile.strip())
        
        print("Dockerfile创建完成")
    
    def create_requirements(self, output_path='./'):
        """创建requirements.txt"""
        requirements = """
torch>=2.0.0
flask>=2.0.0
        """
        
        with open(output_path + 'requirements.txt', 'w') as f:
            f.write(requirements.strip())
        
        print("requirements.txt创建完成")
    
    def create_app(self, output_path='./'):
        """创建应用代码"""
        app_code = f"""
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# 加载模型
model = torch.load('model.pth')
model.eval()

@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json()
    inputs = torch.tensor(data['inputs'])
    
    with torch.no_grad():
        outputs = model(inputs)
    
    return jsonify({{'outputs': outputs.tolist()}})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({{'status': 'healthy'}})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
        """
        
        with open(output_path + 'app.py', 'w') as f:
            f.write(app_code.strip())
        
        print("app.py创建完成")
    
    def save_model(self, output_path='./'):
        """保存模型"""
        torch.save(self.model, output_path + 'model.pth')
        print("模型保存完成")
    
    def build_and_run(self, image_name='inference-service'):
        """构建并运行Docker容器"""
        import subprocess
        
        # 构建镜像
        subprocess.run(['docker', 'build', '-t', image_name, '.'])
        
        # 运行容器
        subprocess.run([
            'docker', 'run', '-p', '8080:8080', '-d', image_name
        ])
        
        print("Docker容器启动完成")

# 测试
deployer = DockerDeployer(model)
deployer.create_dockerfile()
deployer.create_requirements()
deployer.create_app()
deployer.save_model()
# deployer.build_and_run()  # 需要Docker环境
```

### 4.2 Kubernetes部署

```python
class KubernetesDeployer:
    def __init__(self, model_name, image_name):
        self.model_name = model_name
        self.image_name = image_name
    
    def create_deployment(self, replicas=3):
        """
        创建Deployment配置
        
        参数:
            replicas: 副本数量
        
        返回:
            Deployment YAML
        """
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': self.model_name
            },
            'spec': {
                'replicas': replicas,
                'selector': {
                    'matchLabels': {
                        'app': self.model_name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': self.model_name
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': self.model_name,
                            'image': self.image_name,
                            'ports': [{
                                'containerPort': 8080
                            }],
                            'resources': {
                                'requests': {
                                    'cpu': '100m',
                                    'memory': '512Mi'
                                },
                                'limits': {
                                    'cpu': '500m',
                                    'memory': '1Gi'
                                }
                            }
                        }]
                    }
                }
            }
        }
        
        return deployment
    
    def create_service(self):
        """
        创建Service配置
        
        返回:
            Service YAML
        """
        service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f"{self.model_name}-service"
            },
            'spec': {
                'selector': {
                    'app': self.model_name
                },
                'ports': [{
                    'port': 80,
                    'targetPort': 8080
                }],
                'type': 'LoadBalancer'
            }
        }
        
        return service
    
    def create_configmap(self, config):
        """
        创建ConfigMap
        
        参数:
            config: 配置字典
        
        返回:
            ConfigMap YAML
        """
        configmap = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': f"{self.model_name}-config"
            },
            'data': config
        }
        
        return configmap
    
    def deploy(self):
        """部署到Kubernetes"""
        import yaml
        
        # 创建配置文件
        deployment = self.create_deployment()
        service = self.create_service()
        
        # 保存配置文件
        with open('deployment.yaml', 'w') as f:
            yaml.dump(deployment, f)
        
        with open('service.yaml', 'w') as f:
            yaml.dump(service, f)
        
        # 应用配置
        import subprocess
        subprocess.run(['kubectl', 'apply', '-f', 'deployment.yaml'])
        subprocess.run(['kubectl', 'apply', '-f', 'service.yaml'])
        
        print("Kubernetes部署完成")

# 测试
deployer = KubernetesDeployer('inference-model', 'my-inference-image:latest')
deployment = deployer.create_deployment(replicas=3)
service = deployer.create_service()

print("Deployment配置:")
print(deployment)
print("\nService配置:")
print(service)
```

### 4.3 Serverless部署

```python
class ServerlessDeployer:
    def __init__(self, model):
        self.model = model
    
    def create_lambda_function(self):
        """
        创建Lambda函数
        
        返回:
            Lambda配置
        """
        lambda_config = {
            'function_name': 'inference-function',
            'runtime': 'python3.9',
            'handler': 'lambda_function.lambda_handler',
            'memory_size': 512,
            'timeout': 30,
            'environment': {
                'Variables': {
                    'MODEL_S3_PATH': 's3://my-bucket/model.pth'
                }
            }
        }
        
        return lambda_config
    
    def create_lambda_code(self):
        """创建Lambda代码"""
        code = """
import torch
import boto3
import os

# 加载模型（延迟加载）
model = None

def load_model():
    global model
    if model is None:
        s3 = boto3.client('s3')
        s3.download_file(
            os.environ['MODEL_S3_PATH'].split('/')[2],
            '/'.join(os.environ['MODEL_S3_PATH'].split('/')[3:]),
            '/tmp/model.pth'
        )
        model = torch.load('/tmp/model.pth')
        model.eval()

def lambda_handler(event, context):
    load_model()
    
    # 处理请求
    inputs = torch.tensor(event['inputs'])
    
    with torch.no_grad():
        outputs = model(inputs)
    
    return {
        'statusCode': 200,
        'body': {'outputs': outputs.tolist()}
    }
        """
        
        return code
    
    def deploy(self):
        """部署到AWS Lambda"""
        import boto3
        
        client = boto3.client('lambda')
        
        # 创建Lambda函数
        response = client.create_function(
            FunctionName='inference-function',
            Runtime='python3.9',
            Role='arn:aws:iam::123456789012:role/lambda-role',
            Handler='lambda_function.lambda_handler',
            Code={
                'ZipFile': self.create_lambda_code()
            },
            Environment={
                'Variables': {
                    'MODEL_S3_PATH': 's3://my-bucket/model.pth'
                }
            }
        )
        
        print(f"Lambda函数创建完成: {response['FunctionArn']}")

# 测试
deployer = ServerlessDeployer(model)
lambda_config = deployer.create_lambda_function()
print("Lambda配置:")
print(lambda_config)
```

---

## 5. 实践练习

### 练习1：实现推理API服务

```python
class InferenceAPIService:
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def create_app(self):
        """创建Flask应用"""
        from flask import Flask, request, jsonify
        import time
        
        app = Flask(__name__)
        
        @app.route('/api/v1/infer', methods=['POST'])
        def infer():
            try:
                data = request.get_json()
                
                # 验证输入
                if 'inputs' not in data:
                    return jsonify({'error': '缺少inputs字段'}), 400
                
                inputs = torch.tensor(data['inputs'])
                
                # 推理
                start_time = time.time()
                with torch.no_grad():
                    outputs = self.model(inputs)
                inference_time = (time.time() - start_time) * 1000
                
                return jsonify({
                    'outputs': outputs.tolist(),
                    'inference_time_ms': inference_time
                })
            
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/v1/health', methods=['GET'])
        def health():
            return jsonify({
                'status': 'healthy',
                'model_type': type(self.model).__name__
            })
        
        @app.route('/api/v1/metrics', methods=['GET'])
        def metrics():
            return jsonify({
                'model_parameters': sum(p.numel() for p in self.model.parameters()),
                'device': next(self.model.parameters()).device.type
            })
        
        return app
    
    def run(self, host='0.0.0.0', port=8080):
        """运行服务"""
        app = self.create_app()
        app.run(host=host, port=port, threaded=True)

# 测试
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

service = InferenceAPIService(model)
# service.run()  # 启动服务

# 测试API（使用requests）
import requests
import threading

def test_api():
    # 等待服务启动
    import time
    time.sleep(2)
    
    # 测试健康检查
    response = requests.get('http://localhost:8080/api/v1/health')
    print(f"健康检查: {response.status_code}")
    print(f"响应: {response.json()}")
    
    # 测试推理
    data = {'inputs': torch.randn(2, 100).tolist()}
    response = requests.post('http://localhost:8080/api/v1/infer', json=data)
    print(f"\n推理测试: {response.status_code}")
    print(f"响应: {response.json()}")

# 在单独线程中运行测试
# threading.Thread(target=test_api).start()
```

### 练习2：实现模型服务管理器

```python
class ModelServiceManager:
    def __init__(self):
        self.services = {}
        self.running_services = []
    
    def register_service(self, name, model, port):
        """
        注册服务
        
        参数:
            name: 服务名称
            model: 模型
            port: 端口号
        """
        self.services[name] = {
            'model': model,
            'port': port,
            'service': None
        }
    
    def start_service(self, name):
        """
        启动服务
        
        参数:
            name: 服务名称
        """
        if name not in self.services:
            print(f"服务 {name} 未注册")
            return
        
        service_info = self.services[name]
        service = InferenceAPIService(service_info['model'])
        
        # 在单独线程中启动
        import threading
        thread = threading.Thread(
            target=service.run,
            kwargs={'host': '0.0.0.0', 'port': service_info['port']},
            daemon=True
        )
        thread.start()
        
        self.running_services.append({
            'name': name,
            'port': service_info['port'],
            'thread': thread
        })
        
        print(f"服务 {name} 已启动在端口 {service_info['port']}")
    
    def stop_service(self, name):
        """
        停止服务
        
        参数:
            name: 服务名称
        """
        for i, service in enumerate(self.running_services):
            if service['name'] == name:
                # 线程是daemon，主程序退出时会自动停止
                self.running_services.pop(i)
                print(f"服务 {name} 已停止")
                return
        
        print(f"服务 {name} 未运行")
    
    def list_services(self):
        """列出所有服务"""
        print("\n服务列表:")
        for name, info in self.services.items():
            is_running = any(s['name'] == name for s in self.running_services)
            status = '运行中' if is_running else '已停止'
            print(f"- {name}: 端口 {info['port']}, 状态: {status}")

# 测试
manager = ModelServiceManager()

# 注册服务
manager.register_service('text-classification', model, 8080)
manager.register_service('sentiment-analysis', model, 8081)

# 列出服务
manager.list_services()

# 启动服务
manager.start_service('text-classification')
manager.start_service('sentiment-analysis')

# 列出服务
manager.list_services()

# 停止服务
manager.stop_service('sentiment-analysis')

# 列出服务
manager.list_services()
```

### 练习3：实现部署流水线

```python
class DeploymentPipeline:
    def __init__(self):
        self.stages = []
    
    def add_stage(self, name, func):
        """
        添加部署阶段
        
        参数:
            name: 阶段名称
            func: 阶段函数
        """
        self.stages.append({'name': name, 'func': func})
    
    def run(self, model, config):
        """
        运行部署流水线
        
        参数:
            model: 模型
            config: 配置
        
        返回:
            部署结果
        """
        result = {'success': True, 'steps': []}
        
        for stage in self.stages:
            print(f"执行阶段: {stage['name']}")
            
            try:
                output = stage['func'](model, config)
                result['steps'].append({
                    'name': stage['name'],
                    'status': 'success',
                    'output': output
                })
                print(f"  ✓ 成功")
            
            except Exception as e:
                result['success'] = False
                result['steps'].append({
                    'name': stage['name'],
                    'status': 'failed',
                    'error': str(e)
                })
                print(f"  ✗ 失败: {e}")
                break
        
        return result

# 定义部署阶段
def stage_validate(model, config):
    """验证模型"""
    assert model is not None, "模型不能为空"
    return "模型验证通过"

def stage_optimize(model, config):
    """优化模型"""
    # 量化
    for param in model.parameters():
        param.data = param.data.to(torch.float16)
    return "模型优化完成"

def stage_package(model, config):
    """打包模型"""
    torch.save(model, config.get('output_path', 'model.pth'))
    return "模型打包完成"

def stage_deploy(model, config):
    """部署模型"""
    deployer = DockerDeployer(model)
    deployer.create_dockerfile()
    deployer.create_requirements()
    deployer.create_app()
    return "部署配置创建完成"

# 测试
pipeline = DeploymentPipeline()

# 添加阶段
pipeline.add_stage('验证模型', stage_validate)
pipeline.add_stage('优化模型', stage_optimize)
pipeline.add_stage('打包模型', stage_package)
pipeline.add_stage('部署配置', stage_deploy)

# 运行流水线
config = {'output_path': 'models/model.pth'}
result = pipeline.run(model, config)

print("\n部署结果:")
print(f"整体状态: {'成功' if result['success'] else '失败'}")
for step in result['steps']:
    print(f"- {step['name']}: {step['status']}")
```

---

**返回**：[模型压缩](01-model-compression.md)

---

## 参考文献

1. Burns, B., et al. (2016). Kubernetes: Up and Running.
2. Docker Inc. (2023). Docker Documentation.
3. AWS (2023). AWS Lambda Documentation.