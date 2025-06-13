# NeuralTrain Forge 🧠

Uma aplicação web completa e profissional para fine-tuning de modelos de linguagem (LLMs) com suporte a LoRA, QLoRA, PEFT e integração com HuggingFace Hub.

![NeuralTrain Forge](https://img.shields.io/badge/NeuralTrain%20Forge-v1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45+-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Visão Geral

O NeuralTrain Forge é uma plataforma web moderna e intuitiva desenvolvida especificamente para facilitar o fine-tuning de modelos de linguagem. A aplicação oferece uma interface gráfica completa que permite aos desenvolvedores e pesquisadores de IA treinar modelos de forma eficiente, segura e escalável.

### ✨ Características Principais

- **Interface Web Moderna**: Interface desenvolvida com Streamlit, responsiva e intuitiva
- **Suporte Completo a LoRA/QLoRA**: Implementação otimizada para fine-tuning eficiente
- **Integração HuggingFace**: Download e upload direto de modelos do HuggingFace Hub
- **Múltiplos Formatos**: Suporte a .gguf, .bin, .safetensors e outros formatos
- **Monitoramento em Tempo Real**: Acompanhamento de métricas e progresso durante o treinamento
- **Visualizações Avançadas**: Gráficos interativos com Plotly para análise de resultados
- **Arquitetura Modular**: Código organizado e extensível para futuras melhorias
- **Compatibilidade Cloud**: Otimizado para Paperspace, RunPod e outras plataformas

## 🚀 Instalação Rápida

### Pré-requisitos

- Python 3.10 ou superior
- CUDA 11.8+ (opcional, para aceleração GPU)
- 8GB+ RAM (16GB+ recomendado)
- 50GB+ espaço em disco

### Instalação

1. **Clone o repositório:**
```bash
git clone https://github.com/seu-usuario/neuraltrain-forge.git
cd neuraltrain-forge
```

2. **Crie e ative o ambiente virtual:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

4. **Execute a aplicação:**
```bash
streamlit run main.py
```

5. **Acesse no navegador:**
```
http://localhost:8501
```

## 📋 Funcionalidades Detalhadas

### 🏠 Dashboard Principal

O dashboard oferece uma visão geral completa do sistema:

- **Métricas em Tempo Real**: Modelos carregados, datasets disponíveis, jobs ativos
- **Estatísticas de Treinamento**: Progresso, tempo estimado, precisão
- **Gráficos Interativos**: Visualização de tendências e performance
- **Status do Sistema**: Monitoramento de recursos (GPU, RAM, armazenamento)

### 📤 Upload de Modelos

Sistema completo para gerenciamento de modelos:

#### Upload Local
- Suporte a múltiplos formatos: `.gguf`, `.bin`, `.safetensors`, `.pt`
- Validação automática de integridade
- Detecção de arquitetura e parâmetros
- Limite de 200MB por arquivo (configurável)

#### HuggingFace Hub
- Busca e download direto de modelos
- Autenticação com token HF
- Cache inteligente para otimização
- Suporte a modelos privados

#### Modelos Carregados
- Lista organizada de modelos disponíveis
- Informações detalhadas (tamanho, tipo, data)
- Funcionalidades de edição e remoção
- Sistema de tags e categorização

### 📊 Upload de Datasets

Gerenciamento avançado de datasets para treinamento:

#### Formatos Suportados
- **Texto Simples** (`.txt`): Para treinamento de linguagem
- **JSONL** (`.jsonl`): Formato estruturado para conversações
- **CSV** (`.csv`): Dados tabulares com colunas personalizáveis
- **Parquet** (`.parquet`): Formato otimizado para grandes volumes

#### Processamento Automático
- Validação de formato e estrutura
- Estatísticas automáticas (tokens, linhas, tamanho)
- Pré-visualização dos dados
- Divisão automática treino/validação

#### Configurações Avançadas
- Tokenização personalizada
- Filtros de qualidade
- Balanceamento de classes
- Augmentação de dados

### ⚙️ Configuração de Treinamento

Interface completa para configuração de fine-tuning:

#### Configurações Básicas
- **Nome do Job**: Identificação única do treinamento
- **Modelo Base**: Seleção do modelo para fine-tuning
- **Dataset**: Escolha do dataset de treinamento
- **Tipo de Treinamento**: LoRA, QLoRA ou Full Fine-tuning

#### Configurações LoRA/QLoRA
- **Rank (r)**: Controle da complexidade dos adaptadores (1-256)
- **Alpha**: Fator de escala para estabilidade (1-512)
- **Dropout**: Regularização para evitar overfitting (0.0-0.5)
- **Target Modules**: Seleção de camadas para adaptação
- **Bias**: Configuração de tratamento de bias

#### Parâmetros de Treinamento
- **Épocas**: Número de passadas pelos dados (1-100)
- **Learning Rate**: Taxa de aprendizado (1e-6 a 1e-2)
- **Batch Size**: Tamanho do lote (1, 2, 4, 8, 16, 32)
- **Gradient Accumulation**: Acumulação de gradientes (1-32)
- **Max Length**: Comprimento máximo de sequência (128-4096)
- **Warmup Ratio**: Proporção de aquecimento (0.0-0.2)

#### Configurações Avançadas
- **Otimizador**: AdamW, SGD, AdamW HF
- **Scheduler**: Linear, Cosine, Polynomial
- **Weight Decay**: Regularização L2
- **FP16/BF16**: Precisão mista para otimização
- **Gradient Checkpointing**: Economia de memória
- **DataLoader Workers**: Paralelização de carregamento

#### Configurações de Salvamento
- **Estratégia**: Por passos, por época ou desabilitado
- **Intervalo**: Frequência de salvamento
- **Limite de Checkpoints**: Número máximo mantido
- **Melhor Modelo**: Carregamento automático do melhor resultado

#### Configurações de Logging
- **Frequência**: Intervalo de logging de métricas
- **Avaliação**: Estratégia de validação
- **Relatórios**: TensorBoard, Weights & Biases
- **Métricas Customizadas**: Definição de métricas específicas

### 📈 Resultados e Análise

Sistema completo de visualização e análise de resultados:

#### Dashboard de Resultados
- **Métricas Resumidas**: Treinamentos concluídos, melhor loss, tempo médio
- **Gráficos de Tendência**: Evolução temporal dos treinamentos
- **Comparação de Modelos**: Análise comparativa de performance
- **Rankings**: Classificação por performance e métricas

#### Análise Detalhada
- **Curvas de Loss**: Visualização de treino e validação
- **Learning Rate Schedule**: Evolução da taxa de aprendizado
- **Métricas de Treinamento**: Loss, perplexidade, gradient norm
- **Métricas de Avaliação**: BLEU, ROUGE, accuracy, F1-score

#### Modelos Treinados
- **Lista Organizada**: Todos os modelos com filtros avançados
- **Informações Detalhadas**: Configurações, métricas, metadados
- **Ações Disponíveis**: Testar, baixar, arquivar, compartilhar
- **Sistema de Versionamento**: Controle de versões dos modelos

#### Downloads e Exportação
- **Modelos**: Download em múltiplos formatos
- **Relatórios**: PDF completo, CSV de métricas, configurações JSON
- **Dados de Treinamento**: Exportação de logs e métricas
- **Checkpoints**: Acesso a pontos intermediários

## 🏗️ Arquitetura Técnica

### Estrutura do Projeto

```
neuraltrain-forge/
├── main.py                 # Ponto de entrada da aplicação
├── requirements.txt        # Dependências Python
├── Dockerfile             # Containerização
├── README.md              # Documentação principal
├── configs/               # Arquivos de configuração
│   ├── config.yaml        # Configuração principal
│   ├── models/            # Configurações de modelos
│   └── training/          # Configurações de treinamento
├── src/                   # Código fonte principal
│   ├── core/              # Módulos principais
│   │   ├── model_manager.py      # Gerenciamento de modelos
│   │   ├── dataset_manager.py    # Gerenciamento de datasets
│   │   └── training_manager.py   # Gerenciamento de treinamento
│   ├── ui/                # Interface do usuário
│   │   ├── components/    # Componentes reutilizáveis
│   │   │   └── sidebar.py # Barra lateral
│   │   └── pages/         # Páginas da aplicação
│   │       ├── home.py           # Dashboard principal
│   │       ├── model_upload.py   # Upload de modelos
│   │       ├── dataset_upload.py # Upload de datasets
│   │       ├── training.py       # Configuração de treino
│   │       └── results.py        # Resultados e análise
│   └── utils/             # Utilitários
│       ├── logging_utils.py      # Sistema de logging
│       └── file_utils.py         # Utilitários de arquivo
├── data/                  # Dados da aplicação
│   ├── models/            # Modelos armazenados
│   ├── datasets/          # Datasets carregados
│   └── outputs/           # Resultados de treinamento
├── logs/                  # Logs do sistema
└── tests/                 # Testes automatizados
```

### Tecnologias Utilizadas

#### Frontend
- **Streamlit 1.45+**: Framework principal para interface web
- **Plotly 6.1+**: Visualizações interativas e gráficos
- **Pandas 2.3+**: Manipulação e análise de dados
- **NumPy 2.3+**: Computação numérica

#### Backend e ML
- **PyTorch 2.7+**: Framework de deep learning
- **Transformers 4.52+**: Biblioteca HuggingFace para LLMs
- **PEFT 0.15+**: Parameter-Efficient Fine-Tuning
- **Accelerate 1.7+**: Otimização de treinamento
- **Datasets 3.6+**: Carregamento e processamento de dados

#### Infraestrutura
- **Python 3.11**: Linguagem principal
- **YAML**: Configurações estruturadas
- **Docker**: Containerização
- **Git**: Controle de versão

### Padrões de Design

#### Arquitetura MVC
- **Model**: Gerenciadores de dados (modelos, datasets, treinamento)
- **View**: Interface Streamlit (páginas e componentes)
- **Controller**: Lógica de negócio e coordenação

#### Princípios SOLID
- **Single Responsibility**: Cada módulo tem uma responsabilidade específica
- **Open/Closed**: Extensível para novos tipos de modelos e datasets
- **Liskov Substitution**: Interfaces consistentes entre componentes
- **Interface Segregation**: Interfaces específicas e focadas
- **Dependency Inversion**: Dependências abstraídas e injetáveis

#### Padrões Utilizados
- **Factory Pattern**: Criação de modelos e datasets
- **Observer Pattern**: Monitoramento de progresso de treinamento
- **Strategy Pattern**: Diferentes estratégias de fine-tuning
- **Singleton Pattern**: Configurações globais

## 🔧 Configuração Avançada

### Variáveis de Ambiente

```bash
# Configurações de GPU
CUDA_VISIBLE_DEVICES=0,1,2,3
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Configurações de HuggingFace
HF_TOKEN=seu_token_aqui
HF_HOME=/path/to/hf/cache

# Configurações de logging
LOG_LEVEL=INFO
LOG_FILE=/path/to/logs/neuraltrain.log

# Configurações de armazenamento
DATA_DIR=/path/to/data
MODELS_DIR=/path/to/models
OUTPUTS_DIR=/path/to/outputs
```

### Configuração YAML

```yaml
# configs/config.yaml
app:
  name: "NeuralTrain Forge"
  version: "1.0.0"
  debug: false

server:
  host: "0.0.0.0"
  port: 8501
  max_upload_size: 200  # MB

training:
  default_batch_size: 4
  default_learning_rate: 2e-4
  default_epochs: 3
  checkpoint_dir: "./checkpoints"
  logs_dir: "./logs"

models:
  cache_dir: "./data/models"
  supported_formats: [".gguf", ".bin", ".safetensors", ".pt"]
  max_size_gb: 50

datasets:
  cache_dir: "./data/datasets"
  supported_formats: [".txt", ".jsonl", ".csv", ".parquet"]
  max_size_gb: 10
  train_split: 0.8
  validation_split: 0.2
```

### Configuração Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivos de dependências
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fonte
COPY . .

# Expor porta
EXPOSE 8501

# Comando de inicialização
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🚀 Deployment

### Deployment Local

```bash
# Desenvolvimento
streamlit run main.py

# Produção local
streamlit run main.py --server.port=8501 --server.address=0.0.0.0
```

### Deployment com Docker

```bash
# Build da imagem
docker build -t neuraltrain-forge .

# Execução do container
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --gpus all \
  neuraltrain-forge
```

### Deployment em Cloud

#### Paperspace Gradient

```python
# gradient_deploy.py
from gradient import sdk_client

client = sdk_client.SdkClient()

deployment = client.deployments.create(
    name="neuraltrain-forge",
    project_id="your-project-id",
    spec={
        "image": "neuraltrain-forge:latest",
        "port": 8501,
        "resources": {
            "replicas": 1,
            "instance_type": "P4000"
        }
    }
)
```

#### RunPod

```yaml
# runpod-config.yaml
apiVersion: v1
kind: Pod
metadata:
  name: neuraltrain-forge
spec:
  containers:
  - name: app
    image: neuraltrain-forge:latest
    ports:
    - containerPort: 8501
    resources:
      limits:
        nvidia.com/gpu: 1
      requests:
        memory: "8Gi"
        cpu: "4"
```

## 📊 Monitoramento e Logging

### Sistema de Logging

O NeuralTrain Forge implementa um sistema de logging estruturado e abrangente:

```python
# Exemplo de configuração de logging
import logging
from src.utils.logging_utils import setup_logger

# Logger principal
logger = setup_logger(
    name="neuraltrain",
    level=logging.INFO,
    file_path="./logs/neuraltrain.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Logging de treinamento
training_logger = setup_logger(
    name="training",
    level=logging.DEBUG,
    file_path="./logs/training.log"
)
```

### Métricas de Sistema

- **Uso de GPU**: Monitoramento de VRAM e utilização
- **Uso de CPU**: Monitoramento de processamento
- **Uso de RAM**: Monitoramento de memória
- **Uso de Disco**: Monitoramento de armazenamento
- **Rede**: Monitoramento de transferência de dados

### Integração com TensorBoard

```python
# Configuração TensorBoard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="./logs/tensorboard")

# Durante o treinamento
writer.add_scalar("Loss/Train", train_loss, epoch)
writer.add_scalar("Loss/Validation", val_loss, epoch)
writer.add_scalar("Learning_Rate", lr, epoch)
```

## 🧪 Testes

### Estrutura de Testes

```
tests/
├── unit/                  # Testes unitários
│   ├── test_model_manager.py
│   ├── test_dataset_manager.py
│   └── test_training_manager.py
├── integration/           # Testes de integração
│   ├── test_upload_flow.py
│   └── test_training_flow.py
├── e2e/                   # Testes end-to-end
│   └── test_complete_workflow.py
└── fixtures/              # Dados de teste
    ├── sample_model.gguf
    └── sample_dataset.jsonl
```

### Executando Testes

```bash
# Todos os testes
pytest tests/

# Testes específicos
pytest tests/unit/test_model_manager.py

# Com cobertura
pytest --cov=src tests/

# Testes de integração
pytest tests/integration/ -v
```

## 🔒 Segurança

### Práticas de Segurança

- **Validação de Entrada**: Todos os uploads são validados
- **Sanitização**: Limpeza de dados de entrada
- **Autenticação**: Suporte a tokens HuggingFace
- **Autorização**: Controle de acesso a recursos
- **Criptografia**: Dados sensíveis criptografados
- **Auditoria**: Log de todas as ações importantes

### Configurações de Segurança

```python
# Configurações de segurança
SECURITY_CONFIG = {
    "max_upload_size": 200 * 1024 * 1024,  # 200MB
    "allowed_extensions": [".gguf", ".bin", ".safetensors"],
    "scan_uploads": True,
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 60
    },
    "authentication": {
        "required": False,
        "providers": ["huggingface"]
    }
}
```

## 🤝 Contribuição

### Como Contribuir

1. **Fork** o repositório
2. **Crie** uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. **Push** para a branch (`git push origin feature/AmazingFeature`)
5. **Abra** um Pull Request

### Diretrizes de Contribuição

- Siga o padrão de código existente
- Adicione testes para novas funcionalidades
- Atualize a documentação quando necessário
- Use mensagens de commit descritivas
- Mantenha PRs focados e pequenos

### Código de Conduta

Este projeto adere ao [Contributor Covenant](https://www.contributor-covenant.org/). Ao participar, você deve seguir este código.

## 📝 Changelog

### v1.0.0 (2024-06-11)

#### ✨ Funcionalidades
- Interface web completa com Streamlit
- Sistema de upload de modelos (local e HuggingFace)
- Sistema de upload de datasets (múltiplos formatos)
- Configuração completa de fine-tuning com LoRA/QLoRA
- Dashboard com métricas e visualizações
- Sistema de resultados e análise
- Monitoramento em tempo real
- Exportação de modelos e relatórios

#### 🏗️ Arquitetura
- Arquitetura modular e extensível
- Separação clara de responsabilidades
- Sistema de logging estruturado
- Configuração via YAML
- Suporte a Docker

#### 🔧 Tecnologias
- Python 3.11+
- Streamlit 1.45+
- PyTorch 2.7+
- Transformers 4.52+
- PEFT 0.15+
- Plotly 6.1+

## 🆘 Suporte

### Documentação

- [Documentação Completa](docs/)
- [Guia de Instalação](docs/installation.md)
- [Tutorial de Uso](docs/tutorial.md)
- [API Reference](docs/api.md)
- [FAQ](docs/faq.md)

### Comunidade

- [GitHub Issues](https://github.com/seu-usuario/neuraltrain-forge/issues)
- [Discussions](https://github.com/seu-usuario/neuraltrain-forge/discussions)
- [Discord](https://discord.gg/neuraltrain)
- [Twitter](https://twitter.com/neuraltrain)

### Suporte Comercial

Para suporte comercial, treinamentos ou consultoria, entre em contato:
- Email: support@neuraltrain.com
- Website: https://neuraltrain.com

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- [HuggingFace](https://huggingface.co/) pela biblioteca Transformers e PEFT
- [Streamlit](https://streamlit.io/) pelo framework de interface web
- [PyTorch](https://pytorch.org/) pelo framework de deep learning
- [Plotly](https://plotly.com/) pelas visualizações interativas
- Comunidade open source por todas as contribuições

---

<div align="center">

**NeuralTrain Forge** - Democratizando o Fine-tuning de LLMs

[![GitHub stars](https://img.shields.io/github/stars/seu-usuario/neuraltrain-forge?style=social)](https://github.com/seu-usuario/neuraltrain-forge/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/seu-usuario/neuraltrain-forge?style=social)](https://github.com/seu-usuario/neuraltrain-forge/network/members)
[![GitHub issues](https://img.shields.io/github/issues/seu-usuario/neuraltrain-forge)](https://github.com/seu-usuario/neuraltrain-forge/issues)

</div>

