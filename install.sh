#!/bin/bash

# NeuralTrain Forge - Script de Instalação Automatizada
# Versão: 1.0.0
# Autor: Manus AI

set -e  # Parar em caso de erro

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para imprimir mensagens coloridas
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Função para verificar se um comando existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Função para verificar versão do Python
check_python_version() {
    if command_exists python3; then
        python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        required_version="3.10"
        
        if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
            print_message "Python $python_version encontrado ✓"
            return 0
        else
            print_error "Python $python_version encontrado, mas é necessário Python $required_version ou superior"
            return 1
        fi
    else
        print_error "Python3 não encontrado"
        return 1
    fi
}

# Função para instalar dependências do sistema
install_system_dependencies() {
    print_header "=== Instalando Dependências do Sistema ==="
    
    if command_exists apt-get; then
        print_message "Detectado sistema baseado em Debian/Ubuntu"
        sudo apt-get update
        sudo apt-get install -y \
            python3-pip \
            python3-venv \
            python3-dev \
            git \
            curl \
            wget \
            build-essential \
            libssl-dev \
            libffi-dev
    elif command_exists yum; then
        print_message "Detectado sistema baseado em RedHat/CentOS"
        sudo yum update -y
        sudo yum install -y \
            python3-pip \
            python3-devel \
            git \
            curl \
            wget \
            gcc \
            gcc-c++ \
            openssl-devel \
            libffi-devel
    elif command_exists brew; then
        print_message "Detectado macOS com Homebrew"
        brew update
        brew install python3 git curl wget
    else
        print_warning "Sistema operacional não reconhecido. Instale manualmente:"
        print_warning "- Python 3.10+"
        print_warning "- pip"
        print_warning "- git"
        print_warning "- curl/wget"
    fi
}

# Função para criar ambiente virtual
create_virtual_environment() {
    print_header "=== Criando Ambiente Virtual ==="
    
    if [ -d "venv" ]; then
        print_warning "Ambiente virtual já existe. Removendo..."
        rm -rf venv
    fi
    
    print_message "Criando novo ambiente virtual..."
    python3 -m venv venv
    
    print_message "Ativando ambiente virtual..."
    source venv/bin/activate
    
    print_message "Atualizando pip..."
    pip install --upgrade pip setuptools wheel
}

# Função para instalar dependências Python
install_python_dependencies() {
    print_header "=== Instalando Dependências Python ==="
    
    if [ ! -f "requirements.txt" ]; then
        print_error "Arquivo requirements.txt não encontrado!"
        exit 1
    fi
    
    print_message "Instalando dependências do requirements.txt..."
    pip install -r requirements.txt
    
    print_message "Verificando instalação das dependências principais..."
    python3 -c "
import sys
packages = ['streamlit', 'torch', 'transformers', 'peft', 'plotly', 'pandas', 'numpy']
missing = []

for package in packages:
    try:
        __import__(package)
        print(f'✓ {package}')
    except ImportError:
        missing.append(package)
        print(f'✗ {package}')

if missing:
    print(f'\\nPacotes ausentes: {missing}')
    sys.exit(1)
else:
    print('\\n✓ Todas as dependências principais instaladas com sucesso!')
"
}

# Função para configurar estrutura de diretórios
setup_directory_structure() {
    print_header "=== Configurando Estrutura de Diretórios ==="
    
    directories=(
        "data/models"
        "data/datasets" 
        "data/outputs"
        "logs"
        "checkpoints"
        "temp"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            print_message "Criando diretório: $dir"
            mkdir -p "$dir"
            touch "$dir/.gitkeep"
        else
            print_message "Diretório já existe: $dir"
        fi
    done
}

# Função para configurar variáveis de ambiente
setup_environment_variables() {
    print_header "=== Configurando Variáveis de Ambiente ==="
    
    env_file=".env"
    
    if [ ! -f "$env_file" ]; then
        print_message "Criando arquivo .env..."
        cat > "$env_file" << EOF
# NeuralTrain Forge - Configurações de Ambiente

# Configurações de GPU
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Configurações de HuggingFace
# HF_TOKEN=seu_token_aqui
HF_HOME=./cache/huggingface

# Configurações de logging
LOG_LEVEL=INFO
LOG_FILE=./logs/neuraltrain.log

# Configurações de armazenamento
DATA_DIR=./data
MODELS_DIR=./data/models
DATASETS_DIR=./data/datasets
OUTPUTS_DIR=./data/outputs

# Configurações da aplicação
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EOF
        print_message "Arquivo .env criado. Edite conforme necessário."
    else
        print_message "Arquivo .env já existe."
    fi
}

# Função para verificar GPU
check_gpu_support() {
    print_header "=== Verificando Suporte a GPU ==="
    
    if command_exists nvidia-smi; then
        print_message "NVIDIA GPU detectada:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
        
        print_message "Verificando suporte CUDA no PyTorch..."
        python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA disponível: {torch.version.cuda}')
    print(f'✓ GPUs disponíveis: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  - GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('✗ CUDA não disponível - usando CPU')
"
    else
        print_warning "NVIDIA GPU não detectada. A aplicação funcionará apenas com CPU."
    fi
}

# Função para testar a instalação
test_installation() {
    print_header "=== Testando Instalação ==="
    
    print_message "Testando importações principais..."
    python3 -c "
import streamlit as st
import torch
import transformers
import peft
import plotly
import pandas as pd
import numpy as np
print('✓ Todas as importações funcionando!')
"
    
    print_message "Testando aplicação Streamlit..."
    timeout 10s streamlit run main.py --server.headless true --server.port 8502 &
    sleep 5
    
    if curl -s http://localhost:8502 > /dev/null; then
        print_message "✓ Aplicação Streamlit funcionando!"
        pkill -f "streamlit run main.py"
    else
        print_warning "Não foi possível testar a aplicação Streamlit automaticamente."
    fi
}

# Função para criar script de inicialização
create_startup_script() {
    print_header "=== Criando Script de Inicialização ==="
    
    cat > "start.sh" << 'EOF'
#!/bin/bash

# NeuralTrain Forge - Script de Inicialização

# Ativar ambiente virtual
source venv/bin/activate

# Carregar variáveis de ambiente
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Verificar se a porta está livre
PORT=${STREAMLIT_SERVER_PORT:-8501}
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "Porta $PORT já está em uso. Parando processo..."
    kill $(lsof -t -i:$PORT)
    sleep 2
fi

# Iniciar aplicação
echo "Iniciando NeuralTrain Forge na porta $PORT..."
streamlit run main.py \
    --server.port=$PORT \
    --server.address=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0} \
    --browser.gatherUsageStats=false

EOF
    
    chmod +x start.sh
    print_message "Script de inicialização criado: ./start.sh"
}

# Função para exibir informações finais
show_final_info() {
    print_header "=== Instalação Concluída! ==="
    
    echo ""
    print_message "🎉 NeuralTrain Forge instalado com sucesso!"
    echo ""
    print_message "Para iniciar a aplicação:"
    echo "  1. Ative o ambiente virtual: source venv/bin/activate"
    echo "  2. Execute: ./start.sh"
    echo "  3. Ou execute: streamlit run main.py"
    echo ""
    print_message "A aplicação estará disponível em: http://localhost:8501"
    echo ""
    print_message "Arquivos importantes:"
    echo "  - README.md: Documentação completa"
    echo "  - .env: Configurações de ambiente"
    echo "  - start.sh: Script de inicialização"
    echo "  - logs/: Logs da aplicação"
    echo ""
    print_message "Para suporte:"
    echo "  - GitHub: https://github.com/seu-usuario/neuraltrain-forge"
    echo "  - Documentação: ./docs/"
    echo ""
}

# Função principal
main() {
    print_header "🧠 NeuralTrain Forge - Instalação Automatizada"
    print_header "=============================================="
    echo ""
    
    # Verificar se estamos no diretório correto
    if [ ! -f "main.py" ]; then
        print_error "main.py não encontrado. Execute este script no diretório raiz do projeto."
        exit 1
    fi
    
    # Verificar Python
    if ! check_python_version; then
        print_error "Versão do Python incompatível. Instale Python 3.10 ou superior."
        exit 1
    fi
    
    # Perguntar se deve instalar dependências do sistema
    read -p "Instalar dependências do sistema? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_system_dependencies
    fi
    
    # Criar ambiente virtual
    create_virtual_environment
    
    # Instalar dependências Python
    install_python_dependencies
    
    # Configurar estrutura
    setup_directory_structure
    setup_environment_variables
    
    # Verificar GPU
    check_gpu_support
    
    # Criar scripts
    create_startup_script
    
    # Testar instalação
    read -p "Executar testes de instalação? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        test_installation
    fi
    
    # Informações finais
    show_final_info
}

# Executar função principal
main "$@"

