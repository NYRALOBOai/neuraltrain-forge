#!/bin/bash

# NeuralTrain Forge - Script de Inicialização
# Este script garante que o ambiente virtual seja ativado antes de executar a aplicação

set -e  # Sair em caso de erro

echo "🚀 Iniciando NeuralTrain Forge..."

# Verificar se estamos no diretório correto
if [ ! -f "main.py" ]; then
    echo "❌ Erro: main.py não encontrado. Execute este script no diretório do projeto."
    exit 1
fi

# Verificar se o ambiente virtual existe
if [ ! -d "venv" ]; then
    echo "❌ Erro: Ambiente virtual não encontrado. Execute ./install.sh primeiro."
    exit 1
fi

# Ativar ambiente virtual
echo "🔧 Ativando ambiente virtual..."
source venv/bin/activate

# Verificar se as dependências estão instaladas
echo "📦 Verificando dependências..."
python3 -c "import streamlit, torch, transformers" 2>/dev/null || {
    echo "❌ Erro: Dependências não encontradas. Execute ./install.sh primeiro."
    exit 1
}

# Verificar se a estrutura de diretórios está correta
if [ ! -d "src/ui" ]; then
    echo "❌ Erro: Estrutura de diretórios incorreta. Verifique se src/ui existe."
    exit 1
fi

# Definir porta (padrão 8501, ou usar variável de ambiente)
PORT=${STREAMLIT_PORT:-8501}

echo "🌐 Iniciando aplicação na porta $PORT..."
echo "📱 Acesse: http://localhost:$PORT"
echo "🛑 Para parar: Ctrl+C"
echo ""

# Executar aplicação
streamlit run main.py --server.port $PORT --server.address 0.0.0.0

