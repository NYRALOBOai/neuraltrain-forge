# 🚀 Guia de Início Rápido - NeuralTrain Forge

Este guia irá ajudá-lo a configurar e executar o NeuralTrain Forge em poucos minutos.

## ⚡ Instalação Rápida (Recomendada)

### 1. Clone o Repositório
```bash
git clone https://github.com/seu-usuario/neuraltrain-forge.git
cd neuraltrain-forge
```

### 2. Execute o Script de Instalação
```bash
chmod +x install.sh
./install.sh
```

O script irá:
- ✅ Verificar dependências do sistema
- ✅ Criar ambiente virtual Python
- ✅ Instalar todas as dependências
- ✅ Configurar estrutura de diretórios
- ✅ Testar a instalação

### 3. Inicie a Aplicação
```bash
./start.sh
```

### 4. Acesse no Navegador
Abra seu navegador e vá para: **http://localhost:8501**

---

## 🛠️ Instalação Manual

Se preferir instalar manualmente ou o script automático não funcionar:

### Pré-requisitos
- Python 3.10 ou superior
- pip (gerenciador de pacotes Python)
- Git

### Passos

1. **Clone e entre no diretório:**
```bash
git clone https://github.com/seu-usuario/neuraltrain-forge.git
cd neuraltrain-forge
```

2. **Crie ambiente virtual:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Instale dependências:**
```bash
pip install -r requirements.txt
```

4. **Configure ambiente:**
```bash
cp .env.example .env
# Edite o arquivo .env conforme necessário
```

5. **Execute a aplicação:**
```bash
streamlit run main.py
```

---

## 🐳 Usando Docker

### Opção 1: Docker Compose (Recomendada)
```bash
# Construir e executar
docker-compose up --build

# Executar em background
docker-compose up -d

# Parar
docker-compose down
```

### Opção 2: Docker Manual
```bash
# Construir imagem
docker build -t neuraltrain-forge .

# Executar container
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --gpus all \
  neuraltrain-forge
```

---

## 🎯 Primeiro Uso

### 1. Acesse o Dashboard
- Abra http://localhost:8501
- Você verá o dashboard principal com métricas do sistema

### 2. Carregue um Modelo
- Vá para **"📤 Upload de Modelos"**
- Escolha entre:
  - **Upload Local**: Carregue um arquivo .gguf, .bin ou .safetensors
  - **HuggingFace Hub**: Baixe diretamente do HuggingFace

### 3. Carregue um Dataset
- Vá para **"📊 Upload de Datasets"**
- Carregue um arquivo .txt, .jsonl, .csv ou .parquet
- O sistema irá validar e processar automaticamente

### 4. Configure o Treinamento
- Vá para **"⚙️ Configuração de Treino"**
- Preencha:
  - Nome do job
  - Selecione modelo e dataset
  - Configure parâmetros LoRA/QLoRA
  - Ajuste hiperparâmetros

### 5. Inicie o Treinamento
- Clique em **"🚀 Iniciar Treinamento"**
- Monitore o progresso em tempo real
- Veja métricas e gráficos atualizados

### 6. Analise os Resultados
- Vá para **"📈 Resultados"**
- Visualize gráficos de loss e métricas
- Baixe o modelo treinado
- Exporte relatórios

---

## 📋 Exemplos Práticos

### Exemplo 1: Fine-tuning de Modelo de Chat

1. **Modelo**: LLaMA-7B-Chat (do HuggingFace)
2. **Dataset**: Conversações em formato JSONL
3. **Configuração**:
   - Tipo: LoRA
   - Rank: 16
   - Alpha: 32
   - Learning Rate: 2e-4
   - Épocas: 3

### Exemplo 2: Fine-tuning para Tarefa Específica

1. **Modelo**: Mistral-7B-Instruct
2. **Dataset**: Dados de instrução em CSV
3. **Configuração**:
   - Tipo: QLoRA
   - Rank: 8
   - Alpha: 16
   - Learning Rate: 1e-4
   - Épocas: 5

---

## 🔧 Configurações Importantes

### GPU
Se você tem GPU NVIDIA:
```bash
# Verificar se CUDA está disponível
nvidia-smi

# Configurar no .env
CUDA_VISIBLE_DEVICES=0
```

### HuggingFace Token
Para modelos privados:
```bash
# No arquivo .env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Limites de Memória
Para evitar problemas de memória:
```bash
# No arquivo .env
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

## 🚨 Solução de Problemas Comuns

### Erro: "Port 8501 is already in use"
```bash
# Encontrar processo usando a porta
lsof -i :8501

# Matar processo
kill -9 <PID>

# Ou usar porta diferente
streamlit run main.py --server.port 8502
```

### Erro: "CUDA out of memory"
- Reduza o batch size
- Habilite gradient checkpointing
- Use QLoRA em vez de LoRA
- Reduza max_length

### Erro: "Module not found"
```bash
# Reativar ambiente virtual
source venv/bin/activate

# Reinstalar dependências
pip install -r requirements.txt
```

### Upload falha
- Verifique tamanho do arquivo (limite: 200MB)
- Verifique formato suportado
- Verifique espaço em disco

---

## 📚 Próximos Passos

1. **Leia a documentação completa**: [README.md](README.md)
2. **Explore exemplos avançados**: [docs/examples/](docs/examples/)
3. **Configure para produção**: [docs/deployment.md](docs/deployment.md)
4. **Contribua para o projeto**: [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 🆘 Precisa de Ajuda?

- 📖 **Documentação**: [README.md](README.md)
- 🐛 **Reportar Bug**: [GitHub Issues](https://github.com/seu-usuario/neuraltrain-forge/issues)
- 💬 **Discussões**: [GitHub Discussions](https://github.com/seu-usuario/neuraltrain-forge/discussions)
- 📧 **Email**: support@neuraltrain.com

---

## ✅ Checklist de Verificação

Antes de começar, certifique-se de que:

- [ ] Python 3.10+ está instalado
- [ ] Git está instalado
- [ ] Você tem pelo menos 8GB de RAM
- [ ] Você tem pelo menos 50GB de espaço livre
- [ ] (Opcional) GPU NVIDIA com drivers CUDA
- [ ] Conexão com internet para downloads

---

**🎉 Pronto! Você está preparado para usar o NeuralTrain Forge!**

Divirta-se explorando o fine-tuning de modelos de linguagem! 🧠✨

