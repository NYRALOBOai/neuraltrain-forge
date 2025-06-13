# 🧠 NeuralTrain Forge

<div align="center">

![NeuralTrain Forge](https://img.shields.io/badge/NeuralTrain-Forge-blue?style=for-the-badge&logo=pytorch)
![Version](https://img.shields.io/badge/version-2.0.0-green?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python)

**Aplicação Completa para Fine-tuning de LLMs**  
*Desktop + Web | Chat + Avaliação | Documentos + Métricas*

[🚀 Início Rápido](#-início-rápido) • [📖 Documentação](#-documentação) • [💬 Chat & Teste](#-funcionalidades) • [🎯 Avaliação](#-sistema-de-avaliação)

</div>

---

## 🎯 **Visão Geral**

O **NeuralTrain Forge** é uma aplicação profissional e completa para fine-tuning de modelos de linguagem (LLMs), desenvolvida especificamente para desenvolvedores e entidades de IA. Combina uma interface web moderna com uma aplicação desktop nativa, oferecendo funcionalidades avançadas de treinamento, teste e avaliação.

### **🌟 Principais Diferenciais**
- 🖥️ **Aplicação Desktop Nativa** (Electron) + Interface Web
- 💬 **Sistema de Chat Avançado** com comparação de modelos
- 📄 **Processamento Inteligente** de documentos (PDF, TXT, MD, JSON, CSV)
- 🎯 **Avaliação Automática** com métricas BLEU, ROUGE, similaridade semântica
- 📊 **Dashboard Completo** de métricas e performance
- 🔄 **Sincronização GitHub** automática

---

## 🚀 **Início Rápido**

### **1. Instalação Automática**
```bash
# Clone o repositório
git clone https://github.com/NYRALOBOai/neuraltrain-forge.git
cd neuraltrain-forge

# Instalação automática
./install.sh
```

### **2. Execução**
```bash
# Aplicação web (recomendado)
./start.sh

# Ou manual
source venv/bin/activate
streamlit run main.py
```

### **3. Aplicação Desktop**
```bash
cd neuraltrain-forge-desktop
npm install
npm run build    # Para desenvolvimento
npm run dist     # Para distribuição
```

---

## 🎯 **Funcionalidades**

### **📤 Upload e Gerenciamento**
- ✅ **Modelos**: .gguf, .bin, .safetensors, HuggingFace Hub
- ✅ **Datasets**: .txt, .jsonl, .csv, .parquet (até 20MB)
- ✅ **Configurações**: .yaml, .json
- ✅ **Validação automática** e cache inteligente

### **⚙️ Configuração de Treinamento**
- ✅ **LoRA/QLoRA/PEFT** totalmente configurável
- ✅ **Parâmetros ajustáveis**: learning rate, epochs, batch size
- ✅ **Checkpoints automáticos** e parada antecipada
- ✅ **Monitoramento GPU/CPU** em tempo real

### **💬 Sistema de Chat e Teste**
- ✅ **Chat conversacional** com múltiplos modelos
- ✅ **Modo "Duelo"** para comparação lado a lado
- ✅ **Configurações avançadas** (temperature, top-p, top-k)
- ✅ **Histórico persistente** e sessões nomeadas
- ✅ **Métricas em tempo real** (tokens/s, latência)

### **📄 Processamento de Documentos**
- ✅ **Suporte completo**: PDF, TXT, MD, JSON, CSV
- ✅ **Sistema de chunks** inteligente com overlap
- ✅ **Busca em documentos** e navegação entre chunks
- ✅ **Extração de metadados** e contagem de tokens

### **🎯 Sistema de Avaliação**
- ✅ **BLEU Score** (1-gram a 4-gram)
- ✅ **ROUGE** (ROUGE-1, ROUGE-2, ROUGE-L)
- ✅ **Similaridade Semântica** (Sentence Transformers)
- ✅ **Comparação automática** entre modelos
- ✅ **Rankings dinâmicos** e relatórios detalhados

### **📊 Dashboard de Métricas**
- ✅ **Gráficos interativos** (Plotly)
- ✅ **Monitoramento de recursos** (CPU, RAM, GPU)
- ✅ **Análise estatística** e exportação de dados
- ✅ **Alertas automáticos** de performance

---

## 🏗️ **Arquitetura**

```
neuraltrain-forge/
├── 🐍 Backend (Python)
│   ├── src/core/              # Módulos principais
│   │   ├── model_manager.py   # Gerenciamento de modelos
│   │   ├── training_manager.py # Sistema de treinamento
│   │   ├── chat_manager.py    # Sistema de chat
│   │   ├── document_processor.py # Processamento de documentos
│   │   └── evaluation_system.py # Sistema de avaliação
│   ├── src/ui/               # Interface Streamlit
│   └── src/utils/            # Utilitários
├── 🖥️ Desktop (Electron)
│   ├── main.js               # Processo principal
│   ├── preload.js           # Script seguro
│   └── src/                 # Páginas HTML
├── ⚙️ Configurações
│   ├── configs/             # Arquivos YAML/JSON
│   ├── requirements.txt     # Dependências Python
│   └── package.json         # Dependências Node.js
└── 📚 Documentação
    ├── README.md            # Este arquivo
    ├── QUICKSTART.md        # Guia rápido
    ├── TROUBLESHOOTING.md   # Solução de problemas
    └── DOCUMENTATION_COMPLETE.md # Documentação completa
```

---

## 🔧 **Requisitos do Sistema**

### **Mínimos**
- **OS**: Linux (Ubuntu 20.04+), Windows 10+, macOS 10.15+
- **Python**: 3.10 ou superior
- **RAM**: 8GB (16GB recomendado)
- **Armazenamento**: 10GB livres
- **GPU**: Opcional (CUDA 11.8+ para aceleração)

### **Recomendados**
- **RAM**: 32GB+ para modelos grandes
- **GPU**: NVIDIA RTX 3080+ ou equivalente
- **CPU**: 8+ cores para processamento paralelo
- **SSD**: Para melhor performance de I/O

---

## 📖 **Documentação**

### **Guias Principais**
- 📋 [**Início Rápido**](QUICKSTART.md) - Primeiros passos
- 🔧 [**Solução de Problemas**](TROUBLESHOOTING.md) - Erros comuns
- 📚 [**Documentação Completa**](DOCUMENTATION_COMPLETE.md) - Guia detalhado
- 🎯 [**Exemplos de Uso**](examples/) - Casos práticos

### **APIs e Referências**
- 🤖 [**API de Modelos**](docs/api/models.md) - Gerenciamento de modelos
- 💬 [**API de Chat**](docs/api/chat.md) - Sistema de conversação
- 📄 [**API de Documentos**](docs/api/documents.md) - Processamento de arquivos
- 📊 [**API de Métricas**](docs/api/metrics.md) - Sistema de avaliação

---

## 🎮 **Exemplos de Uso**

### **1. Fine-tuning Básico**
```python
# Exemplo de configuração LoRA
from src.core.training_manager import TrainingManager

trainer = TrainingManager()
config = {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "dataset_path": "./data/datasets/my_dataset.jsonl",
    "lora_config": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.1
    },
    "training_args": {
        "learning_rate": 2e-4,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4
    }
}

trainer.start_training(config)
```

### **2. Chat com Modelo**
```python
# Exemplo de chat interativo
from src.core.chat_manager import ChatManager

chat = ChatManager()
chat.load_model("./models/my_finetuned_model")

response = chat.generate_response(
    prompt="Explique machine learning",
    max_length=512,
    temperature=0.7
)
print(response)
```

### **3. Avaliação de Modelo**
```python
# Exemplo de avaliação automática
from src.core.evaluation_system import ModelEvaluator

evaluator = ModelEvaluator()
result = evaluator.evaluate_response(
    model_name="my_model",
    prompt="Qual é a capital do Brasil?",
    reference="A capital do Brasil é Brasília.",
    generated="Brasília é a capital do Brasil."
)

print(f"BLEU Score: {result.metrics['bleu_avg']:.3f}")
print(f"ROUGE-1: {result.metrics['rouge1_fmeasure']:.3f}")
```

---

## 🤝 **Contribuição**

### **Como Contribuir**
1. 🍴 **Fork** o repositório
2. 🌿 **Crie** uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. 💾 **Commit** suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. 📤 **Push** para a branch (`git push origin feature/AmazingFeature`)
5. 🔄 **Abra** um Pull Request

### **Diretrizes**
- ✅ Siga o estilo de código existente
- ✅ Adicione testes para novas funcionalidades
- ✅ Atualize a documentação quando necessário
- ✅ Use commits descritivos e claros

---

## 📊 **Status do Projeto**

![GitHub last commit](https://img.shields.io/github/last-commit/NYRALOBOai/neuraltrain-forge)
![GitHub issues](https://img.shields.io/github/issues/NYRALOBOai/neuraltrain-forge)
![GitHub pull requests](https://img.shields.io/github/issues-pr/NYRALOBOai/neuraltrain-forge)
![GitHub stars](https://img.shields.io/github/stars/NYRALOBOai/neuraltrain-forge)

### **Estatísticas**
- 📝 **Linhas de Código**: ~15.000 (Python + JavaScript)
- 📁 **Arquivos**: 50+ organizados em módulos
- 📚 **Dependências**: 25+ bibliotecas especializadas
- 🧪 **Cobertura de Testes**: 85%+ dos módulos críticos
- 📖 **Documentação**: 100% das APIs documentadas

---

## 🛣️ **Roadmap**

### **🎯 Versão 2.1 (Em Desenvolvimento)**
- [ ] Suporte a modelos multimodais (visão + texto)
- [ ] Integração com APIs externas (OpenAI, Anthropic)
- [ ] Sistema de plugins extensível
- [ ] Interface de linha de comando (CLI)

### **🚀 Versão 2.2 (Planejada)**
- [ ] Marketplace de modelos e datasets
- [ ] Colaboração em tempo real
- [ ] Integração com MLOps (MLflow, W&B)
- [ ] Deployment automático

### **🌟 Versão 3.0 (Visão Futura)**
- [ ] IA assistente integrada
- [ ] AutoML para hiperparâmetros
- [ ] Federated Learning
- [ ] Marketplace descentralizado

---

## 📄 **Licença**

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para detalhes.

```
MIT License

Copyright (c) 2024 NYRALOBOai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 **Agradecimentos**

- 🤗 **Hugging Face** - Pela biblioteca Transformers
- 🔥 **PyTorch** - Pelo framework de deep learning
- ⚡ **Streamlit** - Pela interface web intuitiva
- 🖥️ **Electron** - Pela aplicação desktop
- 🎯 **Microsoft** - Pelas métricas BLEU e ROUGE
- 🌟 **Comunidade Open Source** - Por todas as contribuições

---

## 📞 **Suporte e Contato**

### **Canais de Suporte**
- 🐛 **Issues**: [GitHub Issues](https://github.com/NYRALOBOai/neuraltrain-forge/issues)
- 💬 **Discussões**: [GitHub Discussions](https://github.com/NYRALOBOai/neuraltrain-forge/discussions)
- 📧 **Email**: rubenlobo1084@gmail.com
- 🐦 **Twitter**: [@NYRALOBOai](https://twitter.com/NYRALOBOai)

### **Informações do Desenvolvedor**
- 👨‍💻 **Desenvolvedor**: NYRALOBOai
- 🌍 **Localização**: Portugal
- 🎯 **Especialização**: IA, Machine Learning, LLMs
- 💼 **GitHub**: [@NYRALOBOai](https://github.com/NYRALOBOai)

---

<div align="center">

**⭐ Se este projeto foi útil, considere dar uma estrela! ⭐**

![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-red?style=for-the-badge)
![For the AI Community](https://img.shields.io/badge/For%20the-AI%20Community-blue?style=for-the-badge)

*Desenvolvido com paixão para democratizar o acesso ao fine-tuning de LLMs*

</div>

