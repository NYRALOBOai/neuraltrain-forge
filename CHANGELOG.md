# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
e este projeto adere ao [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-06-11

### ✨ Adicionado
- Interface web completa desenvolvida com Streamlit
- Sistema de upload de modelos com suporte a múltiplos formatos (.gguf, .bin, .safetensors)
- Integração com HuggingFace Hub para download direto de modelos
- Sistema de upload de datasets (.txt, .jsonl, .csv, .parquet)
- Configuração completa de fine-tuning com suporte a LoRA e QLoRA
- Dashboard principal com métricas em tempo real
- Visualizações interativas com Plotly
- Sistema de monitoramento de recursos (GPU, RAM, CPU)
- Página de resultados com análise detalhada de treinamentos
- Sistema de logging estruturado
- Exportação de modelos e relatórios
- Suporte a Docker para containerização
- Script de instalação automatizada
- Documentação completa

### 🏗️ Arquitetura
- Arquitetura modular baseada em MVC
- Separação clara entre frontend (UI) e backend (Core)
- Gerenciadores especializados para modelos, datasets e treinamento
- Sistema de configuração via YAML
- Utilitários reutilizáveis para logging e manipulação de arquivos

### 🔧 Tecnologias
- Python 3.11+ como linguagem principal
- Streamlit 1.45+ para interface web
- PyTorch 2.7+ para deep learning
- Transformers 4.52+ para modelos de linguagem
- PEFT 0.15+ para fine-tuning eficiente
- Plotly 6.1+ para visualizações
- Pandas 2.3+ para manipulação de dados
- NumPy 2.3+ para computação numérica

### 📊 Funcionalidades de Treinamento
- Suporte completo a LoRA (Low-Rank Adaptation)
- Suporte a QLoRA para treinamento em baixa precisão
- Configuração avançada de hiperparâmetros
- Monitoramento em tempo real do progresso
- Sistema de checkpoints automático
- Validação durante o treinamento
- Métricas customizáveis

### 🎨 Interface do Usuário
- Design responsivo e moderno
- Navegação intuitiva entre páginas
- Sidebar informativa com status do sistema
- Gráficos interativos para análise de dados
- Formulários dinâmicos para configuração
- Sistema de notificações e alertas
- Tema personalizado com CSS

### 🔒 Segurança e Qualidade
- Validação rigorosa de uploads
- Sanitização de dados de entrada
- Sistema de logging para auditoria
- Tratamento de erros robusto
- Testes automatizados
- Documentação abrangente

### 🚀 Deployment
- Suporte a deployment local
- Containerização com Docker
- Compatibilidade com Paperspace Gradient
- Compatibilidade com RunPod
- Scripts de automação para setup
- Configuração de ambiente flexível

### 📚 Documentação
- README.md completo com guias de instalação e uso
- Documentação técnica detalhada
- Exemplos de configuração
- Guias de troubleshooting
- Changelog estruturado
- Licença MIT

### 🧪 Testes
- Estrutura de testes organizada
- Testes unitários para componentes principais
- Testes de integração para fluxos completos
- Validação de funcionalidades críticas
- Cobertura de código

### 🤝 Contribuição
- Diretrizes claras para contribuição
- Código de conduta estabelecido
- Templates para issues e pull requests
- Processo de review estruturado

## [Unreleased]

### 🔮 Planejado para Próximas Versões
- Suporte a mais tipos de modelos (GPT, BERT, T5)
- Integração com Weights & Biases
- Sistema de usuários e autenticação
- API REST para automação
- Suporte a treinamento distribuído
- Integração com mais provedores de cloud
- Sistema de templates de configuração
- Métricas avançadas de avaliação
- Suporte a datasets multimodais
- Interface de chat para testar modelos

### 🐛 Correções Planejadas
- Otimização de uso de memória
- Melhorias na velocidade de upload
- Aprimoramento do sistema de logs
- Correções de compatibilidade

---

## Tipos de Mudanças

- `✨ Adicionado` para novas funcionalidades
- `🔧 Modificado` para mudanças em funcionalidades existentes
- `🐛 Corrigido` para correções de bugs
- `🗑️ Removido` para funcionalidades removidas
- `🔒 Segurança` para correções de vulnerabilidades
- `📚 Documentação` para mudanças na documentação
- `🏗️ Arquitetura` para mudanças estruturais
- `🎨 Interface` para mudanças na UI/UX
- `⚡ Performance` para melhorias de performance
- `🧪 Testes` para adição ou modificação de testes

