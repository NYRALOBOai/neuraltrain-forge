# 🔧 Solução de Problemas - NeuralTrain Forge

## ❌ Erro: "ModuleNotFoundError: No module named 'ui'"

### 🔍 Diagnóstico
Este erro ocorre quando o Python não consegue encontrar o módulo `ui` que está localizado em `src/ui/`. 

### 🛠️ Soluções

#### ✅ Solução 1: Usar o Script de Inicialização (Recomendada)
```bash
# Use sempre o script de inicialização
./start.sh
```

#### ✅ Solução 2: Ativar Ambiente Virtual Manualmente
```bash
# Ative o ambiente virtual antes de executar
source venv/bin/activate
streamlit run main.py
```

#### ✅ Solução 3: Verificar Estrutura de Diretórios
```bash
# Verifique se a estrutura está correta
ls -la src/ui/
# Deve mostrar: components/ pages/ __init__.py
```

#### ✅ Solução 4: Reinstalar Dependências
```bash
# Se o problema persistir, reinstale
./install.sh
```

### 🔍 Verificações Adicionais

#### 1. Verificar se está no diretório correto
```bash
pwd
# Deve mostrar: /caminho/para/neuraltrain-forge
ls main.py
# Deve existir
```

#### 2. Verificar ambiente virtual
```bash
ls venv/
# Deve existir e conter: bin/ lib/ include/
```

#### 3. Verificar arquivos __init__.py
```bash
find src -name "__init__.py"
# Deve mostrar todos os arquivos __init__.py
```

#### 4. Testar importações manualmente
```bash
source venv/bin/activate
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.') / 'src'))
from ui.components import sidebar
print('✓ Importação funcionando!')
"
```

---

## ❌ Erro: "streamlit: command not found"

### 🛠️ Solução
```bash
# Ative o ambiente virtual
source venv/bin/activate
# Ou reinstale
./install.sh
```

---

## ❌ Erro: "Permission denied"

### 🛠️ Solução
```bash
# Torne os scripts executáveis
chmod +x install.sh start.sh
```

---

## ❌ Erro: "Port 8501 is already in use"

### 🛠️ Soluções

#### Opção 1: Matar processo existente
```bash
# Encontrar processo
lsof -i :8501
# Matar processo (substitua PID)
kill -9 <PID>
```

#### Opção 2: Usar porta diferente
```bash
# Definir porta diferente
export STREAMLIT_PORT=8502
./start.sh
# Ou diretamente
streamlit run main.py --server.port 8502
```

---

## ❌ Erro: "CUDA out of memory"

### 🛠️ Soluções
1. Reduza o batch size nas configurações
2. Habilite gradient checkpointing
3. Use QLoRA em vez de LoRA
4. Reduza max_length do modelo

---

## ❌ Erro: "No module named 'torch'"

### 🛠️ Solução
```bash
# Reinstale dependências
source venv/bin/activate
pip install torch torchvision torchaudio
# Ou reinstale tudo
./install.sh
```

---

## 🆘 Solução Geral para Problemas de Importação

### Script de Diagnóstico Completo
```bash
#!/bin/bash
echo "🔍 Diagnóstico NeuralTrain Forge"
echo "================================"

echo "📁 Diretório atual:"
pwd

echo "📄 Arquivos principais:"
ls -la main.py install.sh start.sh 2>/dev/null || echo "❌ Arquivos não encontrados"

echo "📦 Ambiente virtual:"
ls -la venv/ 2>/dev/null || echo "❌ Ambiente virtual não encontrado"

echo "🗂️ Estrutura src:"
find src -type f -name "*.py" | head -10 2>/dev/null || echo "❌ Estrutura src não encontrada"

echo "🐍 Python e dependências:"
source venv/bin/activate 2>/dev/null && {
    python3 --version
    pip list | grep -E "(streamlit|torch|transformers)"
} || echo "❌ Não foi possível ativar ambiente virtual"

echo "🧪 Teste de importação:"
source venv/bin/activate 2>/dev/null && python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.') / 'src'))
try:
    from ui.components import sidebar
    print('✅ Importações funcionando!')
except Exception as e:
    print(f'❌ Erro: {e}')
" || echo "❌ Não foi possível testar importações"
```

### Salvar como `diagnose.sh` e executar:
```bash
chmod +x diagnose.sh
./diagnose.sh
```

---

## 📞 Suporte

Se os problemas persistirem:

1. **Execute o diagnóstico completo** acima
2. **Copie a saída** do diagnóstico
3. **Reporte o problema** com a saída completa

### Informações Úteis para Suporte:
- Sistema operacional
- Versão do Python
- Saída do diagnóstico
- Mensagem de erro completa
- Passos que levaram ao erro

---

## ✅ Verificação Final

Após resolver o problema, verifique se tudo está funcionando:

```bash
# 1. Execute o diagnóstico
./diagnose.sh

# 2. Inicie a aplicação
./start.sh

# 3. Acesse no navegador
# http://localhost:8501

# 4. Verifique se todas as páginas carregam:
# - Dashboard (página inicial)
# - Upload de Modelos
# - Upload de Datasets  
# - Configuração de Treino
# - Resultados
```

**🎉 Se tudo funcionar, o problema foi resolvido com sucesso!**

