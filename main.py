#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuralTrain Forge - Aplicação Principal
Plataforma de Fine-tuning de Modelos de Linguagem
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Configuração da página (DEVE ser a primeira chamada Streamlit)
st.set_page_config(
    page_title="NeuralTrain Forge",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/NYRALOBOai/neuraltrain-forge',
        'Report a bug': 'https://github.com/NYRALOBOai/neuraltrain-forge/issues',
        'About': "NeuralTrain Forge v2.0.1 - Plataforma de Fine-tuning de LLMs"
    }
)

# Adicionar src ao path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Verificar se o diretório src existe
if not src_path.exists():
    st.error(f"❌ Diretório 'src' não encontrado em: {src_path}")
    st.info("Certifique-se de que está executando o script a partir do diretório raiz do projeto.")
    st.stop()

try:
    # Importar componentes da UI
    from ui.components.sidebar import render_sidebar
    from ui.pages import home, model_upload, dataset_upload, training, results, chat
    
    # Importar utilitários
    from utils.logging_utils import setup_logger
    from utils.file_utils import ensure_directories
    
except ImportError as e:
    st.error(f"❌ Erro ao importar módulos: {e}")
    st.info("Verifique se todas as dependências estão instaladas e se a estrutura do projeto está correta.")
    st.stop()

# Configurar logging
logger = setup_logger(__name__)

def init_session_state():
    """Inicializa variáveis do session state"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Início"
    
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = {}
    
    if 'training_status' not in st.session_state:
        st.session_state.training_status = "idle"
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def main():
    """Função principal da aplicação"""
    
    # Garantir que os diretórios necessários existem
    ensure_directories()
    
    # Inicializar session state
    init_session_state()
    
    # Renderizar sidebar e obter página selecionada
    selected_page = render_sidebar()
    
    # Atualizar página atual
    st.session_state.current_page = selected_page
    
    # Renderizar página selecionada
    try:
        if selected_page == "Início":
            home.main()
        elif selected_page == "Upload de Modelos":
            model_upload.main()
        elif selected_page == "Upload de Datasets":
            dataset_upload.main()
        elif selected_page == "Configuração de Treino":
            training.main()
        elif selected_page == "Chat & Teste":
            chat.main()
        elif selected_page == "Resultados":
            results.main()
        else:
            st.error(f"❌ Página '{selected_page}' não encontrada")
            
    except Exception as e:
        st.error(f"❌ Erro ao carregar página '{selected_page}': {str(e)}")
        logger.error(f"Erro na página {selected_page}: {e}", exc_info=True)
        
        # Mostrar detalhes do erro em modo debug
        if st.checkbox("🐛 Mostrar detalhes do erro"):
            st.exception(e)

if __name__ == "__main__":
    main()

