#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuralTrain Forge - Aplicação principal
Plataforma de fine-tuning de modelos de linguagem
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Adiciona o diretório src ao path de forma mais robusta
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Importa componentes da UI
from ui.components import sidebar
from ui.pages import home, model_upload, dataset_upload, training, results


def main():
    """Função principal da aplicação."""
    
    # Configuração da página
    st.set_page_config(
        page_title="NeuralTrain Forge",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com',
            'Report a bug': 'https://github.com',
            'About': """
            # NeuralTrain Forge 🧠
            
            Plataforma de fine-tuning de modelos de linguagem (LLMs) com suporte a:
            - LoRA e QLoRA
            - Modelos HuggingFace
            - Datasets personalizados
            - Monitoramento em tempo real
            
            **Versão:** 1.0.0
            **Framework:** Streamlit + PEFT + Transformers
            """
        }
    )
    
    # CSS personalizado
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .stAlert {
        margin: 1rem 0;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stSelectbox > div > div {
        border-radius: 0.5rem;
    }
    
    .stTextInput > div > div {
        border-radius: 0.5rem;
    }
    
    .stTextArea > div > div {
        border-radius: 0.5rem;
    }
    
    .stFileUploader > div {
        border-radius: 0.5rem;
        border: 2px dashed #cccccc;
        padding: 2rem;
        text-align: center;
    }
    
    .stProgress > div > div {
        border-radius: 1rem;
    }
    
    .sidebar .stButton > button {
        margin: 0.25rem 0;
    }
    
    /* Estilo para cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    /* Estilo para status */
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    /* Animações */
    .stSpinner > div {
        border-color: #007bff transparent transparent transparent;
    }
    
    /* Responsividade */
    @media (max-width: 768px) {
        .main > div {
            padding-top: 1rem;
        }
        
        .stColumns > div {
            margin: 0.5rem 0;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Inicialização do estado da sessão
    initialize_session_state()
    
    # Renderiza sidebar
    sidebar.render()
    
    # Roteamento de páginas
    page = st.session_state.get('page', 'home')
    
    # Renderiza página atual
    if page == 'home':
        home.render()
    elif page == 'model_upload':
        model_upload.render()
    elif page == 'dataset_upload':
        dataset_upload.render()
    elif page == 'training':
        training.render()
    elif page == 'results':
        results.render()
    else:
        # Página padrão
        home.render()
    
    # Footer
    render_footer()


def initialize_session_state():
    """Inicializa variáveis do estado da sessão."""
    
    # Página atual
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    # Estado do treinamento
    if 'training_in_progress' not in st.session_state:
        st.session_state.training_in_progress = False
    
    if 'training_status' not in st.session_state:
        st.session_state.training_status = {
            'is_training': False,
            'current_job': None,
            'progress': 0,
            'logs': [],
            'metrics': {}
        }
    
    # Históricos
    if 'upload_history' not in st.session_state:
        st.session_state.upload_history = []
    
    if 'dataset_history' not in st.session_state:
        st.session_state.dataset_history = []
    
    # Configurações
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    
    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = True
    
    # Cache de dados
    if 'loaded_models' not in st.session_state:
        st.session_state.loaded_models = {}
    
    if 'loaded_datasets' not in st.session_state:
        st.session_state.loaded_datasets = {}


def render_footer():
    """Renderiza footer da aplicação."""
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p><strong>NeuralTrain Forge v1.0.0</strong></p>
            <p>Plataforma de Fine-tuning de LLMs | Desenvolvido com ❤️ usando Streamlit</p>
            <p>
                <a href="https://github.com" target="_blank">📖 Documentação</a> | 
                <a href="https://github.com" target="_blank">🐛 Reportar Bug</a> | 
                <a href="https://github.com" target="_blank">💡 Sugestões</a>
            </p>
        </div>
        """, unsafe_allow_html=True)


def check_dependencies():
    """Verifica se todas as dependências estão instaladas."""
    
    required_packages = [
        'streamlit',
        'torch',
        'transformers',
        'peft',
        'datasets',
        'plotly',
        'pandas',
        'numpy',
        'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        st.error(f"""
        ❌ **Dependências ausentes:**
        
        Os seguintes pacotes não estão instalados:
        {', '.join(missing_packages)}
        
        Execute: `pip install {' '.join(missing_packages)}`
        """)
        st.stop()


def show_system_info():
    """Mostra informações do sistema se debug mode estiver ativo."""
    
    if st.session_state.get('debug_mode', False):
        with st.expander("🐛 Debug Info", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Session State:**")
                st.json(dict(st.session_state))
            
            with col2:
                st.markdown("**System Info:**")
                
                try:
                    import torch
                    st.text(f"PyTorch: {torch.__version__}")
                    st.text(f"CUDA Available: {torch.cuda.is_available()}")
                    if torch.cuda.is_available():
                        st.text(f"CUDA Version: {torch.version.cuda}")
                        st.text(f"GPU Count: {torch.cuda.device_count()}")
                except ImportError:
                    st.text("PyTorch: Not installed")
                
                try:
                    import transformers
                    st.text(f"Transformers: {transformers.__version__}")
                except ImportError:
                    st.text("Transformers: Not installed")
                
                try:
                    import peft
                    st.text(f"PEFT: {peft.__version__}")
                except ImportError:
                    st.text("PEFT: Not installed")


if __name__ == "__main__":
    # Verifica dependências
    check_dependencies()
    
    # Executa aplicação principal
    main()
    
    # Mostra informações de debug se ativado
    show_system_info()

