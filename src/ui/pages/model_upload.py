#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Página de upload de modelos do NeuralTrain Forge.
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import json


def render():
    """Renderiza a página de upload de modelos."""
    
    # Header da página
    st.markdown("# 📤 Upload de Modelos")
    st.markdown("Carregue modelos locais ou baixe do HuggingFace Hub")
    
    # Tabs para diferentes métodos de upload
    tab1, tab2, tab3 = st.tabs(["📁 Upload Local", "🤗 HuggingFace Hub", "📋 Modelos Carregados"])
    
    with tab1:
        render_local_upload()
    
    with tab2:
        render_huggingface_download()
    
    with tab3:
        render_loaded_models()


def render_local_upload():
    """Renderiza interface para upload de modelos locais."""
    
    st.markdown("## 📁 Upload de Arquivo Local")
    st.markdown("Carregue modelos nos formatos .gguf, .bin ou .safetensors")
    
    # Informações sobre formatos suportados
    with st.expander("ℹ️ Formatos Suportados"):
        st.markdown("""
        **Formatos aceitos:**
        - **.gguf**: Modelos quantizados (llama.cpp)
        - **.bin**: Modelos PyTorch tradicionais
        - **.safetensors**: Formato seguro do HuggingFace
        
        **Tamanho máximo:** 1GB por arquivo
        
        **Estrutura recomendada:**
        - Para modelos completos: inclua config.json e tokenizer
        - Para modelos únicos: apenas o arquivo do modelo
        """)
    
    # Upload de arquivo
    uploaded_file = st.file_uploader(
        "Selecione o arquivo do modelo",
        type=['gguf', 'bin', 'safetensors'],
        help="Arraste e solte ou clique para selecionar"
    )
    
    if uploaded_file is not None:
        # Mostra informações do arquivo
        file_details = {
            "Nome": uploaded_file.name,
            "Tamanho": f"{uploaded_file.size / (1024*1024):.2f} MB",
            "Tipo": uploaded_file.type or "Desconhecido"
        }
        
        st.markdown("### 📄 Informações do Arquivo")
        for key, value in file_details.items():
            st.info(f"**{key}:** {value}")
        
        # Configurações do modelo
        st.markdown("### ⚙️ Configurações")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.text_input(
                "Nome do modelo",
                value=Path(uploaded_file.name).stem,
                help="Nome que será usado para identificar o modelo"
            )
        
        with col2:
            model_type = st.selectbox(
                "Tipo do modelo",
                ["Auto-detectar", "LLaMA", "Mistral", "GPT", "Outro"],
                help="Tipo do modelo para configurações específicas"
            )
        
        # Descrição opcional
        description = st.text_area(
            "Descrição (opcional)",
            placeholder="Descreva o modelo, sua origem, características especiais...",
            help="Informações adicionais sobre o modelo"
        )
        
        # Validação do nome
        if model_name:
            if not model_name.replace('_', '').replace('-', '').isalnum():
                st.error("❌ Nome deve conter apenas letras, números, hífens e underscores")
            elif len(model_name) < 3:
                st.error("❌ Nome deve ter pelo menos 3 caracteres")
            else:
                # Botão de upload
                if st.button("📤 Carregar Modelo", type="primary", use_container_width=True):
                    upload_local_model(uploaded_file, model_name, model_type, description)
        else:
            st.warning("⚠️ Digite um nome para o modelo")


def render_huggingface_download():
    """Renderiza interface para download do HuggingFace."""
    
    st.markdown("## 🤗 Download do HuggingFace Hub")
    st.markdown("Baixe modelos diretamente do repositório HuggingFace")
    
    # Informações sobre o HuggingFace
    with st.expander("ℹ️ Como usar"):
        st.markdown("""
        **Exemplos de modelos populares:**
        - `microsoft/DialoGPT-medium`: Modelo conversacional
        - `gpt2`: GPT-2 original
        - `distilgpt2`: Versão menor do GPT-2
        - `EleutherAI/gpt-neo-1.3B`: GPT-Neo 1.3B
        
        **Formato do ID:** `organizacao/nome-do-modelo`
        
        **Nota:** Alguns modelos podem ser grandes (vários GB)
        """)
    
    # Modelos populares
    st.markdown("### 🌟 Modelos Populares")
    
    popular_models = [
        {
            "id": "microsoft/DialoGPT-medium",
            "name": "DialoGPT Medium",
            "description": "Modelo conversacional baseado em GPT-2",
            "size": "~1.5GB"
        },
        {
            "id": "gpt2",
            "name": "GPT-2",
            "description": "Modelo original GPT-2 da OpenAI",
            "size": "~500MB"
        },
        {
            "id": "distilgpt2",
            "name": "DistilGPT-2",
            "description": "Versão destilada e menor do GPT-2",
            "size": "~350MB"
        }
    ]
    
    cols = st.columns(len(popular_models))
    
    for i, model in enumerate(popular_models):
        with cols[i]:
            with st.container():
                st.markdown(f"**{model['name']}**")
                st.caption(model['description'])
                st.caption(f"Tamanho: {model['size']}")
                
                if st.button(f"📥 Baixar", key=f"download_{i}", use_container_width=True):
                    download_huggingface_model(model['id'], model['name'])
    
    st.markdown("---")
    
    # Download personalizado
    st.markdown("### 🔍 Download Personalizado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_id = st.text_input(
            "ID do modelo",
            placeholder="ex: microsoft/DialoGPT-medium",
            help="ID completo do modelo no HuggingFace Hub"
        )
    
    with col2:
        local_name = st.text_input(
            "Nome local (opcional)",
            placeholder="Nome para salvar localmente",
            help="Se vazio, usará o nome original"
        )
    
    # Configurações avançadas
    with st.expander("⚙️ Configurações Avançadas"):
        col1, col2 = st.columns(2)
        
        with col1:
            config_name = st.text_input(
                "Configuração específica",
                placeholder="ex: default",
                help="Nome da configuração específica (opcional)"
            )
        
        with col2:
            revision = st.text_input(
                "Revisão/Branch",
                placeholder="ex: main",
                help="Branch ou commit específico (opcional)"
            )
        
        use_auth_token = st.checkbox(
            "Usar token de autenticação",
            help="Para modelos privados ou com restrições"
        )
        
        if use_auth_token:
            auth_token = st.text_input(
                "Token HuggingFace",
                type="password",
                help="Seu token de acesso do HuggingFace"
            )
    
    # Validação e download
    if model_id:
        if '/' not in model_id:
            st.warning("⚠️ ID deve estar no formato 'organizacao/modelo'")
        else:
            if st.button("🤗 Baixar do HuggingFace", type="primary", use_container_width=True):
                download_huggingface_model(
                    model_id, 
                    local_name or model_id.replace('/', '_'),
                    config_name or None,
                    revision or None
                )


def render_loaded_models():
    """Renderiza lista de modelos carregados."""
    
    st.markdown("## 📋 Modelos Carregados")
    st.markdown("Gerencie os modelos disponíveis no sistema")
    
    # Botão para atualizar lista
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔄 Atualizar", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("🗑️ Limpar Cache", use_container_width=True):
            clear_model_cache()
    
    # Lista de modelos (placeholder)
    # Em uma implementação real, isso viria do ModelManager
    models = get_loaded_models_list()
    
    if not models:
        st.info("📭 Nenhum modelo carregado ainda")
        st.markdown("Use as abas acima para carregar seu primeiro modelo!")
    else:
        # Filtros
        st.markdown("### 🔍 Filtros")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_type = st.selectbox(
                "Tipo",
                ["Todos", "LLaMA", "Mistral", "GPT", "Outro"]
            )
        
        with col2:
            filter_source = st.selectbox(
                "Origem",
                ["Todas", "Local", "HuggingFace"]
            )
        
        with col3:
            sort_by = st.selectbox(
                "Ordenar por",
                ["Nome", "Data", "Tamanho", "Tipo"]
            )
        
        # Lista de modelos
        st.markdown("### 📚 Modelos Disponíveis")
        
        for i, model in enumerate(models):
            render_model_card(model, i)


def render_model_card(model: Dict[str, Any], index: int):
    """Renderiza card de um modelo."""
    
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            st.markdown(f"**🤖 {model['name']}**")
            st.caption(f"Tipo: {model['type']} | Origem: {model['source']}")
            if model.get('description'):
                st.caption(model['description'])
        
        with col2:
            st.metric("Tamanho", model['size'])
        
        with col3:
            status_color = "🟢" if model['status'] == 'loaded' else "🔵"
            st.markdown(f"{status_color} {model['status']}")
        
        with col4:
            # Menu de ações
            action = st.selectbox(
                "Ações",
                ["Selecionar", "Carregar", "Detalhes", "Remover"],
                key=f"action_{index}",
                label_visibility="collapsed"
            )
            
            if action == "Carregar":
                load_model(model['name'])
            elif action == "Detalhes":
                show_model_details(model)
            elif action == "Remover":
                remove_model(model['name'])
        
        st.markdown("---")


def upload_local_model(uploaded_file, model_name: str, model_type: str, description: str):
    """Processa upload de modelo local."""
    
    try:
        with st.spinner("📤 Carregando modelo..."):
            # Cria arquivo temporário
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Aqui seria chamado o ModelManager para processar o upload
            # success, message = model_manager.upload_model(tmp_file_path, model_name)
            
            # Simulação de sucesso
            success = True
            message = f"Modelo {model_name} carregado com sucesso!"
            
            # Remove arquivo temporário
            os.unlink(tmp_file_path)
            
            if success:
                st.success(f"✅ {message}")
                
                # Mostra informações do modelo carregado
                st.balloons()
                
                # Adiciona ao histórico
                add_to_upload_history("local", model_name, uploaded_file.size)
                
            else:
                st.error(f"❌ {message}")
                
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {str(e)}")


def download_huggingface_model(model_id: str, local_name: str, config_name: Optional[str] = None, revision: Optional[str] = None):
    """Processa download do HuggingFace."""
    
    try:
        with st.spinner(f"🤗 Baixando {model_id}..."):
            # Aqui seria chamado o ModelManager para baixar do HuggingFace
            # success, message = model_manager.download_from_huggingface(model_id, local_name)
            
            # Simulação de sucesso
            success = True
            message = f"Modelo {model_id} baixado com sucesso!"
            
            if success:
                st.success(f"✅ {message}")
                st.balloons()
                
                # Adiciona ao histórico
                add_to_upload_history("huggingface", local_name, "N/A")
                
            else:
                st.error(f"❌ {message}")
                
    except Exception as e:
        st.error(f"❌ Erro ao baixar modelo: {str(e)}")


def get_loaded_models_list():
    """Retorna lista de modelos carregados (placeholder)."""
    
    # Em uma implementação real, isso viria do ModelManager
    return [
        {
            "name": "LLaMA-7B-Chat",
            "type": "LLaMA",
            "source": "HuggingFace",
            "size": "13.5GB",
            "status": "loaded",
            "description": "Modelo conversacional baseado em LLaMA"
        },
        {
            "name": "Mistral-7B-Instruct",
            "type": "Mistral", 
            "source": "Local",
            "size": "14.2GB",
            "status": "available",
            "description": "Modelo de instrução Mistral"
        },
        {
            "name": "Custom-GPT2",
            "type": "GPT",
            "source": "Local",
            "size": "548MB",
            "status": "loaded",
            "description": "Modelo GPT-2 personalizado"
        }
    ]


def load_model(model_name: str):
    """Carrega modelo na memória."""
    
    with st.spinner(f"🔄 Carregando {model_name}..."):
        # Aqui seria chamado o ModelManager
        # success, message, model = model_manager.load_model(model_name)
        
        # Simulação
        success = True
        message = f"Modelo {model_name} carregado na memória"
        
        if success:
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")


def show_model_details(model: Dict[str, Any]):
    """Mostra detalhes do modelo."""
    
    with st.expander(f"📋 Detalhes - {model['name']}", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Informações Básicas:**")
            st.text(f"Nome: {model['name']}")
            st.text(f"Tipo: {model['type']}")
            st.text(f"Origem: {model['source']}")
            st.text(f"Tamanho: {model['size']}")
            st.text(f"Status: {model['status']}")
        
        with col2:
            st.markdown("**Configurações:**")
            st.text("Arquitetura: Transformer")
            st.text("Precisão: FP16")
            st.text("Contexto: 2048 tokens")
            st.text("Vocabulário: 32000")
        
        if model.get('description'):
            st.markdown("**Descrição:**")
            st.text(model['description'])


def remove_model(model_name: str):
    """Remove modelo do sistema."""
    
    if st.button(f"⚠️ Confirmar remoção de {model_name}", type="secondary"):
        with st.spinner(f"🗑️ Removendo {model_name}..."):
            # Aqui seria chamado o ModelManager
            # success, message = model_manager.delete_model(model_name)
            
            # Simulação
            success = True
            message = f"Modelo {model_name} removido com sucesso"
            
            if success:
                st.success(f"✅ {message}")
                st.rerun()
            else:
                st.error(f"❌ {message}")


def clear_model_cache():
    """Limpa cache de modelos."""
    
    with st.spinner("🧹 Limpando cache..."):
        # Aqui seria implementada a limpeza do cache
        st.success("✅ Cache limpo com sucesso!")


def add_to_upload_history(source: str, name: str, size):
    """Adiciona entrada ao histórico de uploads."""
    
    if 'upload_history' not in st.session_state:
        st.session_state.upload_history = []
    
    entry = {
        'timestamp': str(pd.Timestamp.now()),
        'source': source,
        'name': name,
        'size': size
    }
    
    st.session_state.upload_history.append(entry)
    
    # Mantém apenas os últimos 10
    if len(st.session_state.upload_history) > 10:
        st.session_state.upload_history = st.session_state.upload_history[-10:]

