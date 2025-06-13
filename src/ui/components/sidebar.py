#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Componente da sidebar para navegação no NeuralTrain Forge.
"""

import streamlit as st
from typing import Dict, Any


def render():
    """Renderiza a sidebar com navegação e informações do sistema."""
    
    with st.sidebar:
        # Logo e título
        st.markdown("# 🧠 NeuralTrain Forge")
        st.markdown("### Plataforma de Fine-tuning")
        st.markdown("---")
        
        # Menu de navegação
        st.markdown("## 📋 Navegação")
        
        # Botões de navegação
        if st.button("🏠 Início", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
        
        if st.button("📤 Upload de Modelos", use_container_width=True):
            st.session_state.page = "model_upload"
            st.rerun()
        
        if st.button("📊 Upload de Datasets", use_container_width=True):
            st.session_state.page = "dataset_upload"
            st.rerun()
        
        if st.button("⚙️ Configuração de Treino", use_container_width=True):
            st.session_state.page = "training"
            st.rerun()
        
        if st.button("📈 Resultados", use_container_width=True):
            st.session_state.page = "results"
            st.rerun()
        
        st.markdown("---")
        
        # Informações do sistema
        st.markdown("## 💻 Sistema")
        
        # Status da GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                st.success(f"🎮 GPU: {gpu_name}")
                st.info(f"💾 VRAM: {gpu_memory:.1f}GB")
            else:
                st.warning("⚠️ GPU não disponível")
        except ImportError:
            st.error("❌ PyTorch não instalado")
        
        # Informações de memória
        try:
            import psutil
            memory = psutil.virtual_memory()
            st.info(f"🧠 RAM: {memory.available / (1024**3):.1f}GB disponível")
            st.info(f"💽 Uso: {memory.percent}%")
        except ImportError:
            pass
        
        st.markdown("---")
        
        # Status do treinamento
        st.markdown("## 🔄 Status do Treino")
        
        # Verifica se há treinamento em andamento
        if 'training_status' in st.session_state:
            status = st.session_state.training_status
            
            if status.get('is_training', False):
                st.warning("🔄 Treinamento em andamento")
                
                # Progress bar
                progress = status.get('progress', 0)
                st.progress(progress / 100)
                st.caption(f"Progresso: {progress:.1f}%")
                
                # Job atual
                current_job = status.get('current_job')
                if current_job:
                    st.caption(f"Job: {current_job}")
                
                # Botão para parar treinamento
                if st.button("⏹️ Parar Treino", type="secondary", use_container_width=True):
                    # Aqui seria chamada a função para parar o treinamento
                    st.warning("Funcionalidade de parar treino será implementada")
            else:
                st.success("✅ Sistema pronto")
        else:
            st.success("✅ Sistema pronto")
        
        st.markdown("---")
        
        # Informações da aplicação
        st.markdown("## ℹ️ Sobre")
        st.caption("**Versão:** 1.0.0")
        st.caption("**Framework:** Streamlit + PEFT")
        st.caption("**Suporte:** LoRA, QLoRA")
        
        # Links úteis
        st.markdown("## 🔗 Links")
        st.markdown("[📖 Documentação](https://github.com)")
        st.markdown("[🐛 Reportar Bug](https://github.com)")
        st.markdown("[💡 Sugestões](https://github.com)")


def show_page_indicator(current_page: str):
    """
    Mostra indicador da página atual.
    
    Args:
        current_page: Nome da página atual
    """
    page_names = {
        "home": "🏠 Início",
        "model_upload": "📤 Upload de Modelos", 
        "dataset_upload": "📊 Upload de Datasets",
        "training": "⚙️ Configuração de Treino",
        "results": "📈 Resultados"
    }
    
    current_name = page_names.get(current_page, current_page)
    st.sidebar.markdown(f"**📍 Página atual:** {current_name}")


def show_quick_stats():
    """Mostra estatísticas rápidas na sidebar."""
    
    st.sidebar.markdown("## 📊 Estatísticas")
    
    # Placeholder para estatísticas
    # Estas seriam obtidas dos managers em uma implementação real
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.metric("Modelos", "0", help="Modelos carregados")
        st.metric("Jobs", "0", help="Jobs de treinamento")
    
    with col2:
        st.metric("Datasets", "0", help="Datasets disponíveis")
        st.metric("Concluídos", "0", help="Treinamentos concluídos")


def show_resource_monitor():
    """Mostra monitor de recursos na sidebar."""
    
    st.sidebar.markdown("## 📊 Recursos")
    
    try:
        import psutil
        import torch
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        st.sidebar.progress(cpu_percent / 100)
        st.sidebar.caption(f"CPU: {cpu_percent}%")
        
        # Memória
        memory = psutil.virtual_memory()
        st.sidebar.progress(memory.percent / 100)
        st.sidebar.caption(f"RAM: {memory.percent}%")
        
        # GPU (se disponível)
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            st.sidebar.progress(gpu_memory / 100)
            st.sidebar.caption(f"VRAM: {gpu_memory:.1f}%")
            
    except Exception as e:
        st.sidebar.error(f"Erro ao obter recursos: {e}")


def show_recent_activity():
    """Mostra atividade recente na sidebar."""
    
    st.sidebar.markdown("## 🕒 Atividade Recente")
    
    # Placeholder para atividade recente
    activities = [
        "📤 Modelo carregado: LLaMA-7B",
        "📊 Dataset processado: chat_data",
        "⚙️ Treino iniciado: job_001",
        "✅ Treino concluído: job_000"
    ]
    
    for activity in activities[:3]:  # Mostra apenas os 3 mais recentes
        st.sidebar.caption(activity)
    
    if len(activities) > 3:
        st.sidebar.caption(f"... e mais {len(activities) - 3} atividades")


def render_advanced():
    """Versão avançada da sidebar com mais funcionalidades."""
    
    with st.sidebar:
        # Header principal
        st.markdown("# 🧠 NeuralTrain Forge")
        st.markdown("### Plataforma de Fine-tuning")
        
        # Indicador da página atual
        show_page_indicator(st.session_state.get('page', 'home'))
        
        st.markdown("---")
        
        # Menu de navegação principal
        render()
        
        # Estatísticas rápidas
        show_quick_stats()
        
        # Monitor de recursos
        show_resource_monitor()
        
        # Atividade recente
        show_recent_activity()

