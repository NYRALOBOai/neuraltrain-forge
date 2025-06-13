#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Página inicial do NeuralTrain Forge.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Any


def render():
    """Renderiza a página inicial com dashboard."""
    
    # Header da página
    st.markdown("# 🏠 Dashboard - NeuralTrain Forge")
    st.markdown("Bem-vindo à plataforma de fine-tuning de modelos de linguagem!")
    
    # Métricas principais
    render_main_metrics()
    
    st.markdown("---")
    
    # Layout em colunas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Gráficos e estatísticas
        render_charts()
        
        # Atividade recente
        render_recent_activity()
    
    with col2:
        # Status do sistema
        render_system_status()
        
        # Ações rápidas
        render_quick_actions()
        
        # Recursos úteis
        render_resources()


def render_main_metrics():
    """Renderiza métricas principais em cards."""
    
    st.markdown("## 📊 Visão Geral")
    
    # Cria 4 colunas para métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🤖 Modelos",
            value="3",
            delta="1",
            help="Modelos carregados no sistema"
        )
    
    with col2:
        st.metric(
            label="📊 Datasets",
            value="5",
            delta="2",
            help="Datasets disponíveis para treinamento"
        )
    
    with col3:
        st.metric(
            label="⚙️ Jobs Ativos",
            value="1",
            delta="-1",
            help="Jobs de treinamento em andamento"
        )
    
    with col4:
        st.metric(
            label="✅ Concluídos",
            value="12",
            delta="3",
            help="Treinamentos concluídos com sucesso"
        )


def render_charts():
    """Renderiza gráficos de estatísticas."""
    
    st.markdown("## 📈 Estatísticas de Treinamento")
    
    # Tabs para diferentes gráficos
    tab1, tab2, tab3 = st.tabs(["📊 Progresso", "⏱️ Tempo", "🎯 Precisão"])
    
    with tab1:
        render_progress_chart()
    
    with tab2:
        render_time_chart()
    
    with tab3:
        render_accuracy_chart()


def render_progress_chart():
    """Renderiza gráfico de progresso dos jobs."""
    
    # Dados de exemplo
    jobs_data = {
        'Job': ['job_001', 'job_002', 'job_003', 'job_004', 'job_005'],
        'Progresso': [100, 85, 60, 30, 10],
        'Status': ['Concluído', 'Treinando', 'Treinando', 'Pausado', 'Iniciando']
    }
    
    df = pd.DataFrame(jobs_data)
    
    # Gráfico de barras
    fig = px.bar(
        df, 
        x='Job', 
        y='Progresso',
        color='Status',
        title="Progresso dos Jobs de Treinamento",
        color_discrete_map={
            'Concluído': '#28a745',
            'Treinando': '#007bff', 
            'Pausado': '#ffc107',
            'Iniciando': '#6c757d'
        }
    )
    
    fig.update_layout(
        xaxis_title="Jobs",
        yaxis_title="Progresso (%)",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_time_chart():
    """Renderiza gráfico de tempo de treinamento."""
    
    # Dados de exemplo
    dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='D')
    training_hours = [2.5, 4.2, 3.8, 5.1, 2.9, 6.3, 4.7]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=training_hours,
        mode='lines+markers',
        name='Horas de Treinamento',
        line=dict(color='#007bff', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Tempo de Treinamento por Dia",
        xaxis_title="Data",
        yaxis_title="Horas",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_accuracy_chart():
    """Renderiza gráfico de precisão dos modelos."""
    
    # Dados de exemplo
    models = ['LLaMA-7B', 'Mistral-7B', 'GPT-2', 'Custom-1', 'Custom-2']
    accuracy = [92.5, 89.3, 85.7, 94.1, 87.9]
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=accuracy,
            marker_color=['#28a745' if acc > 90 else '#007bff' if acc > 85 else '#ffc107' for acc in accuracy]
        )
    ])
    
    fig.update_layout(
        title="Precisão dos Modelos Treinados",
        xaxis_title="Modelos",
        yaxis_title="Precisão (%)",
        yaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_recent_activity():
    """Renderiza atividade recente."""
    
    st.markdown("## 🕒 Atividade Recente")
    
    # Lista de atividades de exemplo
    activities = [
        {
            'time': '10:30',
            'action': 'Treinamento concluído',
            'details': 'job_003 - LLaMA-7B fine-tuning',
            'status': 'success'
        },
        {
            'time': '09:45',
            'action': 'Dataset carregado',
            'details': 'conversational_data.jsonl (2.3MB)',
            'status': 'info'
        },
        {
            'time': '09:15',
            'action': 'Modelo baixado',
            'details': 'microsoft/DialoGPT-medium do HuggingFace',
            'status': 'info'
        },
        {
            'time': '08:30',
            'action': 'Treinamento iniciado',
            'details': 'job_004 - Configuração LoRA r=16',
            'status': 'warning'
        },
        {
            'time': '08:00',
            'action': 'Sistema iniciado',
            'details': 'NeuralTrain Forge v1.0.0',
            'status': 'info'
        }
    ]
    
    for activity in activities:
        status_color = {
            'success': '🟢',
            'warning': '🟡', 
            'error': '🔴',
            'info': '🔵'
        }.get(activity['status'], '⚪')
        
        with st.container():
            col1, col2 = st.columns([1, 4])
            
            with col1:
                st.caption(activity['time'])
            
            with col2:
                st.markdown(f"{status_color} **{activity['action']}**")
                st.caption(activity['details'])
        
        st.markdown("---")


def render_system_status():
    """Renderiza status do sistema."""
    
    st.markdown("## 💻 Status do Sistema")
    
    # Status da GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            st.success(f"🎮 GPU: {gpu_name}")
            
            # Uso da VRAM
            allocated = torch.cuda.memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            usage_percent = (allocated / total) * 100
            
            st.progress(usage_percent / 100)
            st.caption(f"VRAM: {allocated:.1f}GB / {total:.1f}GB ({usage_percent:.1f}%)")
        else:
            st.warning("⚠️ GPU não disponível")
    except ImportError:
        st.error("❌ PyTorch não instalado")
    
    # Status da CPU e RAM
    try:
        import psutil
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        st.info(f"🖥️ CPU: {cpu_percent}%")
        st.progress(cpu_percent / 100)
        
        # RAM
        memory = psutil.virtual_memory()
        st.info(f"🧠 RAM: {memory.percent}%")
        st.progress(memory.percent / 100)
        st.caption(f"Disponível: {memory.available / (1024**3):.1f}GB")
        
    except ImportError:
        st.warning("⚠️ psutil não disponível")
    
    # Status dos serviços
    st.markdown("### 🔧 Serviços")
    st.success("✅ Model Manager")
    st.success("✅ Dataset Manager") 
    st.success("✅ Training Manager")
    st.success("✅ File System")


def render_quick_actions():
    """Renderiza ações rápidas."""
    
    st.markdown("## ⚡ Ações Rápidas")
    
    # Botões de ação
    if st.button("📤 Carregar Modelo", use_container_width=True):
        st.session_state.page = "model_upload"
        st.rerun()
    
    if st.button("📊 Carregar Dataset", use_container_width=True):
        st.session_state.page = "dataset_upload"
        st.rerun()
    
    if st.button("⚙️ Novo Treinamento", use_container_width=True):
        st.session_state.page = "training"
        st.rerun()
    
    if st.button("📈 Ver Resultados", use_container_width=True):
        st.session_state.page = "results"
        st.rerun()
    
    st.markdown("---")
    
    # Configurações rápidas
    st.markdown("### ⚙️ Configurações")
    
    # Toggle para modo debug
    debug_mode = st.checkbox("🐛 Modo Debug", value=False)
    if debug_mode:
        st.session_state.debug_mode = True
        st.info("Modo debug ativado")
    
    # Seletor de tema (placeholder)
    theme = st.selectbox(
        "🎨 Tema",
        ["Claro", "Escuro", "Auto"],
        index=0
    )


def render_resources():
    """Renderiza recursos úteis."""
    
    st.markdown("## 📚 Recursos")
    
    # Links úteis
    st.markdown("### 🔗 Links Úteis")
    st.markdown("- [📖 Documentação](https://github.com)")
    st.markdown("- [🤗 HuggingFace](https://huggingface.co)")
    st.markdown("- [📊 PEFT](https://github.com/huggingface/peft)")
    st.markdown("- [🔧 Transformers](https://github.com/huggingface/transformers)")
    
    st.markdown("### 💡 Dicas")
    
    with st.expander("🎯 Configuração LoRA"):
        st.markdown("""
        **Parâmetros recomendados:**
        - **r**: 8-16 para modelos pequenos, 32-64 para grandes
        - **alpha**: 2x o valor de r
        - **dropout**: 0.1 para dados pequenos, 0.05 para grandes
        """)
    
    with st.expander("📊 Preparação de Dados"):
        st.markdown("""
        **Formatos suportados:**
        - **JSONL**: Uma linha por exemplo
        - **CSV**: Colunas estruturadas
        - **TXT**: Texto simples
        - **Parquet**: Dados otimizados
        """)
    
    with st.expander("⚡ Otimização"):
        st.markdown("""
        **Para melhor performance:**
        - Use batch size múltiplo de 8
        - Ative gradient checkpointing
        - Use FP16 se disponível
        - Monitore uso de VRAM
        """)


def render_welcome_message():
    """Renderiza mensagem de boas-vindas para novos usuários."""
    
    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = True
    
    if st.session_state.first_visit:
        st.info("""
        👋 **Bem-vindo ao NeuralTrain Forge!**
        
        Esta é sua primeira visita. Para começar:
        1. 📤 Carregue um modelo base
        2. 📊 Adicione um dataset
        3. ⚙️ Configure o treinamento
        4. 🚀 Inicie o fine-tuning
        """)
        
        if st.button("✅ Entendi, não mostrar novamente"):
            st.session_state.first_visit = False
            st.rerun()


def render_with_welcome():
    """Renderiza página com mensagem de boas-vindas."""
    
    # Mensagem de boas-vindas
    render_welcome_message()
    
    # Conteúdo principal
    render()

