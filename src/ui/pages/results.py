#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Página de resultados do NeuralTrain Forge.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json


def render():
    """Renderiza a página de resultados."""
    
    # Header da página
    st.markdown("# 📈 Resultados de Treinamento")
    st.markdown("Visualize métricas, analise performance e baixe modelos treinados")
    
    # Tabs para diferentes visualizações
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Métricas", "🤖 Modelos", "📁 Downloads"])
    
    with tab1:
        render_dashboard()
    
    with tab2:
        render_metrics()
    
    with tab3:
        render_trained_models()
    
    with tab4:
        render_downloads()


def render_dashboard():
    """Renderiza dashboard geral de resultados."""
    
    st.markdown("## 📊 Dashboard de Resultados")
    
    # Métricas principais
    render_summary_metrics()
    
    st.markdown("---")
    
    # Layout em colunas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Gráficos principais
        render_training_overview()
        render_loss_comparison()
    
    with col2:
        # Estatísticas e rankings
        render_model_rankings()
        render_recent_completions()


def render_summary_metrics():
    """Renderiza métricas resumidas."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Treinamentos Concluídos",
            "12",
            delta="3",
            help="Total de treinamentos finalizados com sucesso"
        )
    
    with col2:
        st.metric(
            "Melhor Loss",
            "0.234",
            delta="-0.045",
            help="Menor loss de validação alcançado"
        )
    
    with col3:
        st.metric(
            "Tempo Médio",
            "2.4h",
            delta="-0.3h",
            help="Tempo médio de treinamento"
        )
    
    with col4:
        st.metric(
            "Taxa de Sucesso",
            "92%",
            delta="5%",
            help="Porcentagem de treinamentos bem-sucedidos"
        )


def render_training_overview():
    """Renderiza visão geral dos treinamentos."""
    
    st.markdown("### 📈 Visão Geral dos Treinamentos")
    
    # Dados de exemplo
    dates = pd.date_range(start='2024-01-01', end='2024-01-15', freq='D')
    completed = [0, 1, 1, 2, 1, 3, 2, 1, 2, 3, 1, 2, 4, 2, 1]
    failed = [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=completed,
        mode='lines+markers',
        name='Concluídos',
        line=dict(color='#28a745', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=failed,
        mode='lines+markers',
        name='Falharam',
        line=dict(color='#dc3545', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Treinamentos por Dia",
        xaxis_title="Data",
        yaxis_title="Número de Treinamentos",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_loss_comparison():
    """Renderiza comparação de loss entre modelos."""
    
    st.markdown("### 📉 Comparação de Loss")
    
    # Dados de exemplo
    models = ['LLaMA-7B-v1', 'LLaMA-7B-v2', 'Mistral-7B-v1', 'GPT-2-Custom', 'LLaMA-7B-v3']
    train_loss = [0.245, 0.198, 0.267, 0.312, 0.189]
    val_loss = [0.289, 0.234, 0.298, 0.356, 0.221]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Loss de Treino',
        x=models,
        y=train_loss,
        marker_color='#007bff'
    ))
    
    fig.add_trace(go.Bar(
        name='Loss de Validação',
        x=models,
        y=val_loss,
        marker_color='#28a745'
    ))
    
    fig.update_layout(
        title="Comparação de Loss Final",
        xaxis_title="Modelos",
        yaxis_title="Loss",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_model_rankings():
    """Renderiza ranking de modelos."""
    
    st.markdown("### 🏆 Ranking de Modelos")
    
    rankings = [
        {"rank": 1, "model": "LLaMA-7B-v3", "loss": 0.221, "improvement": "↗️ +12%"},
        {"rank": 2, "model": "LLaMA-7B-v2", "loss": 0.234, "improvement": "↗️ +8%"},
        {"rank": 3, "model": "Mistral-7B-v1", "loss": 0.298, "improvement": "↗️ +5%"},
        {"rank": 4, "model": "GPT-2-Custom", "loss": 0.356, "improvement": "↘️ -2%"}
    ]
    
    for item in rankings:
        with st.container():
            col1, col2, col3 = st.columns([1, 3, 2])
            
            with col1:
                medal = "🥇" if item["rank"] == 1 else "🥈" if item["rank"] == 2 else "🥉" if item["rank"] == 3 else f"{item['rank']}º"
                st.markdown(f"### {medal}")
            
            with col2:
                st.markdown(f"**{item['model']}**")
                st.caption(f"Loss: {item['loss']:.3f}")
            
            with col3:
                st.markdown(item['improvement'])
        
        st.markdown("---")


def render_recent_completions():
    """Renderiza treinamentos recentes."""
    
    st.markdown("### 🕒 Concluídos Recentemente")
    
    recent = [
        {"name": "LLaMA-Chat-v3", "time": "2h atrás", "status": "✅"},
        {"name": "Mistral-Instruct", "time": "5h atrás", "status": "✅"},
        {"name": "GPT-2-Fine", "time": "1d atrás", "status": "❌"},
        {"name": "LLaMA-Code", "time": "2d atrás", "status": "✅"}
    ]
    
    for item in recent:
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{item['name']}**")
                st.caption(item['time'])
            
            with col2:
                st.markdown(item['status'])
        
        st.markdown("---")


def render_metrics():
    """Renderiza análise detalhada de métricas."""
    
    st.markdown("## 📈 Análise de Métricas")
    
    # Seletor de job
    jobs = get_completed_jobs()
    
    if not jobs:
        st.info("📭 Nenhum treinamento concluído para análise")
        return
    
    selected_job = st.selectbox(
        "Selecione um treinamento",
        options=[job['name'] for job in jobs],
        help="Treinamento para análise detalhada"
    )
    
    if selected_job:
        job = next(j for j in jobs if j['name'] == selected_job)
        
        # Informações do job
        render_job_info(job)
        
        # Métricas detalhadas
        col1, col2 = st.columns(2)
        
        with col1:
            render_loss_curves(job)
            render_learning_rate_schedule(job)
        
        with col2:
            render_training_metrics(job)
            render_evaluation_metrics(job)


def render_job_info(job: Dict[str, Any]):
    """Renderiza informações do job."""
    
    st.markdown(f"### 📋 Informações - {job['name']}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Modelo Base", job['base_model'])
    
    with col2:
        st.metric("Dataset", job['dataset'])
    
    with col3:
        st.metric("Duração", job['duration'])
    
    with col4:
        st.metric("Épocas", job['epochs'])
    
    # Configurações
    with st.expander("⚙️ Configurações Utilizadas"):
        config = job.get('config', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Treinamento:**")
            st.text(f"Learning Rate: {config.get('learning_rate', 'N/A')}")
            st.text(f"Batch Size: {config.get('batch_size', 'N/A')}")
            st.text(f"Épocas: {config.get('epochs', 'N/A')}")
            st.text(f"Optimizer: {config.get('optimizer', 'N/A')}")
        
        with col2:
            st.markdown("**LoRA:**")
            lora = config.get('lora', {})
            st.text(f"Rank (r): {lora.get('r', 'N/A')}")
            st.text(f"Alpha: {lora.get('alpha', 'N/A')}")
            st.text(f"Dropout: {lora.get('dropout', 'N/A')}")
            st.text(f"Target Modules: {', '.join(lora.get('target_modules', []))}")


def render_loss_curves(job: Dict[str, Any]):
    """Renderiza curvas de loss."""
    
    st.markdown("#### 📉 Curvas de Loss")
    
    # Dados de exemplo
    steps = list(range(0, 1000, 50))
    train_loss = [0.8 - (i * 0.0006) + (0.05 * (i % 100) / 100) for i in steps]
    val_loss = [0.85 - (i * 0.0005) + (0.08 * (i % 150) / 150) for i in steps]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=steps,
        y=train_loss,
        mode='lines',
        name='Loss de Treino',
        line=dict(color='#007bff', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=steps,
        y=val_loss,
        mode='lines',
        name='Loss de Validação',
        line=dict(color='#28a745', width=2)
    ))
    
    fig.update_layout(
        xaxis_title="Passos",
        yaxis_title="Loss",
        hovermode='x unified',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_learning_rate_schedule(job: Dict[str, Any]):
    """Renderiza schedule da learning rate."""
    
    st.markdown("#### 📊 Learning Rate Schedule")
    
    # Dados de exemplo
    steps = list(range(0, 1000, 50))
    lr = []
    
    for step in steps:
        if step < 30:  # Warmup
            lr.append(2e-4 * (step / 30))
        else:  # Cosine decay
            lr.append(2e-4 * 0.5 * (1 + np.cos(np.pi * (step - 30) / (1000 - 30))))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=steps,
        y=lr,
        mode='lines',
        name='Learning Rate',
        line=dict(color='#ffc107', width=2)
    ))
    
    fig.update_layout(
        xaxis_title="Passos",
        yaxis_title="Learning Rate",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_training_metrics(job: Dict[str, Any]):
    """Renderiza métricas de treinamento."""
    
    st.markdown("#### 🎯 Métricas de Treinamento")
    
    metrics = {
        "Loss Final": "0.234",
        "Perplexidade": "1.26",
        "Gradiente Norm": "0.89",
        "Tempo por Época": "45min"
    }
    
    for metric, value in metrics.items():
        st.metric(metric, value)


def render_evaluation_metrics(job: Dict[str, Any]):
    """Renderiza métricas de avaliação."""
    
    st.markdown("#### 📊 Métricas de Avaliação")
    
    metrics = {
        "BLEU Score": "0.78",
        "ROUGE-L": "0.82",
        "Accuracy": "89.5%",
        "F1 Score": "0.87"
    }
    
    for metric, value in metrics.items():
        st.metric(metric, value)


def render_trained_models():
    """Renderiza lista de modelos treinados."""
    
    st.markdown("## 🤖 Modelos Treinados")
    
    # Filtros
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_filter = st.selectbox(
            "Modelo Base",
            ["Todos", "LLaMA", "Mistral", "GPT"]
        )
    
    with col2:
        status_filter = st.selectbox(
            "Status",
            ["Todos", "Disponível", "Arquivado"]
        )
    
    with col3:
        sort_by = st.selectbox(
            "Ordenar por",
            ["Data", "Performance", "Nome"]
        )
    
    # Lista de modelos
    models = get_trained_models()
    
    if not models:
        st.info("📭 Nenhum modelo treinado disponível")
    else:
        for model in models:
            render_model_card(model)


def render_model_card(model: Dict[str, Any]):
    """Renderiza card de modelo treinado."""
    
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            st.markdown(f"**🤖 {model['name']}**")
            st.caption(f"Base: {model['base_model']} | Dataset: {model['dataset']}")
            st.caption(f"Treinado em: {model['trained_date']}")
        
        with col2:
            st.metric("Loss", model['final_loss'])
        
        with col3:
            st.metric("Tamanho", model['size'])
        
        with col4:
            # Menu de ações
            action = st.selectbox(
                "Ações",
                ["Selecionar", "Testar", "Baixar", "Detalhes", "Arquivar"],
                key=f"model_action_{model['id']}",
                label_visibility="collapsed"
            )
            
            if action == "Testar":
                test_model(model)
            elif action == "Baixar":
                download_model(model)
            elif action == "Detalhes":
                show_model_details(model)
            elif action == "Arquivar":
                archive_model(model)
        
        st.markdown("---")


def render_downloads():
    """Renderiza seção de downloads."""
    
    st.markdown("## 📁 Downloads e Exportação")
    
    # Tipos de download
    tab1, tab2, tab3 = st.tabs(["🤖 Modelos", "📊 Relatórios", "📈 Dados"])
    
    with tab1:
        render_model_downloads()
    
    with tab2:
        render_report_downloads()
    
    with tab3:
        render_data_exports()


def render_model_downloads():
    """Renderiza downloads de modelos."""
    
    st.markdown("### 🤖 Download de Modelos")
    
    models = get_trained_models()
    
    for model in models:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{model['name']}**")
                st.caption(f"Tamanho: {model['size']} | Formato: {model['format']}")
            
            with col2:
                st.markdown(f"**{model['final_loss']}**")
                st.caption("Loss Final")
            
            with col3:
                if st.button(f"📥 Baixar", key=f"download_model_{model['id']}", use_container_width=True):
                    download_model(model)
        
        st.markdown("---")


def render_report_downloads():
    """Renderiza downloads de relatórios."""
    
    st.markdown("### 📊 Relatórios de Treinamento")
    
    report_types = [
        {"name": "Relatório Completo", "description": "Todas as métricas e gráficos", "format": "PDF"},
        {"name": "Métricas CSV", "description": "Dados de treinamento em CSV", "format": "CSV"},
        {"name": "Configurações", "description": "Configurações utilizadas", "format": "JSON"},
        {"name": "Logs de Treinamento", "description": "Logs detalhados", "format": "TXT"}
    ]
    
    for report in report_types:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{report['name']}**")
                st.caption(report['description'])
            
            with col2:
                st.markdown(f"**{report['format']}**")
                st.caption("Formato")
            
            with col3:
                if st.button(f"📥 Gerar", key=f"report_{report['name']}", use_container_width=True):
                    generate_report(report)
        
        st.markdown("---")


def render_data_exports():
    """Renderiza exportação de dados."""
    
    st.markdown("### 📈 Exportação de Dados")
    
    # Seletor de job
    jobs = get_completed_jobs()
    
    if jobs:
        selected_job = st.selectbox(
            "Selecione um treinamento",
            options=[job['name'] for job in jobs]
        )
        
        if selected_job:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Dados Disponíveis:**")
                
                data_types = st.multiselect(
                    "Selecione os dados",
                    ["Métricas de Treinamento", "Métricas de Validação", "Learning Rate", "Gradientes", "Checkpoints"],
                    default=["Métricas de Treinamento", "Métricas de Validação"]
                )
            
            with col2:
                st.markdown("**Formato de Exportação:**")
                
                export_format = st.selectbox(
                    "Formato",
                    ["CSV", "JSON", "Excel", "Parquet"]
                )
                
                include_config = st.checkbox("Incluir configurações", value=True)
            
            if st.button("📤 Exportar Dados", type="primary", use_container_width=True):
                export_training_data(selected_job, data_types, export_format, include_config)


def get_completed_jobs():
    """Retorna lista de jobs concluídos."""
    return [
        {
            'name': 'LLaMA-Chat-v3',
            'base_model': 'LLaMA-7B',
            'dataset': 'conversational_data',
            'duration': '2h 15min',
            'epochs': 3,
            'final_loss': 0.221,
            'config': {
                'learning_rate': 2e-4,
                'batch_size': 4,
                'epochs': 3,
                'optimizer': 'adamw',
                'lora': {
                    'r': 16,
                    'alpha': 32,
                    'dropout': 0.1,
                    'target_modules': ['q_proj', 'v_proj']
                }
            }
        }
    ]


def get_trained_models():
    """Retorna lista de modelos treinados."""
    return [
        {
            'id': 'model_001',
            'name': 'LLaMA-Chat-v3',
            'base_model': 'LLaMA-7B',
            'dataset': 'conversational_data',
            'trained_date': '2024-01-15',
            'final_loss': '0.221',
            'size': '13.5GB',
            'format': 'SafeTensors'
        },
        {
            'id': 'model_002',
            'name': 'Mistral-Instruct-v1',
            'base_model': 'Mistral-7B',
            'dataset': 'instruction_following',
            'trained_date': '2024-01-14',
            'final_loss': '0.298',
            'size': '14.2GB',
            'format': 'SafeTensors'
        }
    ]


def test_model(model: Dict[str, Any]):
    """Testa modelo treinado."""
    st.success(f"🧪 Iniciando teste do modelo {model['name']}...")


def download_model(model: Dict[str, Any]):
    """Baixa modelo treinado."""
    st.success(f"📥 Preparando download do modelo {model['name']}...")


def show_model_details(model: Dict[str, Any]):
    """Mostra detalhes do modelo."""
    with st.expander(f"📋 Detalhes - {model['name']}", expanded=True):
        st.json(model)


def archive_model(model: Dict[str, Any]):
    """Arquiva modelo."""
    st.success(f"📦 Modelo {model['name']} arquivado!")


def generate_report(report: Dict[str, Any]):
    """Gera relatório."""
    st.success(f"📊 Gerando {report['name']}...")


def export_training_data(job_name: str, data_types: List[str], format: str, include_config: bool):
    """Exporta dados de treinamento."""
    st.success(f"📤 Exportando dados de {job_name} em formato {format}...")


# Importação necessária para o gráfico de learning rate
import numpy as np

