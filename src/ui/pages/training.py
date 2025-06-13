#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Página de configuração de treinamento do NeuralTrain Forge.
"""

import streamlit as st
import json
from datetime import datetime
from typing import Dict, Any, List, Optional


def render():
    """Renderiza a página de configuração de treinamento."""
    
    # Header da página
    st.markdown("# ⚙️ Configuração de Treinamento")
    st.markdown("Configure e inicie jobs de fine-tuning de modelos")
    
    # Verifica se há treinamento em andamento
    if check_training_in_progress():
        render_training_monitor()
    else:
        # Tabs para diferentes funcionalidades
        tab1, tab2, tab3 = st.tabs(["🚀 Novo Treinamento", "📋 Jobs Ativos", "📊 Histórico"])
        
        with tab1:
            render_new_training()
        
        with tab2:
            render_active_jobs()
        
        with tab3:
            render_training_history()


def render_new_training():
    """Renderiza interface para criar novo treinamento."""
    
    st.markdown("## 🚀 Configurar Novo Treinamento")
    
    # Verificações preliminares
    models_available = get_available_models()
    datasets_available = get_available_datasets()
    
    if not models_available:
        st.error("❌ Nenhum modelo disponível. Carregue um modelo primeiro.")
        if st.button("📤 Ir para Upload de Modelos"):
            st.session_state.page = "model_upload"
            st.rerun()
        return
    
    if not datasets_available:
        st.error("❌ Nenhum dataset disponível. Carregue um dataset primeiro.")
        if st.button("📊 Ir para Upload de Datasets"):
            st.session_state.page = "dataset_upload"
            st.rerun()
        return
    
    # Formulário de configuração
    with st.form("training_config"):
        # Configurações básicas
        st.markdown("### 📋 Configurações Básicas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            job_name = st.text_input(
                "Nome do Job",
                value=f"training_{datetime.now().strftime('%Y%m%d_%H%M')}",
                help="Nome único para identificar este treinamento"
            )
            
            selected_model = st.selectbox(
                "Modelo Base",
                options=[m['name'] for m in models_available],
                help="Modelo que será usado como base para o fine-tuning"
            )
        
        with col2:
            selected_dataset = st.selectbox(
                "Dataset",
                options=[d['name'] for d in datasets_available],
                help="Dataset que será usado para o treinamento"
            )
            
            training_type = st.selectbox(
                "Tipo de Treinamento",
                ["LoRA", "QLoRA", "Full Fine-tuning"],
                help="Método de fine-tuning a ser usado"
            )
        
        # Configurações de LoRA
        if training_type in ["LoRA", "QLoRA"]:
            st.markdown("### 🎯 Configurações LoRA")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                lora_r = st.number_input(
                    "Rank (r)",
                    min_value=1,
                    max_value=256,
                    value=16,
                    help="Rank das matrizes LoRA (maior = mais parâmetros)"
                )
                
                lora_alpha = st.number_input(
                    "Alpha",
                    min_value=1,
                    max_value=512,
                    value=32,
                    help="Fator de escala LoRA (geralmente 2x o rank)"
                )
            
            with col2:
                lora_dropout = st.slider(
                    "Dropout",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.1,
                    step=0.05,
                    help="Taxa de dropout para regularização"
                )
                
                target_modules = st.multiselect(
                    "Módulos Alvo",
                    ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    default=["q_proj", "v_proj"],
                    help="Camadas que receberão adaptadores LoRA"
                )
            
            with col3:
                bias = st.selectbox(
                    "Bias",
                    ["none", "all", "lora_only"],
                    index=0,
                    help="Como tratar os bias durante o treinamento"
                )
        
        # Configurações de treinamento
        st.markdown("### 🏃 Configurações de Treinamento")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_epochs = st.number_input(
                "Épocas",
                min_value=1,
                max_value=100,
                value=3,
                help="Número de épocas de treinamento"
            )
            
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=1e-6,
                max_value=1e-2,
                value=2e-4,
                format="%.2e",
                help="Taxa de aprendizado"
            )
        
        with col2:
            batch_size = st.selectbox(
                "Batch Size",
                [1, 2, 4, 8, 16, 32],
                index=2,
                help="Tamanho do batch (ajuste conforme VRAM)"
            )
            
            gradient_accumulation = st.number_input(
                "Acumulação de Gradiente",
                min_value=1,
                max_value=32,
                value=4,
                help="Passos para acumular gradientes"
            )
        
        with col3:
            max_length = st.number_input(
                "Comprimento Máximo",
                min_value=128,
                max_value=4096,
                value=512,
                help="Máximo de tokens por sequência"
            )
            
            warmup_ratio = st.slider(
                "Warmup Ratio",
                min_value=0.0,
                max_value=0.2,
                value=0.03,
                step=0.01,
                help="Proporção de passos para warmup"
            )
        
        # Configurações avançadas
        with st.expander("🔧 Configurações Avançadas"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Otimização:**")
                
                optimizer = st.selectbox(
                    "Otimizador",
                    ["adamw_torch", "adamw_hf", "sgd"],
                    help="Algoritmo de otimização"
                )
                
                scheduler = st.selectbox(
                    "Scheduler",
                    ["linear", "cosine", "cosine_with_restarts", "polynomial"],
                    index=1,
                    help="Scheduler da learning rate"
                )
                
                weight_decay = st.number_input(
                    "Weight Decay",
                    min_value=0.0,
                    max_value=0.1,
                    value=0.01,
                    step=0.001,
                    help="Regularização L2"
                )
            
            with col2:
                st.markdown("**Hardware:**")
                
                fp16 = st.checkbox(
                    "FP16",
                    value=True,
                    help="Usar precisão mista para economizar VRAM"
                )
                
                gradient_checkpointing = st.checkbox(
                    "Gradient Checkpointing",
                    value=True,
                    help="Economizar VRAM com checkpointing"
                )
                
                dataloader_workers = st.number_input(
                    "Workers do DataLoader",
                    min_value=0,
                    max_value=8,
                    value=4,
                    help="Número de workers para carregamento de dados"
                )
        
        # Configurações de salvamento
        with st.expander("💾 Configurações de Salvamento"):
            col1, col2 = st.columns(2)
            
            with col1:
                save_strategy = st.selectbox(
                    "Estratégia de Salvamento",
                    ["steps", "epoch", "no"],
                    help="Quando salvar checkpoints"
                )
                
                if save_strategy == "steps":
                    save_steps = st.number_input(
                        "Salvar a cada X passos",
                        min_value=10,
                        max_value=1000,
                        value=500,
                        help="Intervalo de passos para salvamento"
                    )
                else:
                    save_steps = None
            
            with col2:
                save_total_limit = st.number_input(
                    "Limite de Checkpoints",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="Máximo de checkpoints a manter"
                )
                
                load_best_model = st.checkbox(
                    "Carregar Melhor Modelo",
                    value=True,
                    help="Carregar o melhor modelo ao final"
                )
        
        # Configurações de logging
        with st.expander("📊 Configurações de Logging"):
            col1, col2 = st.columns(2)
            
            with col1:
                logging_steps = st.number_input(
                    "Log a cada X passos",
                    min_value=1,
                    max_value=100,
                    value=10,
                    help="Frequência de logging"
                )
                
                eval_strategy = st.selectbox(
                    "Estratégia de Avaliação",
                    ["steps", "epoch", "no"],
                    help="Quando avaliar o modelo"
                )
            
            with col2:
                if eval_strategy == "steps":
                    eval_steps = st.number_input(
                        "Avaliar a cada X passos",
                        min_value=10,
                        max_value=1000,
                        value=500,
                        help="Intervalo de passos para avaliação"
                    )
                else:
                    eval_steps = None
                
                report_to = st.multiselect(
                    "Relatórios",
                    ["tensorboard", "wandb", "none"],
                    default=["tensorboard"],
                    help="Onde enviar métricas de treinamento"
                )
        
        # Botão de submissão
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            submitted = st.form_submit_button(
                "🚀 Iniciar Treinamento",
                type="primary",
                use_container_width=True
            )
        
        if submitted:
            # Valida configurações
            if not job_name or len(job_name) < 3:
                st.error("❌ Nome do job deve ter pelo menos 3 caracteres")
                return
            
            # Cria configuração do job
            job_config = create_job_config(
                job_name, selected_model, selected_dataset, training_type,
                {
                    'r': lora_r if training_type in ["LoRA", "QLoRA"] else None,
                    'alpha': lora_alpha if training_type in ["LoRA", "QLoRA"] else None,
                    'dropout': lora_dropout if training_type in ["LoRA", "QLoRA"] else None,
                    'target_modules': target_modules if training_type in ["LoRA", "QLoRA"] else None,
                    'bias': bias if training_type in ["LoRA", "QLoRA"] else None,
                },
                {
                    'num_epochs': num_epochs,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'gradient_accumulation': gradient_accumulation,
                    'max_length': max_length,
                    'warmup_ratio': warmup_ratio,
                    'optimizer': optimizer,
                    'scheduler': scheduler,
                    'weight_decay': weight_decay,
                    'fp16': fp16,
                    'gradient_checkpointing': gradient_checkpointing,
                    'dataloader_workers': dataloader_workers,
                    'save_strategy': save_strategy,
                    'save_steps': save_steps,
                    'save_total_limit': save_total_limit,
                    'load_best_model': load_best_model,
                    'logging_steps': logging_steps,
                    'eval_strategy': eval_strategy,
                    'eval_steps': eval_steps,
                    'report_to': report_to
                }
            )
            
            # Inicia treinamento
            start_training(job_config)


def render_training_monitor():
    """Renderiza monitor de treinamento em andamento."""
    
    st.markdown("## 🔄 Treinamento em Andamento")
    
    # Status atual
    training_status = get_training_status()
    
    if training_status:
        # Informações do job
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Job Atual", training_status.get('current_job', 'N/A'))
        
        with col2:
            progress = training_status.get('progress', 0)
            st.metric("Progresso", f"{progress:.1f}%")
        
        with col3:
            st.metric("Status", "🔄 Treinando")
        
        # Barra de progresso
        st.progress(progress / 100)
        
        # Métricas em tempo real
        st.markdown("### 📊 Métricas em Tempo Real")
        
        metrics = training_status.get('metrics', {})
        
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'train_loss' in metrics:
                    st.metric("Loss de Treino", f"{metrics['train_loss']:.4f}")
            
            with col2:
                if 'eval_loss' in metrics:
                    st.metric("Loss de Validação", f"{metrics['eval_loss']:.4f}")
            
            with col3:
                if 'learning_rate' in metrics:
                    st.metric("Learning Rate", f"{metrics['learning_rate']:.2e}")
            
            with col4:
                if 'epoch' in metrics:
                    st.metric("Época", f"{metrics['epoch']:.2f}")
        
        # Logs recentes
        st.markdown("### 📝 Logs Recentes")
        
        logs = training_status.get('logs', [])
        
        if logs:
            # Mostra últimos 10 logs
            for log in logs[-10:]:
                timestamp = log.get('timestamp', '')
                message = log.get('message', '')
                
                if message:
                    st.text(f"[{timestamp}] {message}")
        else:
            st.info("Aguardando logs...")
        
        # Controles
        st.markdown("### 🎛️ Controles")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⏸️ Pausar Treinamento", use_container_width=True):
                pause_training()
        
        with col2:
            if st.button("⏹️ Parar Treinamento", type="secondary", use_container_width=True):
                stop_training()
        
        with col3:
            if st.button("🔄 Atualizar", use_container_width=True):
                st.rerun()


def render_active_jobs():
    """Renderiza lista de jobs ativos."""
    
    st.markdown("## 📋 Jobs Ativos")
    
    active_jobs = get_active_jobs()
    
    if not active_jobs:
        st.info("📭 Nenhum job ativo no momento")
    else:
        for job in active_jobs:
            render_job_card(job)


def render_training_history():
    """Renderiza histórico de treinamentos."""
    
    st.markdown("## 📊 Histórico de Treinamentos")
    
    # Filtros
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox(
            "Status",
            ["Todos", "Concluído", "Falhou", "Cancelado"]
        )
    
    with col2:
        model_filter = st.selectbox(
            "Modelo",
            ["Todos"] + [m['name'] for m in get_available_models()]
        )
    
    with col3:
        date_filter = st.selectbox(
            "Período",
            ["Todos", "Última semana", "Último mês", "Últimos 3 meses"]
        )
    
    # Lista de jobs históricos
    history_jobs = get_training_history()
    
    if not history_jobs:
        st.info("📭 Nenhum treinamento no histórico")
    else:
        for job in history_jobs:
            render_history_job_card(job)


def render_job_card(job: Dict[str, Any]):
    """Renderiza card de job ativo."""
    
    with st.container():
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.markdown(f"**🚀 {job['name']}**")
            st.caption(f"Modelo: {job['model']} | Dataset: {job['dataset']}")
        
        with col2:
            progress = job.get('progress', 0)
            st.metric("Progresso", f"{progress:.1f}%")
        
        with col3:
            st.metric("Status", job['status'])
        
        with col4:
            if st.button(f"👁️ Ver", key=f"view_{job['id']}", use_container_width=True):
                view_job_details(job)
        
        st.markdown("---")


def render_history_job_card(job: Dict[str, Any]):
    """Renderiza card de job histórico."""
    
    with st.container():
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.markdown(f"**📊 {job['name']}**")
            st.caption(f"Modelo: {job['model']} | Dataset: {job['dataset']}")
            st.caption(f"Concluído em: {job['completed']}")
        
        with col2:
            if 'final_loss' in job:
                st.metric("Loss Final", f"{job['final_loss']:.4f}")
        
        with col3:
            status_icon = {
                'Concluído': '✅',
                'Falhou': '❌',
                'Cancelado': '⏹️'
            }.get(job['status'], '❓')
            st.markdown(f"{status_icon} {job['status']}")
        
        with col4:
            if st.button(f"📁 Baixar", key=f"download_{job['id']}", use_container_width=True):
                download_job_results(job)
        
        st.markdown("---")


def get_available_models():
    """Retorna lista de modelos disponíveis."""
    return [
        {'name': 'LLaMA-7B-Chat', 'type': 'LLaMA'},
        {'name': 'Mistral-7B-Instruct', 'type': 'Mistral'},
        {'name': 'Custom-GPT2', 'type': 'GPT'}
    ]


def get_available_datasets():
    """Retorna lista de datasets disponíveis."""
    return [
        {'name': 'conversational_data', 'type': 'Conversação'},
        {'name': 'instruction_following', 'type': 'Instrução'},
        {'name': 'text_classification', 'type': 'Classificação'}
    ]


def check_training_in_progress():
    """Verifica se há treinamento em andamento."""
    return st.session_state.get('training_in_progress', False)


def get_training_status():
    """Retorna status do treinamento atual."""
    return st.session_state.get('training_status', {})


def get_active_jobs():
    """Retorna lista de jobs ativos."""
    return []  # Placeholder


def get_training_history():
    """Retorna histórico de treinamentos."""
    return [
        {
            'id': 'job_001',
            'name': 'LLaMA Chat Fine-tune',
            'model': 'LLaMA-7B-Chat',
            'dataset': 'conversational_data',
            'status': 'Concluído',
            'completed': '2024-01-15 14:30',
            'final_loss': 0.234
        },
        {
            'id': 'job_002',
            'name': 'Mistral Instruction',
            'model': 'Mistral-7B-Instruct',
            'dataset': 'instruction_following',
            'status': 'Falhou',
            'completed': '2024-01-14 09:15',
            'error': 'CUDA out of memory'
        }
    ]


def create_job_config(job_name: str, model: str, dataset: str, training_type: str,
                     lora_config: Dict[str, Any], training_config: Dict[str, Any]) -> Dict[str, Any]:
    """Cria configuração do job de treinamento."""
    
    config = {
        'job_name': job_name,
        'model_name': model,
        'dataset_name': dataset,
        'training_type': training_type,
        'lora': lora_config if training_type in ["LoRA", "QLoRA"] else {},
        'training': training_config,
        'created': datetime.now().isoformat()
    }
    
    return config


def start_training(job_config: Dict[str, Any]):
    """Inicia o treinamento."""
    
    try:
        with st.spinner("🚀 Iniciando treinamento..."):
            # Aqui seria chamado o TrainingManager
            # success, message, job_id = training_manager.create_training_job(job_config)
            
            # Simulação
            success = True
            message = "Treinamento iniciado com sucesso!"
            job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if success:
                st.success(f"✅ {message}")
                st.info(f"Job ID: {job_id}")
                
                # Atualiza estado
                st.session_state.training_in_progress = True
                st.session_state.training_status = {
                    'current_job': job_id,
                    'progress': 0,
                    'logs': [],
                    'metrics': {}
                }
                
                st.balloons()
                st.rerun()
            else:
                st.error(f"❌ {message}")
                
    except Exception as e:
        st.error(f"❌ Erro ao iniciar treinamento: {str(e)}")


def pause_training():
    """Pausa o treinamento atual."""
    st.warning("⏸️ Funcionalidade de pausar será implementada")


def stop_training():
    """Para o treinamento atual."""
    
    if st.button("⚠️ Confirmar parada do treinamento", type="secondary"):
        with st.spinner("⏹️ Parando treinamento..."):
            # Aqui seria chamado o TrainingManager
            st.session_state.training_in_progress = False
            st.success("✅ Treinamento parado!")
            st.rerun()


def view_job_details(job: Dict[str, Any]):
    """Mostra detalhes do job."""
    
    with st.expander(f"📋 Detalhes - {job['name']}", expanded=True):
        st.json(job)


def download_job_results(job: Dict[str, Any]):
    """Baixa resultados do job."""
    
    st.success(f"📁 Preparando download dos resultados de {job['name']}...")
    # Aqui seria implementado o download real

