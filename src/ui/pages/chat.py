#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuralTrain Forge - Página de Chat e Teste de Modelos
Interface para testar modelos treinados através de chat conversacional
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.core.chat_manager import ChatManager, ModelLoader
    from src.core.document_processor import DocumentProcessor
    from src.core.evaluation_system import ModelEvaluator, MetricsCalculator
    from src.utils.logging_utils import setup_logger
except ImportError as e:
    st.error(f"Erro ao importar módulos: {e}")
    st.stop()

# Configurar logger
logger = setup_logger(__name__)

def init_chat_manager():
    """Inicializa o gerenciador de chat e processador de documentos"""
    if 'chat_manager' not in st.session_state:
        try:
            st.session_state.chat_manager = ChatManager()
            logger.info("ChatManager inicializado com sucesso")
        except Exception as e:
            st.error(f"Erro ao inicializar ChatManager: {e}")
            st.stop()
    
    if 'document_processor' not in st.session_state:
        try:
            st.session_state.document_processor = DocumentProcessor()
            logger.info("DocumentProcessor inicializado com sucesso")
        except Exception as e:
            st.error(f"Erro ao inicializar DocumentProcessor: {e}")
            st.stop()
    
    if 'model_evaluator' not in st.session_state:
        try:
            st.session_state.model_evaluator = ModelEvaluator()
            logger.info("ModelEvaluator inicializado com sucesso")
        except Exception as e:
            st.error(f"Erro ao inicializar ModelEvaluator: {e}")
            st.stop()

def render_model_selector():
    """Renderiza seletor de modelos"""
    st.subheader("🤖 Seleção de Modelos")
    
    chat_manager = st.session_state.chat_manager
    
    # Listar modelos disponíveis
    available_models = chat_manager.model_loader.list_available_models()
    loaded_models = list(chat_manager.model_loader.loaded_models.keys())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Modelos Disponíveis:**")
        if available_models:
            for model in available_models:
                status = "🟢 Carregado" if model['name'] in loaded_models else "⚪ Disponível"
                st.write(f"- {model['name']} ({model['size']}) {status}")
        else:
            st.info("Nenhum modelo encontrado. Faça upload de modelos primeiro.")
    
    with col2:
        st.write("**Carregar Modelo:**")
        
        if available_models:
            model_to_load = st.selectbox(
                "Selecionar modelo:",
                options=[m['name'] for m in available_models if m['name'] not in loaded_models],
                key="model_to_load"
            )
            
            col2a, col2b = st.columns(2)
            with col2a:
                load_4bit = st.checkbox("Quantização 4-bit", help="Reduz uso de memória")
            with col2b:
                load_8bit = st.checkbox("Quantização 8-bit", help="Reduz uso de memória")
            
            if st.button("🚀 Carregar Modelo", key="load_model_btn"):
                if model_to_load:
                    with st.spinner(f"Carregando {model_to_load}..."):
                        model_info = next(m for m in available_models if m['name'] == model_to_load)
                        success = chat_manager.model_loader.load_model(
                            model_to_load,
                            model_info['path'] if model_info['type'] == 'local' else None,
                            load_in_8bit=load_8bit,
                            load_in_4bit=load_4bit
                        )
                        
                        if success:
                            st.success(f"✅ Modelo {model_to_load} carregado com sucesso!")
                            st.rerun()
                        else:
                            st.error(f"❌ Erro ao carregar modelo {model_to_load}")
        
        # Descarregar modelos
        if loaded_models:
            st.write("**Modelos Carregados:**")
            model_to_unload = st.selectbox(
                "Descarregar modelo:",
                options=loaded_models,
                key="model_to_unload"
            )
            
            if st.button("🗑️ Descarregar", key="unload_model_btn"):
                success = chat_manager.model_loader.unload_model(model_to_unload)
                if success:
                    st.success(f"✅ Modelo {model_to_unload} descarregado")
                    st.rerun()

def render_document_manager():
    """Renderiza gerenciador de documentos"""
    st.subheader("📄 Gerenciamento de Documentos")
    
    doc_processor = st.session_state.document_processor
    
    # Tabs para diferentes funcionalidades
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "📋 Documentos", "🔍 Busca"])
    
    with tab1:
        st.write("**Upload de Documentos**")
        
        # Mostrar formatos suportados
        supported_formats = doc_processor.get_supported_formats()
        st.info(f"Formatos suportados: {', '.join(supported_formats)} (máximo 20MB)")
        
        # Upload de arquivo
        uploaded_file = st.file_uploader(
            "Escolha um arquivo:",
            type=[fmt[1:] for fmt in supported_formats],  # Remove o ponto
            help="Selecione um documento para processar"
        )
        
        if uploaded_file is not None:
            # Salvar arquivo temporariamente
            temp_path = Path("data/temp") / uploaded_file.name
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Arquivo:** {uploaded_file.name}")
                st.write(f"**Tamanho:** {uploaded_file.size / 1024:.1f} KB")
                st.write(f"**Tipo:** {uploaded_file.type}")
            
            with col2:
                if st.button("🔄 Processar", key="process_doc"):
                    with st.spinner("Processando documento..."):
                        doc_info = doc_processor.process_document(temp_path)
                        
                        if doc_info:
                            st.success(f"✅ Documento processado!")
                            st.write(f"**Chunks criados:** {doc_info.total_chunks}")
                            st.write(f"**Tokens totais:** {doc_info.total_tokens}")
                            
                            # Limpar arquivo temporário
                            temp_path.unlink(missing_ok=True)
                            st.rerun()
                        else:
                            st.error("❌ Erro ao processar documento")
    
    with tab2:
        st.write("**Documentos Processados**")
        
        documents = doc_processor.list_documents()
        
        if documents:
            for doc in documents:
                with st.expander(f"📄 {doc['filename']}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Tipo:** {doc['file_type']}")
                        st.write(f"**Tamanho:** {doc['file_size'] / 1024:.1f} KB")
                        st.write(f"**Processado:** {doc['processed_at'][:19]}")
                    
                    with col2:
                        st.write(f"**Chunks:** {doc['total_chunks']}")
                        st.write(f"**Tokens:** {doc['total_tokens']}")
                        st.write(f"**Tokens/Chunk:** {doc['total_tokens'] // doc['total_chunks'] if doc['total_chunks'] > 0 else 0}")
                    
                    with col3:
                        if st.button(f"🗑️ Remover", key=f"delete_{doc['id']}"):
                            if doc_processor.delete_document(doc['id']):
                                st.success("Documento removido!")
                                st.rerun()
                        
                        if st.button(f"📊 Detalhes", key=f"details_{doc['id']}"):
                            st.session_state.selected_document = doc['id']
                            st.rerun()
        else:
            st.info("Nenhum documento processado ainda.")
    
    with tab3:
        st.write("**Busca em Documentos**")
        
        documents = doc_processor.list_documents()
        
        if documents:
            # Seletor de documento
            doc_options = {doc['filename']: doc['id'] for doc in documents}
            selected_doc_name = st.selectbox(
                "Documento:",
                options=list(doc_options.keys()),
                key="search_document"
            )
            
            if selected_doc_name:
                selected_doc_id = doc_options[selected_doc_name]
                
                # Campo de busca
                search_query = st.text_input(
                    "Buscar no documento:",
                    placeholder="Digite palavras-chave...",
                    key="search_query"
                )
                
                if search_query:
                    results = doc_processor.search_in_document(selected_doc_id, search_query)
                    
                    if results:
                        st.write(f"**{len(results)} resultado(s) encontrado(s):**")
                        
                        for i, chunk in enumerate(results):
                            with st.expander(f"Resultado {i+1} (Chunk {chunk.chunk_index})", expanded=False):
                                st.write(chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content)
                                st.caption(f"Tokens: {chunk.token_count} | Posição: {chunk.start_position}-{chunk.end_position}")
                    else:
                        st.info("Nenhum resultado encontrado.")
        else:
            st.info("Nenhum documento disponível para busca.")

def render_chat_interface():
    """Renderiza interface de chat"""
    st.subheader("💬 Chat com Modelos")
    
    chat_manager = st.session_state.chat_manager
    loaded_models = list(chat_manager.model_loader.loaded_models.keys())
    
    if not loaded_models:
        st.warning("⚠️ Nenhum modelo carregado. Carregue um modelo primeiro.")
        return
    
    # Configurações do chat
    with st.expander("⚙️ Configurações de Geração", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1, 
                                   help="Controla criatividade (maior = mais criativo)")
            max_tokens = st.slider("Max Tokens", 50, 2048, 512, 50,
                                 help="Máximo de tokens na resposta")
        
        with col2:
            top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.1,
                            help="Nucleus sampling")
            top_k = st.slider("Top-k", 1, 100, 50, 1,
                            help="Top-k sampling")
        
        with col3:
            chat_mode = st.radio(
                "Modo de Chat:",
                ["🤖 Modelo Único", "⚔️ Duelo de Modelos"],
                help="Escolha entre chat com um modelo ou comparação"
            )
    
    # Seleção de modelos
    if chat_mode == "🤖 Modelo Único":
        selected_model = st.selectbox("Modelo:", loaded_models, key="single_model")
        models_to_use = [selected_model]
    else:
        models_to_use = st.multiselect(
            "Modelos para comparar:",
            loaded_models,
            default=loaded_models[:2] if len(loaded_models) >= 2 else loaded_models,
            key="comparison_models"
        )
        
        if len(models_to_use) < 2:
            st.warning("⚠️ Selecione pelo menos 2 modelos para comparação")
            return
    
    # Gerenciamento de sessões
    st.write("**Sessões de Chat:**")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        session_name = st.text_input("Nome da nova sessão:", 
                                   value=f"Chat_{datetime.now().strftime('%Y%m%d_%H%M')}")
    
    with col2:
        if st.button("➕ Nova Sessão"):
            if session_name:
                session_type = "comparison" if chat_mode == "⚔️ Duelo de Modelos" else "single"
                session_id = chat_manager.create_session(session_name, session_type)
                st.session_state.current_session = session_id
                st.success(f"✅ Sessão '{session_name}' criada!")
                st.rerun()
    
    with col3:
        sessions = chat_manager.list_sessions()
        if sessions:
            session_options = {f"{s['name']} ({s['id'][:8]})": s['id'] for s in sessions}
            selected_session_key = st.selectbox("Carregar sessão:", 
                                               options=list(session_options.keys()),
                                               key="session_selector")
            if selected_session_key:
                st.session_state.current_session = session_options[selected_session_key]
    
    # Interface de chat
    if 'current_session' in st.session_state:
        session_id = st.session_state.current_session
        session_data = chat_manager.get_session(session_id)
        
        if session_data:
            st.write(f"**Sessão Ativa:** {session_data['name']}")
            
            # Histórico de mensagens
            st.write("**Histórico:**")
            chat_container = st.container()
            
            with chat_container:
                for message in session_data['messages']:
                    if message['role'] == 'user':
                        st.chat_message("user").write(message['content'])
                    elif message['role'] == 'assistant':
                        with st.chat_message("assistant"):
                            st.write(f"**{message['model_name']}:**")
                            st.write(message['content'])
                            
                            # Mostrar métricas se disponíveis
                            if message.get('metadata'):
                                with st.expander("📊 Métricas", expanded=False):
                                    metrics = message['metadata']
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Tokens/seg", f"{metrics.get('tokens_per_second', 0):.1f}")
                                    with col2:
                                        st.metric("Tempo (s)", f"{metrics.get('generation_time', 0):.2f}")
                                    with col3:
                                        st.metric("Tokens", metrics.get('tokens_generated', 0))
            
            # Input de nova mensagem
            user_input = st.chat_input("Digite sua mensagem...")
            
            if user_input:
                # Mostrar mensagem do usuário imediatamente
                st.chat_message("user").write(user_input)
                
                # Gerar respostas
                with st.spinner("🤔 Gerando resposta..."):
                    try:
                        if chat_mode == "🤖 Modelo Único":
                            result = chat_manager.send_message(
                                session_id=session_id,
                                content=user_input,
                                model_name=models_to_use[0],
                                temperature=temperature,
                                max_tokens=max_tokens
                            )
                            
                            # Mostrar resposta
                            with st.chat_message("assistant"):
                                st.write(f"**{models_to_use[0]}:**")
                                st.write(result['message']['content'])
                                
                                with st.expander("📊 Métricas", expanded=False):
                                    metrics = result['metrics']
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Tokens/seg", f"{metrics['tokens_per_second']:.1f}")
                                    with col2:
                                        st.metric("Tempo (s)", f"{metrics['generation_time']:.2f}")
                                    with col3:
                                        st.metric("Tokens", metrics['tokens_generated'])
                        
                        else:  # Duelo de modelos
                            result = chat_manager.compare_models(
                                session_id=session_id,
                                content=user_input,
                                model_names=models_to_use,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                top_p=top_p,
                                top_k=top_k
                            )
                            
                            # Mostrar respostas lado a lado
                            cols = st.columns(len(models_to_use))
                            
                            for i, model_name in enumerate(models_to_use):
                                with cols[i]:
                                    with st.chat_message("assistant"):
                                        st.write(f"**{model_name}:**")
                                        
                                        if 'error' in result['responses'][model_name]:
                                            st.error(f"Erro: {result['responses'][model_name]['error']}")
                                        else:
                                            response_data = result['responses'][model_name]
                                            st.write(response_data['message']['content'])
                                            
                                            with st.expander("📊 Métricas", expanded=False):
                                                metrics = response_data['metrics']
                                                st.metric("Tokens/seg", f"{metrics['tokens_per_second']:.1f}")
                                                st.metric("Tempo (s)", f"{metrics['generation_time']:.2f}")
                                                st.metric("Tokens", metrics['tokens_generated'])
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Erro ao gerar resposta: {e}")
            
            # Ações da sessão
            st.write("**Ações:**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔄 Reiniciar Sessão"):
                    # Criar nova sessão com mesmo nome
                    new_session_id = chat_manager.create_session(
                        f"{session_data['name']}_novo",
                        session_data['session_type']
                    )
                    st.session_state.current_session = new_session_id
                    st.rerun()
            
            with col2:
                if st.button("💾 Exportar JSON"):
                    filepath = chat_manager.export_session(session_id, "json")
                    if filepath:
                        st.success(f"✅ Exportado para: {filepath}")
            
            with col3:
                if st.button("📄 Exportar JSONL"):
                    filepath = chat_manager.export_session(session_id, "jsonl")
                    if filepath:
                        st.success(f"✅ Exportado para: {filepath}")
            
            with col4:
                if st.button("🗑️ Deletar Sessão", type="secondary"):
                    if st.session_state.get('confirm_delete'):
                        chat_manager.delete_session(session_id)
                        if 'current_session' in st.session_state:
                            del st.session_state.current_session
                        st.success("✅ Sessão deletada!")
                        st.rerun()
                    else:
                        st.session_state.confirm_delete = True
                        st.warning("⚠️ Clique novamente para confirmar")

def render_metrics_dashboard():
    """Renderiza dashboard de métricas"""
    st.subheader("📊 Dashboard de Métricas")
    
    chat_manager = st.session_state.chat_manager
    metrics_summary = chat_manager.get_metrics_summary()
    
    if not metrics_summary:
        st.info("📈 Nenhuma métrica disponível. Inicie um chat para gerar dados.")
        return
    
    # Métricas gerais
    col1, col2, col3, col4 = st.columns(4)
    
    total_generations = sum(data['total_generations'] for data in metrics_summary.values())
    total_tokens = sum(data['total_tokens'] for data in metrics_summary.values())
    avg_speed = sum(data['avg_tokens_per_second'] for data in metrics_summary.values()) / len(metrics_summary)
    
    with col1:
        st.metric("Total de Gerações", total_generations)
    with col2:
        st.metric("Total de Tokens", f"{total_tokens:,}")
    with col3:
        st.metric("Velocidade Média", f"{avg_speed:.1f} tok/s")
    with col4:
        st.metric("Modelos Ativos", len(metrics_summary))
    
    # Gráficos comparativos
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de velocidade por modelo
        models = list(metrics_summary.keys())
        speeds = [data['avg_tokens_per_second'] for data in metrics_summary.values()]
        
        fig_speed = go.Figure(data=[
            go.Bar(x=models, y=speeds, name="Tokens/segundo")
        ])
        fig_speed.update_layout(
            title="Velocidade de Geração por Modelo",
            xaxis_title="Modelo",
            yaxis_title="Tokens por Segundo"
        )
        st.plotly_chart(fig_speed, use_container_width=True)
    
    with col2:
        # Gráfico de uso por modelo
        generations = [data['total_generations'] for data in metrics_summary.values()]
        
        fig_usage = go.Figure(data=[
            go.Pie(labels=models, values=generations, name="Uso")
        ])
        fig_usage.update_layout(title="Distribuição de Uso por Modelo")
        st.plotly_chart(fig_usage, use_container_width=True)
    
    # Tabela detalhada
    st.write("**Métricas Detalhadas:**")
    
    metrics_df = pd.DataFrame([
        {
            'Modelo': model,
            'Gerações': data['total_generations'],
            'Tokens Totais': data['total_tokens'],
            'Velocidade Média (tok/s)': f"{data['avg_tokens_per_second']:.1f}",
            'Tempo Médio (s)': f"{data['avg_generation_time']:.2f}",
            'Tamanho Médio Resposta': f"{data['avg_response_length']:.0f}"
        }
        for model, data in metrics_summary.items()
    ])
    
    st.dataframe(metrics_df, use_container_width=True)

def render_performance_monitor():
    """Renderiza monitor de performance em tempo real"""
    st.subheader("⚡ Monitor de Performance")
    
    # Placeholder para métricas em tempo real
    metrics_placeholder = st.empty()
    
    # Simular métricas em tempo real (seria conectado ao sistema real)
    import psutil
    import torch
    
    col1, col2, col3, col4 = st.columns(4)
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    with col1:
        st.metric("CPU", f"{cpu_percent:.1f}%")
    
    # RAM
    memory = psutil.virtual_memory()
    with col2:
        st.metric("RAM", f"{memory.percent:.1f}%", 
                 f"{memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB")
    
    # GPU (se disponível)
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_percent = (gpu_memory / gpu_total) * 100
        
        with col3:
            st.metric("GPU Memory", f"{gpu_percent:.1f}%",
                     f"{gpu_memory:.1f}GB / {gpu_total:.1f}GB")
        with col4:
            st.metric("GPU Temp", "N/A")  # Seria obtido do sistema
    else:
        with col3:
            st.metric("GPU", "Não disponível")
        with col4:
            st.metric("GPU Temp", "N/A")
    
    # Gráfico de uso ao longo do tempo (simulado)
    if 'performance_history' not in st.session_state:
        st.session_state.performance_history = []
    
    # Adicionar ponto atual
    current_time = datetime.now()
    st.session_state.performance_history.append({
        'time': current_time,
        'cpu': cpu_percent,
        'memory': memory.percent,
        'gpu': gpu_percent if torch.cuda.is_available() else 0
    })
    
    # Manter apenas últimos 50 pontos
    if len(st.session_state.performance_history) > 50:
        st.session_state.performance_history = st.session_state.performance_history[-50:]
    
    # Criar gráfico
    if len(st.session_state.performance_history) > 1:
        df = pd.DataFrame(st.session_state.performance_history)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['time'], y=df['cpu'], name='CPU %', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['time'], y=df['memory'], name='RAM %', line=dict(color='green')))
        if torch.cuda.is_available():
            fig.add_trace(go.Scatter(x=df['time'], y=df['gpu'], name='GPU %', line=dict(color='red')))
        
        fig.update_layout(
            title="Performance em Tempo Real",
            xaxis_title="Tempo",
            yaxis_title="Uso (%)",
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_evaluation_interface():
    """Renderiza interface de avaliação avançada"""
    st.subheader("🎯 Avaliação Avançada de Modelos")
    
    evaluator = st.session_state.model_evaluator
    
    # Tabs para diferentes funcionalidades de avaliação
    eval_tab1, eval_tab2, eval_tab3, eval_tab4 = st.tabs([
        "📝 Avaliação Manual", 
        "⚔️ Comparação", 
        "📊 Relatórios", 
        "🏆 Rankings"
    ])
    
    with eval_tab1:
        st.write("**Avaliação Manual de Respostas**")
        
        # Seletor de modelo
        chat_manager = st.session_state.chat_manager
        loaded_models = list(chat_manager.model_loader.loaded_models.keys())
        
        if not loaded_models:
            st.warning("⚠️ Nenhum modelo carregado. Carregue um modelo primeiro.")
            return
        
        selected_model = st.selectbox("Modelo para avaliar:", loaded_models)
        
        col1, col2 = st.columns(2)
        
        with col1:
            prompt = st.text_area(
                "Prompt:",
                placeholder="Digite o prompt para o modelo...",
                height=100
            )
            
            reference = st.text_area(
                "Resposta de Referência:",
                placeholder="Digite a resposta ideal/esperada...",
                help="Resposta que você considera ideal para comparação",
                height=100
            )
        
        with col2:
            generated = st.text_area(
                "Resposta Gerada:",
                placeholder="Cole aqui a resposta do modelo...",
                height=200
            )
        
        if st.button("🔍 Avaliar Resposta", key="evaluate_response"):
            if prompt and reference and generated:
                with st.spinner("Calculando métricas..."):
                    result = evaluator.evaluate_response(
                        model_name=selected_model,
                        prompt=prompt,
                        reference=reference,
                        generated=generated
                    )
                    
                    st.success("✅ Avaliação concluída!")
                    
                    # Mostrar métricas principais
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        bleu_score = result.metrics.get('bleu_avg', 0)
                        st.metric("BLEU Score", f"{bleu_score:.3f}")
                    
                    with col2:
                        rouge_score = result.metrics.get('rouge1_fmeasure', 0)
                        st.metric("ROUGE-1 F1", f"{rouge_score:.3f}")
                    
                    with col3:
                        semantic_sim = result.metrics.get('semantic_similarity', 0)
                        st.metric("Similaridade Semântica", f"{semantic_sim:.3f}")
                    
                    with col4:
                        coherence = result.metrics.get('coherence_score', 0)
                        st.metric("Coerência", f"{coherence:.3f}")
                    
                    # Métricas detalhadas
                    with st.expander("📊 Métricas Detalhadas", expanded=False):
                        metrics_df = pd.DataFrame([
                            {"Métrica": k, "Valor": f"{v:.4f}" if isinstance(v, float) else str(v)}
                            for k, v in result.metrics.items()
                            if not k.endswith('_error')
                        ])
                        st.dataframe(metrics_df, use_container_width=True)
            else:
                st.error("❌ Preencha todos os campos para avaliar")
    
    with eval_tab2:
        st.write("**Comparação Entre Modelos**")
        
        if len(loaded_models) < 2:
            st.warning("⚠️ Carregue pelo menos 2 modelos para comparação.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_a = st.selectbox("Modelo A:", loaded_models, key="model_a")
        
        with col2:
            model_b = st.selectbox("Modelo B:", 
                                 [m for m in loaded_models if m != model_a], 
                                 key="model_b")
        
        prompt_comp = st.text_area(
            "Prompt para Comparação:",
            placeholder="Digite o prompt que ambos os modelos devem responder...",
            height=100
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Resposta do {model_a}:**")
            response_a = st.text_area(
                f"Resposta {model_a}:",
                placeholder=f"Cole a resposta do {model_a}...",
                height=150,
                label_visibility="collapsed"
            )
        
        with col2:
            st.write(f"**Resposta do {model_b}:**")
            response_b = st.text_area(
                f"Resposta {model_b}:",
                placeholder=f"Cole a resposta do {model_b}...",
                height=150,
                label_visibility="collapsed"
            )
        
        reference_comp = st.text_area(
            "Resposta de Referência (Opcional):",
            placeholder="Resposta ideal para comparação (opcional)...",
            height=80
        )
        
        if st.button("⚔️ Comparar Modelos", key="compare_models"):
            if prompt_comp and response_a and response_b:
                with st.spinner("Comparando modelos..."):
                    result = evaluator.compare_models(
                        model_a=model_a,
                        model_b=model_b,
                        prompt=prompt_comp,
                        response_a=response_a,
                        response_b=response_b,
                        reference=reference_comp if reference_comp else None
                    )
                    
                    st.success("✅ Comparação concluída!")
                    
                    # Resultado da comparação
                    if result.winner:
                        winner_name = model_a if result.winner == "model_a" else model_b
                        st.success(f"🏆 **Vencedor:** {winner_name} (Confiança: {result.confidence:.3f})")
                    else:
                        st.info("🤝 **Resultado:** Empate técnico")
                    
                    # Métricas lado a lado
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Métricas - {model_a}:**")
                        metrics_a_df = pd.DataFrame([
                            {"Métrica": k, "Valor": f"{v:.4f}" if isinstance(v, float) else str(v)}
                            for k, v in result.metrics_a.items()
                            if not k.endswith('_error')
                        ])
                        st.dataframe(metrics_a_df, use_container_width=True)
                    
                    with col2:
                        st.write(f"**Métricas - {model_b}:**")
                        metrics_b_df = pd.DataFrame([
                            {"Métrica": k, "Valor": f"{v:.4f}" if isinstance(v, float) else str(v)}
                            for k, v in result.metrics_b.items()
                            if not k.endswith('_error')
                        ])
                        st.dataframe(metrics_b_df, use_container_width=True)
            else:
                st.error("❌ Preencha pelo menos prompt e ambas as respostas")
    
    with eval_tab3:
        st.write("**Relatórios de Avaliação**")
        
        # Seletor de modelo para relatório
        all_models = list(set(r.model_name for r in evaluator.evaluation_history))
        
        if not all_models:
            st.info("📝 Nenhuma avaliação realizada ainda.")
            return
        
        report_model = st.selectbox(
            "Modelo para relatório:",
            ["Todos os Modelos"] + all_models,
            key="report_model"
        )
        
        model_filter = None if report_model == "Todos os Modelos" else report_model
        
        if st.button("📊 Gerar Relatório", key="generate_report"):
            with st.spinner("Gerando relatório..."):
                report = evaluator.generate_evaluation_report(model_filter)
                
                if "error" in report:
                    st.error(f"❌ {report['error']}")
                    return
                
                # Informações gerais
                st.write("### 📋 Informações Gerais")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total de Avaliações", report['total_evaluations'])
                
                with col2:
                    if report['date_range']['start']:
                        start_date = pd.to_datetime(report['date_range']['start']).strftime('%d/%m/%Y')
                        st.metric("Primeira Avaliação", start_date)
                
                with col3:
                    if report['date_range']['end']:
                        end_date = pd.to_datetime(report['date_range']['end']).strftime('%d/%m/%Y')
                        st.metric("Última Avaliação", end_date)
                
                # Estatísticas de métricas
                if report['metric_statistics']:
                    st.write("### 📊 Estatísticas de Métricas")
                    
                    metrics_data = []
                    for metric, stats in report['metric_statistics'].items():
                        metrics_data.append({
                            'Métrica': metric,
                            'Média': f"{stats['mean']:.4f}",
                            'Desvio Padrão': f"{stats['std']:.4f}",
                            'Mínimo': f"{stats['min']:.4f}",
                            'Máximo': f"{stats['max']:.4f}",
                            'Contagem': stats['count']
                        })
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df, use_container_width=True)
                
                # Performance recente
                if 'recent_performance' in report and 'average_metrics' in report['recent_performance']:
                    st.write("### 📈 Performance Recente (7 dias)")
                    recent_data = []
                    for metric, value in report['recent_performance']['average_metrics'].items():
                        recent_data.append({
                            'Métrica': metric,
                            'Média Recente': f"{value:.4f}"
                        })
                    
                    if recent_data:
                        recent_df = pd.DataFrame(recent_data)
                        st.dataframe(recent_df, use_container_width=True)
                
                # Recomendações
                if report['recommendations']:
                    st.write("### 💡 Recomendações")
                    for i, rec in enumerate(report['recommendations'], 1):
                        st.write(f"{i}. {rec}")
        
        # Exportar resultados
        st.write("### 📤 Exportar Resultados")
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox("Formato:", ["JSON", "CSV"])
        
        with col2:
            export_model = st.selectbox(
                "Modelo:",
                ["Todos"] + all_models,
                key="export_model"
            )
        
        if st.button("💾 Exportar", key="export_results"):
            model_filter = None if export_model == "Todos" else export_model
            
            try:
                filepath = evaluator.export_results(
                    format=export_format.lower(),
                    model_name=model_filter
                )
                st.success(f"✅ Resultados exportados para: {filepath}")
                
                # Oferecer download
                with open(filepath, 'rb') as f:
                    st.download_button(
                        label="📥 Download",
                        data=f.read(),
                        file_name=Path(filepath).name,
                        mime="application/json" if export_format == "JSON" else "text/csv"
                    )
            except Exception as e:
                st.error(f"❌ Erro ao exportar: {e}")
    
    with eval_tab4:
        st.write("**Rankings de Modelos**")
        
        if st.button("🏆 Atualizar Rankings", key="update_rankings"):
            with st.spinner("Calculando rankings..."):
                rankings = evaluator.get_model_rankings()
                
                if "message" in rankings:
                    st.info(f"📝 {rankings['message']}")
                    return
                
                st.write("### 🏆 Ranking de Modelos")
                
                ranking_data = []
                for i, (model, stats) in enumerate(rankings['rankings'], 1):
                    ranking_data.append({
                        'Posição': i,
                        'Modelo': model,
                        'Score Médio': f"{stats['average_score']:.4f}",
                        'Desvio Padrão': f"{stats['std_score']:.4f}",
                        'Avaliações': stats['evaluation_count']
                    })
                
                ranking_df = pd.DataFrame(ranking_data)
                st.dataframe(ranking_df, use_container_width=True)
                
                # Gráfico de rankings
                if len(ranking_data) > 1:
                    fig = go.Figure(data=go.Bar(
                        x=[r['Modelo'] for r in ranking_data],
                        y=[float(r['Score Médio']) for r in ranking_data],
                        text=[r['Score Médio'] for r in ranking_data],
                        textposition='auto',
                    ))
                    
                    fig.update_layout(
                        title="Ranking de Modelos por Score Médio",
                        xaxis_title="Modelo",
                        yaxis_title="Score Médio",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Estatísticas gerais
        stats = evaluator.get_statistics()
        
        st.write("### 📊 Estatísticas Gerais")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Avaliações", stats['total_evaluations'])
        
        with col2:
            st.metric("Total Comparações", stats['total_comparisons'])
        
        with col3:
            st.metric("Modelos Únicos", stats['unique_models'])
        
        with col4:
            st.metric("Métricas Disponíveis", len(stats['available_metrics']))
        
        # Status dos componentes
        st.write("### 🔧 Status dos Componentes")
        components = stats['components_status']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = "✅ Ativo" if components['nltk'] else "❌ Inativo"
            st.write(f"**NLTK (BLEU):** {status}")
        
        with col2:
            status = "✅ Ativo" if components['rouge'] else "❌ Inativo"
            st.write(f"**ROUGE:** {status}")
        
        with col3:
            status = "✅ Ativo" if components['sentence_transformers'] else "❌ Inativo"
            st.write(f"**Sentence Transformers:** {status}")

def main():
    """Função principal da página"""
    st.set_page_config(
        page_title="NeuralTrain Forge - Chat & Teste",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("💬 Chat & Teste de Modelos")
    st.markdown("---")
    
    # Inicializar chat manager
    init_chat_manager()
    
    # Tabs principais
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🤖 Modelos", 
        "📄 Documentos",
        "💬 Chat", 
        "🎯 Avaliação",
        "📊 Métricas", 
        "⚡ Performance"
    ])
    
    with tab1:
        render_model_selector()
    
    with tab2:
        render_document_manager()
    
    with tab3:
        render_chat_interface()
    
    with tab4:
        render_evaluation_interface()
    
    with tab5:
        render_metrics_dashboard()
    
    with tab6:
        render_performance_monitor()

if __name__ == "__main__":
    main()

