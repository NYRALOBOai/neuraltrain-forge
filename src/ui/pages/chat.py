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
    st.subheader("🧠 Seleção de Modelos")
    
    chat_manager = st.session_state.chat_manager
    
    # Listar modelos disponíveis
    available_models = chat_manager.model_loader.list_available_models()
    loaded_models = list(chat_manager.model_loader.loaded_models.keys())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("***Modelos Disponíveis:***")
        if available_models:
            for model in available_models:
                status = "🟢 Carregado" if model['name'] in loaded_models else "🔴 Disponível"
                st.write(f"- {model['name']} ({model['size']}) {status}")
        else:
            st.info("Nenhum modelo encontrado. Faça upload de modelos primeiro.")
    
    with col2:
        st.write("***Carregar Modelo:***")
        
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
            st.write("***Modelos Carregados:***")
            model_to_unload = st.selectbox(
                "Descarregar modelo:",
                options=loaded_models,
                key="model_to_unload"
            )
            
            if st.button("🗑️ Descarregar", key="unload_model_btn"):
                if model_to_unload:
                    success = chat_manager.model_loader.unload_model(model_to_unload)
                    if success:
                        st.success(f"✅ Modelo {model_to_unload} descarregado!")
                        st.rerun()
                    else:
                        st.error(f"❌ Erro ao descarregar modelo")

def render_document_manager():
    """Renderiza gerenciador de documentos"""
    st.subheader("📄 Gerenciamento de Documentos")
    
    doc_processor = st.session_state.document_processor
    
    # Tabs para diferentes funcionalidades
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "📋 Lista", "🔍 Busca"])
    
    with tab1:
        st.write("**Upload de Documentos**")
        uploaded_files = st.file_uploader(
            "Selecione documentos",
            type=['txt', 'pdf', 'md', 'json', 'csv'],
            accept_multiple_files=True,
            help="Formatos suportados: TXT, PDF, MD, JSON, CSV (sem limite de tamanho)"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if st.button(f"📤 Processar {uploaded_file.name}", key=f"process_{uploaded_file.name}"):
                    with st.spinner(f"Processando {uploaded_file.name}..."):
                        try:
                            # Salvar arquivo temporariamente
                            temp_path = f"/tmp/{uploaded_file.name}"
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Processar documento
                            doc_id = doc_processor.process_document(temp_path)
                            
                            if doc_id:
                                st.success(f"✅ Documento {uploaded_file.name} processado com sucesso!")
                                st.info(f"ID do documento: {doc_id}")
                            else:
                                st.error(f"❌ Erro ao processar {uploaded_file.name}")
                                
                        except Exception as e:
                            st.error(f"❌ Erro: {str(e)}")
    
    with tab2:
        st.write("**Documentos Processados**")
        documents = doc_processor.list_documents()
        
        if documents:
            for doc in documents:
                with st.expander(f"📄 {doc['filename']} ({doc['size']} tokens)"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Tipo:** {doc['type']}")
                        st.write(f"**Processado em:** {doc['processed_at']}")
                        st.write(f"**Chunks:** {doc['chunks']}")
                        if doc['metadata']:
                            st.write("**Metadados:**")
                            st.json(doc['metadata'])
                    
                    with col2:
                        if st.button("🗑️ Remover", key=f"remove_{doc['id']}"):
                            if doc_processor.remove_document(doc['id']):
                                st.success("Documento removido!")
                                st.rerun()
        else:
            st.info("Nenhum documento processado ainda.")
    
    with tab3:
        st.write("**Busca em Documentos**")
        search_query = st.text_input("Digite sua busca:", key="doc_search")
        
        if search_query:
            results = doc_processor.search_documents(search_query)
            
            if results:
                st.write(f"**Encontrados {len(results)} resultados:**")
                for i, result in enumerate(results):
                    with st.expander(f"Resultado {i+1} - {result['filename']} (Score: {result['score']:.2f})"):
                        st.write(result['content'])
            else:
                st.info("Nenhum resultado encontrado.")

def render_chat_interface():
    """Renderiza interface de chat"""
    st.subheader("💬 Chat com Modelos")
    
    chat_manager = st.session_state.chat_manager
    loaded_models = list(chat_manager.model_loader.loaded_models.keys())
    
    if not loaded_models:
        st.warning("⚠️ Nenhum modelo carregado. Carregue um modelo primeiro.")
        return
    
    # Configurações de chat
    with st.expander("⚙️ Configurações de Geração"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
            top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
        
        with col2:
            top_k = st.slider("Top-k", 1, 100, 50, 1)
            max_tokens = st.slider("Max Tokens", 50, 2048, 512, 50)
        
        with col3:
            repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.1, 0.1)
    
    # Seleção de modo
    chat_mode = st.radio(
        "Modo de Chat:",
        ["🤖 Modelo Único", "⚔️ Duelo de Modelos"],
        horizontal=True
    )
    
    if chat_mode == "🤖 Modelo Único":
        render_single_chat(loaded_models, temperature, top_p, top_k, max_tokens, repetition_penalty)
    else:
        render_dual_chat(loaded_models, temperature, top_p, top_k, max_tokens, repetition_penalty)

def render_single_chat(loaded_models, temperature, top_p, top_k, max_tokens, repetition_penalty):
    """Renderiza chat com modelo único"""
    
    # Seleção de modelo
    selected_model = st.selectbox("Selecione o modelo:", loaded_models, key="single_model")
    
    # Histórico de chat
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Exibir histórico
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message['role'] == 'user':
                st.chat_message("user").write(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.write(f"**{message['model']}:**")
                    st.write(message['content'])
                    if 'metrics' in message:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Tokens/s", f"{message['metrics']['tokens_per_second']:.1f}")
                        with col2:
                            st.metric("Tempo", f"{message['metrics']['response_time']:.1f}s")
                        with col3:
                            st.metric("Tokens", message['metrics']['total_tokens'])
    
    # Input de mensagem
    user_input = st.chat_input("Digite sua mensagem...")
    
    if user_input:
        # Adicionar mensagem do usuário
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        })
        
        # Gerar resposta
        with st.spinner("Gerando resposta..."):
            start_time = time.time()
            
            response = st.session_state.chat_manager.generate_response(
                selected_model,
                user_input,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response:
                # Calcular métricas
                tokens_per_second = len(response.split()) / response_time if response_time > 0 else 0
                
                # Adicionar resposta ao histórico
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response,
                    'model': selected_model,
                    'timestamp': datetime.now(),
                    'metrics': {
                        'response_time': response_time,
                        'tokens_per_second': tokens_per_second,
                        'total_tokens': len(response.split())
                    }
                })
                
                st.rerun()
            else:
                st.error("❌ Erro ao gerar resposta")
    
    # Controles do chat
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ Limpar Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("💾 Salvar Sessão"):
            save_chat_session()
    
    with col3:
        if st.button("📊 Ver Métricas"):
            show_chat_metrics()

def render_dual_chat(loaded_models, temperature, top_p, top_k, max_tokens, repetition_penalty):
    """Renderiza chat com duelo de modelos"""
    
    if len(loaded_models) < 2:
        st.warning("⚠️ Carregue pelo menos 2 modelos para o modo duelo.")
        return
    
    # Seleção de modelos
    col1, col2 = st.columns(2)
    with col1:
        model1 = st.selectbox("Modelo 1:", loaded_models, key="dual_model1")
    with col2:
        model2 = st.selectbox("Modelo 2:", [m for m in loaded_models if m != model1], key="dual_model2")
    
    # Histórico de duelo
    if 'dual_history' not in st.session_state:
        st.session_state.dual_history = []
    
    # Exibir histórico
    for i, exchange in enumerate(st.session_state.dual_history):
        st.chat_message("user").write(exchange['user_input'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.chat_message("assistant"):
                st.write(f"**{exchange['model1']}:**")
                st.write(exchange['response1'])
                if 'metrics1' in exchange:
                    st.caption(f"⏱️ {exchange['metrics1']['response_time']:.1f}s | 🚀 {exchange['metrics1']['tokens_per_second']:.1f} t/s")
        
        with col2:
            with st.chat_message("assistant"):
                st.write(f"**{exchange['model2']}:**")
                st.write(exchange['response2'])
                if 'metrics2' in exchange:
                    st.caption(f"⏱️ {exchange['metrics2']['response_time']:.1f}s | 🚀 {exchange['metrics2']['tokens_per_second']:.1f} t/s")
        
        # Avaliação comparativa
        if 'evaluation' in exchange:
            st.info(f"🏆 Vencedor: {exchange['evaluation']['winner']} (Confiança: {exchange['evaluation']['confidence']:.1%})")
    
    # Input de mensagem
    user_input = st.chat_input("Digite sua mensagem para ambos os modelos...")
    
    if user_input:
        with st.spinner("Gerando respostas..."):
            # Gerar respostas de ambos os modelos
            start_time1 = time.time()
            response1 = st.session_state.chat_manager.generate_response(
                model1, user_input, temperature=temperature, top_p=top_p, 
                top_k=top_k, max_tokens=max_tokens, repetition_penalty=repetition_penalty
            )
            end_time1 = time.time()
            
            start_time2 = time.time()
            response2 = st.session_state.chat_manager.generate_response(
                model2, user_input, temperature=temperature, top_p=top_p,
                top_k=top_k, max_tokens=max_tokens, repetition_penalty=repetition_penalty
            )
            end_time2 = time.time()
            
            if response1 and response2:
                # Calcular métricas
                metrics1 = {
                    'response_time': end_time1 - start_time1,
                    'tokens_per_second': len(response1.split()) / (end_time1 - start_time1) if (end_time1 - start_time1) > 0 else 0,
                    'total_tokens': len(response1.split())
                }
                
                metrics2 = {
                    'response_time': end_time2 - start_time2,
                    'tokens_per_second': len(response2.split()) / (end_time2 - start_time2) if (end_time2 - start_time2) > 0 else 0,
                    'total_tokens': len(response2.split())
                }
                
                # Avaliação automática
                evaluation = st.session_state.model_evaluator.compare_responses(
                    user_input, response1, response2, model1, model2
                )
                
                # Adicionar ao histórico
                st.session_state.dual_history.append({
                    'user_input': user_input,
                    'model1': model1,
                    'model2': model2,
                    'response1': response1,
                    'response2': response2,
                    'metrics1': metrics1,
                    'metrics2': metrics2,
                    'evaluation': evaluation,
                    'timestamp': datetime.now()
                })
                
                st.rerun()
            else:
                st.error("❌ Erro ao gerar uma ou ambas as respostas")
    
    # Controles do duelo
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ Limpar Duelo"):
            st.session_state.dual_history = []
            st.rerun()
    
    with col2:
        if st.button("💾 Salvar Duelo"):
            save_dual_session()
    
    with col3:
        if st.button("📊 Comparar Modelos"):
            show_model_comparison()

def save_chat_session():
    """Salva sessão de chat"""
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_session_{timestamp}.json"
        
        session_data = {
            'timestamp': timestamp,
            'messages': st.session_state.chat_history,
            'type': 'single_chat'
        }
        
        # Salvar em data/sessions/
        sessions_dir = Path("data/sessions")
        sessions_dir.mkdir(parents=True, exist_ok=True)
        
        with open(sessions_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2, default=str)
        
        st.success(f"✅ Sessão salva como {filename}")

def save_dual_session():
    """Salva sessão de duelo"""
    if 'dual_history' in st.session_state and st.session_state.dual_history:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dual_session_{timestamp}.json"
        
        session_data = {
            'timestamp': timestamp,
            'exchanges': st.session_state.dual_history,
            'type': 'dual_chat'
        }
        
        # Salvar em data/sessions/
        sessions_dir = Path("data/sessions")
        sessions_dir.mkdir(parents=True, exist_ok=True)
        
        with open(sessions_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2, default=str)
        
        st.success(f"✅ Duelo salvo como {filename}")

def show_chat_metrics():
    """Exibe métricas do chat"""
    if 'chat_history' not in st.session_state or not st.session_state.chat_history:
        st.info("Nenhuma métrica disponível ainda.")
        return
    
    # Filtrar apenas mensagens do assistente com métricas
    assistant_messages = [
        msg for msg in st.session_state.chat_history 
        if msg['role'] == 'assistant' and 'metrics' in msg
    ]
    
    if not assistant_messages:
        st.info("Nenhuma métrica disponível ainda.")
        return
    
    # Criar DataFrame para análise
    metrics_data = []
    for msg in assistant_messages:
        metrics_data.append({
            'modelo': msg['model'],
            'tempo_resposta': msg['metrics']['response_time'],
            'tokens_por_segundo': msg['metrics']['tokens_per_second'],
            'total_tokens': msg['metrics']['total_tokens'],
            'timestamp': msg['timestamp']
        })
    
    df = pd.DataFrame(metrics_data)
    
    # Métricas gerais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Respostas", len(df))
    
    with col2:
        st.metric("Tempo Médio", f"{df['tempo_resposta'].mean():.1f}s")
    
    with col3:
        st.metric("Tokens/s Médio", f"{df['tokens_por_segundo'].mean():.1f}")
    
    with col4:
        st.metric("Total de Tokens", df['total_tokens'].sum())
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de tempo de resposta ao longo do tempo
        fig_time = px.line(
            df, x='timestamp', y='tempo_resposta', color='modelo',
            title="Tempo de Resposta ao Longo do Tempo",
            labels={'tempo_resposta': 'Tempo (s)', 'timestamp': 'Horário'}
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        # Gráfico de tokens por segundo
        fig_tokens = px.line(
            df, x='timestamp', y='tokens_por_segundo', color='modelo',
            title="Velocidade de Geração (Tokens/s)",
            labels={'tokens_por_segundo': 'Tokens/s', 'timestamp': 'Horário'}
        )
        st.plotly_chart(fig_tokens, use_container_width=True)

def show_model_comparison():
    """Exibe comparação entre modelos no duelo"""
    if 'dual_history' not in st.session_state or not st.session_state.dual_history:
        st.info("Nenhum duelo realizado ainda.")
        return
    
    # Analisar resultados dos duelos
    model1_wins = 0
    model2_wins = 0
    ties = 0
    
    model1_name = None
    model2_name = None
    
    for exchange in st.session_state.dual_history:
        if 'evaluation' in exchange:
            model1_name = exchange['model1']
            model2_name = exchange['model2']
            
            winner = exchange['evaluation']['winner']
            if winner == model1_name:
                model1_wins += 1
            elif winner == model2_name:
                model2_wins += 1
            else:
                ties += 1
    
    if model1_name and model2_name:
        # Exibir resultados
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"Vitórias {model1_name}", model1_wins)
        
        with col2:
            st.metric(f"Vitórias {model2_name}", model2_wins)
        
        with col3:
            st.metric("Empates", ties)
        
        # Gráfico de pizza
        fig = go.Figure(data=[go.Pie(
            labels=[model1_name, model2_name, 'Empates'],
            values=[model1_wins, model2_wins, ties],
            hole=.3
        )])
        
        fig.update_layout(title="Resultados do Duelo de Modelos")
        st.plotly_chart(fig, use_container_width=True)
        
        # Métricas de performance
        metrics_data = []
        for exchange in st.session_state.dual_history:
            if 'metrics1' in exchange and 'metrics2' in exchange:
                metrics_data.append({
                    'modelo': exchange['model1'],
                    'tempo_resposta': exchange['metrics1']['response_time'],
                    'tokens_por_segundo': exchange['metrics1']['tokens_per_second']
                })
                metrics_data.append({
                    'modelo': exchange['model2'],
                    'tempo_resposta': exchange['metrics2']['response_time'],
                    'tokens_por_segundo': exchange['metrics2']['tokens_per_second']
                })
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Comparação de tempo de resposta
                fig_time = px.box(
                    df, x='modelo', y='tempo_resposta',
                    title="Comparação de Tempo de Resposta"
                )
                st.plotly_chart(fig_time, use_container_width=True)
            
            with col2:
                # Comparação de velocidade
                fig_speed = px.box(
                    df, x='modelo', y='tokens_por_segundo',
                    title="Comparação de Velocidade (Tokens/s)"
                )
                st.plotly_chart(fig_speed, use_container_width=True)

def main():
    """Função principal da página de chat"""
    
    # Inicializar gerenciadores
    init_chat_manager()
    
    # Título da página
    st.title("💬 Chat & Teste de Modelos")
    st.markdown("---")
    
    # Tabs principais
    tab1, tab2, tab3 = st.tabs(["🧠 Modelos", "📄 Documentos", "💬 Chat"])
    
    with tab1:
        render_model_selector()
    
    with tab2:
        render_document_manager()
    
    with tab3:
        render_chat_interface()

if __name__ == "__main__":
    main()

