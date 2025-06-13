#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Página de upload de datasets do NeuralTrain Forge.
"""

import streamlit as st
import pandas as pd
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List, Optional


def render():
    """Renderiza a página de upload de datasets."""
    
    # Header da página
    st.markdown("# 📊 Upload de Datasets")
    st.markdown("Carregue e gerencie datasets para treinamento de modelos")
    
    # Tabs para diferentes funcionalidades
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Local", "🤗 HuggingFace", "📋 Datasets Carregados", "🔍 Preview"])
    
    with tab1:
        render_local_upload()
    
    with tab2:
        render_huggingface_datasets()
    
    with tab3:
        render_loaded_datasets()
    
    with tab4:
        render_dataset_preview()


def render_local_upload():
    """Renderiza interface para upload de datasets locais."""
    
    st.markdown("## 📁 Upload de Dataset Local")
    st.markdown("Carregue datasets nos formatos .txt, .jsonl, .csv ou .parquet")
    
    # Informações sobre formatos
    with st.expander("ℹ️ Formatos Suportados"):
        st.markdown("""
        **Formatos aceitos:**
        - **.txt**: Texto simples (uma linha por exemplo)
        - **.jsonl**: JSON Lines (um objeto JSON por linha)
        - **.csv**: Valores separados por vírgula
        - **.parquet**: Formato otimizado para dados estruturados
        
        **Estrutura recomendada:**
        - **Para chat/conversação**: coluna 'text' com diálogos
        - **Para instrução**: colunas 'instruction', 'input', 'output'
        - **Para texto livre**: coluna 'text' com conteúdo
        
        **Tamanho máximo:** 1GB por arquivo
        """)
    
    # Upload de arquivo
    uploaded_file = st.file_uploader(
        "Selecione o arquivo do dataset",
        type=['txt', 'jsonl', 'csv', 'parquet'],
        help="Arraste e solte ou clique para selecionar"
    )
    
    if uploaded_file is not None:
        # Mostra informações do arquivo
        file_details = {
            "Nome": uploaded_file.name,
            "Tamanho": f"{uploaded_file.size / (1024*1024):.2f} MB",
            "Tipo": uploaded_file.type or Path(uploaded_file.name).suffix
        }
        
        st.markdown("### 📄 Informações do Arquivo")
        for key, value in file_details.items():
            st.info(f"**{key}:** {value}")
        
        # Preview do conteúdo
        st.markdown("### 👀 Preview do Conteúdo")
        preview_data = get_file_preview(uploaded_file)
        
        if preview_data:
            display_preview(preview_data)
        
        # Configurações do dataset
        st.markdown("### ⚙️ Configurações")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_name = st.text_input(
                "Nome do dataset",
                value=Path(uploaded_file.name).stem,
                help="Nome que será usado para identificar o dataset"
            )
        
        with col2:
            dataset_type = st.selectbox(
                "Tipo de dados",
                ["Auto-detectar", "Conversação", "Instrução", "Texto Livre", "Classificação"],
                help="Tipo de dados para otimizações específicas"
            )
        
        # Configurações de processamento
        with st.expander("🔧 Configurações de Processamento"):
            col1, col2 = st.columns(2)
            
            with col1:
                text_column = st.text_input(
                    "Coluna de texto principal",
                    value="text",
                    help="Nome da coluna que contém o texto principal"
                )
                
                max_length = st.number_input(
                    "Comprimento máximo (tokens)",
                    min_value=128,
                    max_value=4096,
                    value=512,
                    help="Máximo de tokens por exemplo"
                )
            
            with col2:
                encoding = st.selectbox(
                    "Codificação",
                    ["utf-8", "latin-1", "ascii"],
                    help="Codificação do arquivo de texto"
                )
                
                remove_duplicates = st.checkbox(
                    "Remover duplicatas",
                    value=True,
                    help="Remove exemplos duplicados automaticamente"
                )
        
        # Descrição opcional
        description = st.text_area(
            "Descrição (opcional)",
            placeholder="Descreva o dataset, sua origem, características especiais...",
            help="Informações adicionais sobre o dataset"
        )
        
        # Validação e upload
        if dataset_name:
            if not dataset_name.replace('_', '').replace('-', '').isalnum():
                st.error("❌ Nome deve conter apenas letras, números, hífens e underscores")
            elif len(dataset_name) < 3:
                st.error("❌ Nome deve ter pelo menos 3 caracteres")
            else:
                if st.button("📊 Carregar Dataset", type="primary", use_container_width=True):
                    upload_local_dataset(
                        uploaded_file, dataset_name, dataset_type, 
                        text_column, max_length, encoding, remove_duplicates, description
                    )
        else:
            st.warning("⚠️ Digite um nome para o dataset")


def render_huggingface_datasets():
    """Renderiza interface para datasets do HuggingFace."""
    
    st.markdown("## 🤗 Datasets do HuggingFace")
    st.markdown("Carregue datasets diretamente do HuggingFace Hub")
    
    # Datasets populares
    st.markdown("### 🌟 Datasets Populares")
    
    popular_datasets = [
        {
            "id": "squad",
            "name": "SQuAD",
            "description": "Stanford Question Answering Dataset",
            "size": "~100MB",
            "type": "Q&A"
        },
        {
            "id": "imdb",
            "name": "IMDB Reviews",
            "description": "Análise de sentimento de reviews de filmes",
            "size": "~80MB",
            "type": "Classificação"
        },
        {
            "id": "wikitext",
            "name": "WikiText",
            "description": "Texto da Wikipedia para modelagem de linguagem",
            "size": "~200MB",
            "type": "Texto"
        }
    ]
    
    cols = st.columns(len(popular_datasets))
    
    for i, dataset in enumerate(popular_datasets):
        with cols[i]:
            with st.container():
                st.markdown(f"**{dataset['name']}**")
                st.caption(dataset['description'])
                st.caption(f"Tipo: {dataset['type']}")
                st.caption(f"Tamanho: {dataset['size']}")
                
                if st.button(f"📥 Carregar", key=f"hf_dataset_{i}", use_container_width=True):
                    download_huggingface_dataset(dataset['id'], dataset['name'])
    
    st.markdown("---")
    
    # Download personalizado
    st.markdown("### 🔍 Dataset Personalizado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dataset_id = st.text_input(
            "ID do dataset",
            placeholder="ex: squad, imdb, wikitext",
            help="ID do dataset no HuggingFace Hub"
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
                "Configuração",
                placeholder="ex: plain_text",
                help="Configuração específica do dataset"
            )
            
            split = st.selectbox(
                "Split",
                ["train", "validation", "test", "all"],
                help="Parte específica do dataset"
            )
        
        with col2:
            streaming = st.checkbox(
                "Modo streaming",
                help="Para datasets muito grandes"
            )
            
            trust_remote_code = st.checkbox(
                "Confiar em código remoto",
                help="Para datasets que requerem código personalizado"
            )
    
    # Download
    if dataset_id:
        if st.button("🤗 Carregar do HuggingFace", type="primary", use_container_width=True):
            download_huggingface_dataset(
                dataset_id,
                local_name or dataset_id,
                config_name or None,
                split if split != "all" else None,
                streaming,
                trust_remote_code
            )


def render_loaded_datasets():
    """Renderiza lista de datasets carregados."""
    
    st.markdown("## 📋 Datasets Carregados")
    st.markdown("Gerencie os datasets disponíveis no sistema")
    
    # Controles
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔄 Atualizar", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("🧹 Limpar Cache", use_container_width=True):
            clear_dataset_cache()
    
    # Lista de datasets
    datasets = get_loaded_datasets_list()
    
    if not datasets:
        st.info("📭 Nenhum dataset carregado ainda")
        st.markdown("Use as abas acima para carregar seu primeiro dataset!")
    else:
        # Filtros
        st.markdown("### 🔍 Filtros")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_type = st.selectbox(
                "Tipo",
                ["Todos", "Conversação", "Instrução", "Texto Livre", "Classificação"]
            )
        
        with col2:
            filter_source = st.selectbox(
                "Origem",
                ["Todas", "Local", "HuggingFace"]
            )
        
        with col3:
            sort_by = st.selectbox(
                "Ordenar por",
                ["Nome", "Data", "Tamanho", "Exemplos"]
            )
        
        # Lista de datasets
        st.markdown("### 📚 Datasets Disponíveis")
        
        for i, dataset in enumerate(datasets):
            render_dataset_card(dataset, i)


def render_dataset_preview():
    """Renderiza preview de datasets."""
    
    st.markdown("## 🔍 Preview de Datasets")
    st.markdown("Visualize e analise o conteúdo dos datasets")
    
    # Seletor de dataset
    datasets = get_loaded_datasets_list()
    
    if not datasets:
        st.info("📭 Nenhum dataset disponível para preview")
        return
    
    dataset_names = [d['name'] for d in datasets]
    selected_dataset = st.selectbox(
        "Selecione um dataset",
        dataset_names,
        help="Dataset para visualizar"
    )
    
    if selected_dataset:
        dataset = next(d for d in datasets if d['name'] == selected_dataset)
        
        # Informações do dataset
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Exemplos", dataset['examples'])
        
        with col2:
            st.metric("Colunas", dataset['columns'])
        
        with col3:
            st.metric("Tamanho", dataset['size'])
        
        with col4:
            st.metric("Tipo", dataset['type'])
        
        # Preview dos dados
        st.markdown("### 📊 Amostra dos Dados")
        
        # Gera dados de exemplo
        sample_data = generate_sample_data(dataset)
        
        if sample_data:
            # Mostra tabela
            st.dataframe(sample_data, use_container_width=True)
            
            # Estatísticas
            st.markdown("### 📈 Estatísticas")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Distribuição de Comprimento:**")
                # Gráfico de distribuição (placeholder)
                lengths = [len(str(text)) for text in sample_data.iloc[:, 0] if pd.notna(text)]
                if lengths:
                    st.bar_chart(pd.Series(lengths).value_counts().head(10))
            
            with col2:
                st.markdown("**Informações Gerais:**")
                st.text(f"Média de caracteres: {sum(lengths)/len(lengths):.0f}")
                st.text(f"Mínimo: {min(lengths)}")
                st.text(f"Máximo: {max(lengths)}")
                st.text(f"Total de exemplos: {len(sample_data)}")


def render_dataset_card(dataset: Dict[str, Any], index: int):
    """Renderiza card de um dataset."""
    
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            st.markdown(f"**📊 {dataset['name']}**")
            st.caption(f"Tipo: {dataset['type']} | Origem: {dataset['source']}")
            if dataset.get('description'):
                st.caption(dataset['description'])
        
        with col2:
            st.metric("Exemplos", dataset['examples'])
        
        with col3:
            st.metric("Tamanho", dataset['size'])
        
        with col4:
            # Menu de ações
            action = st.selectbox(
                "Ações",
                ["Selecionar", "Preview", "Processar", "Dividir", "Remover"],
                key=f"dataset_action_{index}",
                label_visibility="collapsed"
            )
            
            if action == "Preview":
                show_dataset_preview(dataset)
            elif action == "Processar":
                process_dataset(dataset['name'])
            elif action == "Dividir":
                split_dataset(dataset['name'])
            elif action == "Remover":
                remove_dataset(dataset['name'])
        
        st.markdown("---")


def get_file_preview(uploaded_file) -> Optional[Dict[str, Any]]:
    """Obtém preview do arquivo carregado."""
    
    try:
        file_ext = Path(uploaded_file.name).suffix.lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(uploaded_file, nrows=5)
            return {
                'type': 'csv',
                'data': df,
                'columns': list(df.columns),
                'rows': len(df)
            }
        
        elif file_ext == '.jsonl':
            lines = []
            content = uploaded_file.getvalue().decode('utf-8')
            for i, line in enumerate(content.split('\n')[:5]):
                if line.strip():
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            
            return {
                'type': 'jsonl',
                'data': lines,
                'rows': len(lines)
            }
        
        elif file_ext == '.txt':
            content = uploaded_file.getvalue().decode('utf-8')
            lines = content.split('\n')[:5]
            
            return {
                'type': 'txt',
                'data': lines,
                'rows': len(lines)
            }
        
        elif file_ext == '.parquet':
            df = pd.read_parquet(uploaded_file)
            preview_df = df.head(5)
            
            return {
                'type': 'parquet',
                'data': preview_df,
                'columns': list(df.columns),
                'rows': len(preview_df)
            }
    
    except Exception as e:
        st.error(f"Erro ao gerar preview: {str(e)}")
        return None


def display_preview(preview_data: Dict[str, Any]):
    """Exibe preview dos dados."""
    
    if preview_data['type'] in ['csv', 'parquet']:
        st.dataframe(preview_data['data'], use_container_width=True)
        st.caption(f"Mostrando {preview_data['rows']} de ? linhas")
    
    elif preview_data['type'] == 'jsonl':
        for i, item in enumerate(preview_data['data']):
            with st.expander(f"Linha {i+1}"):
                st.json(item)
    
    elif preview_data['type'] == 'txt':
        for i, line in enumerate(preview_data['data']):
            st.text(f"Linha {i+1}: {line[:100]}{'...' if len(line) > 100 else ''}")


def upload_local_dataset(uploaded_file, dataset_name: str, dataset_type: str, 
                        text_column: str, max_length: int, encoding: str, 
                        remove_duplicates: bool, description: str):
    """Processa upload de dataset local."""
    
    try:
        with st.spinner("📊 Carregando dataset..."):
            # Cria arquivo temporário
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Aqui seria chamado o DatasetManager
            # success, message = dataset_manager.upload_dataset(tmp_file_path, dataset_name)
            
            # Simulação
            success = True
            message = f"Dataset {dataset_name} carregado com sucesso!"
            
            # Remove arquivo temporário
            os.unlink(tmp_file_path)
            
            if success:
                st.success(f"✅ {message}")
                st.balloons()
                
                # Adiciona ao histórico
                add_to_dataset_history("local", dataset_name, uploaded_file.size)
                
            else:
                st.error(f"❌ {message}")
                
    except Exception as e:
        st.error(f"❌ Erro ao carregar dataset: {str(e)}")


def download_huggingface_dataset(dataset_id: str, local_name: str, 
                                config_name: Optional[str] = None,
                                split: Optional[str] = None,
                                streaming: bool = False,
                                trust_remote_code: bool = False):
    """Processa download de dataset do HuggingFace."""
    
    try:
        with st.spinner(f"🤗 Carregando {dataset_id}..."):
            # Aqui seria chamado o DatasetManager
            # success, message = dataset_manager.load_dataset_from_hub(dataset_id, local_name, config_name, split)
            
            # Simulação
            success = True
            message = f"Dataset {dataset_id} carregado com sucesso!"
            
            if success:
                st.success(f"✅ {message}")
                st.balloons()
                
                # Adiciona ao histórico
                add_to_dataset_history("huggingface", local_name, "N/A")
                
            else:
                st.error(f"❌ {message}")
                
    except Exception as e:
        st.error(f"❌ Erro ao carregar dataset: {str(e)}")


def get_loaded_datasets_list():
    """Retorna lista de datasets carregados (placeholder)."""
    
    return [
        {
            "name": "conversational_data",
            "type": "Conversação",
            "source": "Local",
            "size": "2.3MB",
            "examples": "1,250",
            "columns": "3",
            "description": "Dataset de conversas para chatbot"
        },
        {
            "name": "instruction_following",
            "type": "Instrução",
            "source": "HuggingFace",
            "size": "15.7MB",
            "examples": "5,000",
            "columns": "4",
            "description": "Dataset de instruções e respostas"
        },
        {
            "name": "text_classification",
            "type": "Classificação",
            "source": "Local",
            "size": "8.1MB",
            "examples": "3,200",
            "columns": "2",
            "description": "Dataset para classificação de texto"
        }
    ]


def generate_sample_data(dataset: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Gera dados de amostra para preview."""
    
    # Dados de exemplo baseados no tipo
    if dataset['type'] == 'Conversação':
        return pd.DataFrame({
            'input': [
                "Olá, como você está?",
                "Qual é a capital do Brasil?",
                "Me conte uma piada",
                "Como fazer um bolo?",
                "Que horas são?"
            ],
            'output': [
                "Olá! Estou bem, obrigado por perguntar. Como posso ajudá-lo?",
                "A capital do Brasil é Brasília.",
                "Por que o livro de matemática estava triste? Porque tinha muitos problemas!",
                "Para fazer um bolo, você precisa de farinha, ovos, açúcar...",
                "Desculpe, não tenho acesso ao horário atual."
            ]
        })
    
    elif dataset['type'] == 'Instrução':
        return pd.DataFrame({
            'instruction': [
                "Traduza para o inglês",
                "Resuma o texto",
                "Corrija a gramática",
                "Explique o conceito",
                "Crie uma lista"
            ],
            'input': [
                "Olá mundo",
                "Este é um texto muito longo sobre...",
                "Eu vai na escola ontem",
                "Machine Learning",
                "Frutas tropicais"
            ],
            'output': [
                "Hello world",
                "Este texto fala sobre...",
                "Eu fui à escola ontem",
                "Machine Learning é...",
                "1. Manga 2. Abacaxi 3. Caju..."
            ]
        })
    
    else:
        return pd.DataFrame({
            'text': [
                "Este é um exemplo de texto para análise.",
                "Outro exemplo com conteúdo diferente.",
                "Mais um texto de amostra aqui.",
                "Texto adicional para demonstração.",
                "Último exemplo de texto."
            ],
            'label': [
                "positivo",
                "negativo", 
                "neutro",
                "positivo",
                "negativo"
            ]
        })


def show_dataset_preview(dataset: Dict[str, Any]):
    """Mostra preview detalhado do dataset."""
    
    with st.expander(f"🔍 Preview - {dataset['name']}", expanded=True):
        # Informações básicas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Exemplos", dataset['examples'])
        
        with col2:
            st.metric("Colunas", dataset['columns'])
        
        with col3:
            st.metric("Tamanho", dataset['size'])
        
        # Amostra dos dados
        sample_data = generate_sample_data(dataset)
        if sample_data is not None:
            st.dataframe(sample_data, use_container_width=True)


def process_dataset(dataset_name: str):
    """Processa dataset para treinamento."""
    
    with st.spinner(f"🔄 Processando {dataset_name}..."):
        # Aqui seria chamado o DatasetManager
        st.success(f"✅ Dataset {dataset_name} processado!")


def split_dataset(dataset_name: str):
    """Divide dataset em treino/validação/teste."""
    
    with st.spinner(f"✂️ Dividindo {dataset_name}..."):
        # Aqui seria chamado o DatasetManager
        st.success(f"✅ Dataset {dataset_name} dividido!")


def remove_dataset(dataset_name: str):
    """Remove dataset do sistema."""
    
    if st.button(f"⚠️ Confirmar remoção de {dataset_name}", type="secondary"):
        with st.spinner(f"🗑️ Removendo {dataset_name}..."):
            # Aqui seria chamado o DatasetManager
            st.success(f"✅ Dataset {dataset_name} removido!")
            st.rerun()


def clear_dataset_cache():
    """Limpa cache de datasets."""
    
    with st.spinner("🧹 Limpando cache..."):
        st.success("✅ Cache limpo com sucesso!")


def add_to_dataset_history(source: str, name: str, size):
    """Adiciona entrada ao histórico de datasets."""
    
    if 'dataset_history' not in st.session_state:
        st.session_state.dataset_history = []
    
    entry = {
        'timestamp': str(pd.Timestamp.now()),
        'source': source,
        'name': name,
        'size': size
    }
    
    st.session_state.dataset_history.append(entry)
    
    # Mantém apenas os últimos 10
    if len(st.session_state.dataset_history) > 10:
        st.session_state.dataset_history = st.session_state.dataset_history[-10:]

