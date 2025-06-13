#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerenciador de datasets para o NeuralTrain Forge.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import yaml
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from ..utils.logging_utils import LoggerMixin
from ..utils.file_utils import FileUtils


class DatasetManager(LoggerMixin):
    """Gerenciador de datasets para treinamento."""
    
    def __init__(self):
        self.file_utils = FileUtils()
        self.config = self._load_config()
        self.datasets_dir = Path(self.config.get('directories', {}).get('datasets', 'data/datasets'))
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache de datasets carregados
        self._loaded_datasets = {}
        
    def _load_config(self) -> dict:
        """Carrega configuração da aplicação."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Arquivo de configuração não encontrado: {config_path}")
            return {}
    
    def list_available_datasets(self) -> List[Dict[str, Any]]:
        """
        Lista todos os datasets disponíveis.
        
        Returns:
            Lista de dicionários com informações dos datasets
        """
        datasets = []
        
        try:
            for dataset_dir in self.datasets_dir.iterdir():
                if dataset_dir.is_dir():
                    dataset_info = self._get_dataset_info(dataset_dir)
                    if dataset_info:
                        datasets.append(dataset_info)
        except Exception as e:
            self.logger.error(f"Erro ao listar datasets: {e}")
        
        return sorted(datasets, key=lambda x: x.get('name', ''))
    
    def _get_dataset_info(self, dataset_path: Path) -> Optional[Dict[str, Any]]:
        """
        Obtém informações de um dataset.
        
        Args:
            dataset_path: Caminho para o diretório do dataset
            
        Returns:
            Dicionário com informações do dataset ou None se inválido
        """
        try:
            info_file = dataset_path / "dataset_info.json"
            
            # Se existe arquivo de informações, carrega
            if info_file.exists():
                with open(info_file, 'r', encoding='utf-8') as f:
                    dataset_info = json.load(f)
            else:
                # Cria informações básicas
                dataset_info = {
                    'name': dataset_path.name,
                    'path': str(dataset_path),
                    'type': 'unknown',
                    'size': 0,
                    'created': datetime.now().isoformat()
                }
            
            # Atualiza informações básicas
            dataset_info['path'] = str(dataset_path)
            dataset_info['exists'] = dataset_path.exists()
            
            # Calcula tamanho total e conta arquivos
            total_size = 0
            data_files = []
            
            for file_path in dataset_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.jsonl', '.csv', '.parquet']:
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    
                    data_files.append({
                        'name': file_path.name,
                        'size': file_size,
                        'type': file_path.suffix.lower(),
                        'relative_path': str(file_path.relative_to(dataset_path))
                    })
            
            dataset_info['size'] = total_size
            dataset_info['size_mb'] = round(total_size / (1024 * 1024), 2)
            dataset_info['files'] = data_files
            dataset_info['file_count'] = len(data_files)
            
            # Tenta detectar tipo do dataset
            if 'type' not in dataset_info or dataset_info['type'] == 'unknown':
                dataset_info['type'] = self._detect_dataset_type(dataset_path)
            
            return dataset_info
            
        except Exception as e:
            self.logger.error(f"Erro ao obter informações do dataset {dataset_path}: {e}")
            return None
    
    def _detect_dataset_type(self, dataset_path: Path) -> str:
        """
        Detecta o tipo do dataset baseado nos arquivos.
        
        Args:
            dataset_path: Caminho para o diretório do dataset
            
        Returns:
            Tipo do dataset detectado
        """
        try:
            # Conta tipos de arquivos
            file_types = {}
            for file_path in dataset_path.rglob('*'):
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext in ['.txt', '.jsonl', '.csv', '.parquet']:
                        file_types[ext] = file_types.get(ext, 0) + 1
            
            # Determina tipo principal
            if file_types:
                main_type = max(file_types, key=file_types.get)
                return main_type[1:]  # Remove o ponto
            
        except Exception as e:
            self.logger.error(f"Erro ao detectar tipo do dataset {dataset_path}: {e}")
        
        return 'unknown'
    
    def upload_dataset(self, file_path: str, dataset_name: Optional[str] = None) -> Tuple[bool, str]:
        """
        Faz upload de um dataset local.
        
        Args:
            file_path: Caminho para o arquivo do dataset
            dataset_name: Nome para o dataset (opcional)
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            # Valida o arquivo
            is_valid, message = self.file_utils.validate_dataset_file(file_path)
            if not is_valid:
                return False, message
            
            # Define nome do dataset
            if not dataset_name:
                dataset_name = Path(file_path).stem
            
            dataset_name = self.file_utils.safe_filename(dataset_name)
            dataset_dir = self.datasets_dir / dataset_name
            
            # Cria diretório do dataset
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Copia arquivo
            dest_file = dataset_dir / Path(file_path).name
            shutil.copy2(file_path, dest_file)
            
            # Gera preview do dataset
            preview = self.file_utils.get_dataset_preview(str(dest_file))
            
            # Cria arquivo de informações
            dataset_info = {
                'name': dataset_name,
                'original_file': Path(file_path).name,
                'type': self._detect_dataset_type(dataset_dir),
                'uploaded': datetime.now().isoformat(),
                'source': 'local_upload',
                'preview': preview
            }
            
            info_file = dataset_dir / "dataset_info.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Dataset {dataset_name} carregado com sucesso")
            return True, f"Dataset {dataset_name} carregado com sucesso"
            
        except Exception as e:
            self.logger.error(f"Erro ao fazer upload do dataset: {e}")
            return False, f"Erro ao fazer upload: {str(e)}"
    
    def load_dataset_from_hub(self, dataset_id: str, dataset_name: Optional[str] = None, 
                             config_name: Optional[str] = None, split: Optional[str] = None) -> Tuple[bool, str]:
        """
        Carrega dataset do HuggingFace Hub.
        
        Args:
            dataset_id: ID do dataset no HuggingFace
            dataset_name: Nome local para o dataset (opcional)
            config_name: Configuração específica do dataset (opcional)
            split: Split específico (train, test, validation)
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            # Define nome do dataset
            if not dataset_name:
                dataset_name = dataset_id.replace('/', '_')
            
            dataset_name = self.file_utils.safe_filename(dataset_name)
            dataset_dir = self.datasets_dir / dataset_name
            
            # Cria diretório do dataset
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Carregando dataset {dataset_id} do HuggingFace...")
            
            # Carrega dataset
            dataset = load_dataset(
                dataset_id,
                name=config_name,
                split=split,
                cache_dir=str(dataset_dir / "cache")
            )
            
            # Salva como parquet para acesso rápido
            output_file = dataset_dir / "data.parquet"
            dataset.to_parquet(str(output_file))
            
            # Gera preview
            preview_data = dataset.select(range(min(10, len(dataset))))
            preview = {
                'format': 'parquet',
                'rows': len(preview_data),
                'columns': list(dataset.column_names),
                'preview': preview_data.to_pandas().to_dict('records'),
                'total_rows': len(dataset)
            }
            
            # Cria arquivo de informações
            dataset_info = {
                'name': dataset_name,
                'huggingface_id': dataset_id,
                'config_name': config_name,
                'split': split,
                'type': 'parquet',
                'downloaded': datetime.now().isoformat(),
                'source': 'huggingface',
                'preview': preview,
                'columns': dataset.column_names,
                'features': str(dataset.features)
            }
            
            info_file = dataset_dir / "dataset_info.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Dataset {dataset_id} carregado com sucesso como {dataset_name}")
            return True, f"Dataset {dataset_id} carregado com sucesso"
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar dataset do HuggingFace: {e}")
            return False, f"Erro ao carregar dataset: {str(e)}"
    
    def load_dataset(self, dataset_name: str) -> Tuple[bool, str, Optional[Dataset]]:
        """
        Carrega um dataset na memória.
        
        Args:
            dataset_name: Nome do dataset
            
        Returns:
            Tupla (sucesso, mensagem, dataset_carregado)
        """
        try:
            # Verifica se já está carregado
            if dataset_name in self._loaded_datasets:
                return True, "Dataset já carregado", self._loaded_datasets[dataset_name]
            
            dataset_dir = self.datasets_dir / dataset_name
            if not dataset_dir.exists():
                return False, f"Dataset {dataset_name} não encontrado", None
            
            # Carrega informações do dataset
            info_file = dataset_dir / "dataset_info.json"
            if info_file.exists():
                with open(info_file, 'r', encoding='utf-8') as f:
                    dataset_info = json.load(f)
            else:
                dataset_info = {'type': 'unknown'}
            
            # Encontra arquivo de dados
            data_files = []
            for ext in ['.parquet', '.csv', '.jsonl', '.txt']:
                files = list(dataset_dir.glob(f"*{ext}"))
                if files:
                    data_files.extend(files)
            
            if not data_files:
                return False, f"Nenhum arquivo de dados encontrado para {dataset_name}", None
            
            # Carrega dataset baseado no tipo
            dataset = self._load_dataset_files(data_files, dataset_info.get('type', 'unknown'))
            
            if dataset is None:
                return False, f"Erro ao carregar dataset {dataset_name}", None
            
            # Armazena no cache
            self._loaded_datasets[dataset_name] = dataset
            
            self.logger.info(f"Dataset {dataset_name} carregado com {len(dataset)} exemplos")
            return True, f"Dataset carregado com {len(dataset)} exemplos", dataset
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar dataset {dataset_name}: {e}")
            return False, f"Erro ao carregar dataset: {str(e)}", None
    
    def _load_dataset_files(self, data_files: List[Path], dataset_type: str) -> Optional[Dataset]:
        """
        Carrega arquivos de dados como Dataset.
        
        Args:
            data_files: Lista de arquivos de dados
            dataset_type: Tipo do dataset
            
        Returns:
            Dataset carregado ou None
        """
        try:
            # Converte paths para strings
            file_paths = [str(f) for f in data_files]
            
            if dataset_type == 'parquet':
                return Dataset.from_parquet(file_paths[0])
            
            elif dataset_type == 'csv':
                df = pd.read_csv(file_paths[0])
                return Dataset.from_pandas(df)
            
            elif dataset_type == 'jsonl':
                return Dataset.from_json(file_paths[0])
            
            elif dataset_type == 'txt':
                # Para arquivos de texto, cada linha é um exemplo
                texts = []
                for file_path in file_paths:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        texts.extend([line.strip() for line in f if line.strip()])
                
                return Dataset.from_dict({'text': texts})
            
            else:
                # Tenta detectar automaticamente
                for file_path in file_paths:
                    ext = Path(file_path).suffix.lower()
                    
                    if ext == '.parquet':
                        return Dataset.from_parquet(file_path)
                    elif ext == '.csv':
                        df = pd.read_csv(file_path)
                        return Dataset.from_pandas(df)
                    elif ext == '.jsonl':
                        return Dataset.from_json(file_path)
                    elif ext == '.txt':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            texts = [line.strip() for line in f if line.strip()]
                        return Dataset.from_dict({'text': texts})
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar arquivos de dados: {e}")
            return None
    
    def preprocess_dataset(self, dataset_name: str, tokenizer_name: str, 
                          text_column: str = 'text', max_length: int = 512) -> Tuple[bool, str, Optional[Dataset]]:
        """
        Preprocessa dataset para treinamento.
        
        Args:
            dataset_name: Nome do dataset
            tokenizer_name: Nome do tokenizer/modelo
            text_column: Nome da coluna de texto
            max_length: Comprimento máximo dos tokens
            
        Returns:
            Tupla (sucesso, mensagem, dataset_preprocessado)
        """
        try:
            # Carrega dataset
            success, message, dataset = self.load_dataset(dataset_name)
            if not success or dataset is None:
                return False, message, None
            
            # Carrega tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                
                # Adiciona pad token se não existir
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
            except Exception as e:
                return False, f"Erro ao carregar tokenizer {tokenizer_name}: {str(e)}", None
            
            # Verifica se a coluna de texto existe
            if text_column not in dataset.column_names:
                available_columns = ', '.join(dataset.column_names)
                return False, f"Coluna '{text_column}' não encontrada. Colunas disponíveis: {available_columns}", None
            
            # Função de tokenização
            def tokenize_function(examples):
                # Tokeniza textos
                tokenized = tokenizer(
                    examples[text_column],
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors=None
                )
                
                # Para modelos causais, labels são os mesmos que input_ids
                tokenized['labels'] = tokenized['input_ids'].copy()
                
                return tokenized
            
            # Aplica tokenização
            self.logger.info(f"Tokenizando dataset {dataset_name}...")
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
                desc="Tokenizando"
            )
            
            # Armazena dataset preprocessado
            preprocessed_name = f"{dataset_name}_preprocessed"
            self._loaded_datasets[preprocessed_name] = tokenized_dataset
            
            self.logger.info(f"Dataset {dataset_name} preprocessado com sucesso")
            return True, f"Dataset preprocessado com {len(tokenized_dataset)} exemplos", tokenized_dataset
            
        except Exception as e:
            self.logger.error(f"Erro ao preprocessar dataset: {e}")
            return False, f"Erro ao preprocessar dataset: {str(e)}", None
    
    def split_dataset(self, dataset_name: str, train_ratio: float = 0.8, 
                     test_ratio: float = 0.1, val_ratio: float = 0.1) -> Tuple[bool, str, Optional[Dict[str, Dataset]]]:
        """
        Divide dataset em treino, validação e teste.
        
        Args:
            dataset_name: Nome do dataset
            train_ratio: Proporção para treino
            test_ratio: Proporção para teste
            val_ratio: Proporção para validação
            
        Returns:
            Tupla (sucesso, mensagem, splits_dict)
        """
        try:
            # Verifica se as proporções somam 1
            if abs(train_ratio + test_ratio + val_ratio - 1.0) > 0.01:
                return False, "As proporções devem somar 1.0", None
            
            # Carrega dataset
            success, message, dataset = self.load_dataset(dataset_name)
            if not success or dataset is None:
                return False, message, None
            
            # Divide dataset
            total_size = len(dataset)
            train_size = int(total_size * train_ratio)
            test_size = int(total_size * test_ratio)
            val_size = total_size - train_size - test_size
            
            # Embaralha dataset
            dataset = dataset.shuffle(seed=42)
            
            # Cria splits
            train_dataset = dataset.select(range(train_size))
            test_dataset = dataset.select(range(train_size, train_size + test_size))
            val_dataset = dataset.select(range(train_size + test_size, total_size))
            
            splits = {
                'train': train_dataset,
                'test': test_dataset,
                'validation': val_dataset
            }
            
            # Armazena splits
            for split_name, split_dataset in splits.items():
                cache_name = f"{dataset_name}_{split_name}"
                self._loaded_datasets[cache_name] = split_dataset
            
            self.logger.info(f"Dataset {dataset_name} dividido: {len(train_dataset)} treino, {len(test_dataset)} teste, {len(val_dataset)} validação")
            return True, f"Dataset dividido com sucesso", splits
            
        except Exception as e:
            self.logger.error(f"Erro ao dividir dataset: {e}")
            return False, f"Erro ao dividir dataset: {str(e)}", None
    
    def get_dataset_stats(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Obtém estatísticas de um dataset.
        
        Args:
            dataset_name: Nome do dataset
            
        Returns:
            Dicionário com estatísticas ou None
        """
        try:
            success, message, dataset = self.load_dataset(dataset_name)
            if not success or dataset is None:
                return None
            
            stats = {
                'total_examples': len(dataset),
                'columns': dataset.column_names,
                'features': str(dataset.features)
            }
            
            # Estatísticas específicas para colunas de texto
            if 'text' in dataset.column_names:
                texts = dataset['text']
                text_lengths = [len(text) for text in texts[:1000]]  # Amostra de 1000
                
                stats['text_stats'] = {
                    'avg_length': sum(text_lengths) / len(text_lengths),
                    'min_length': min(text_lengths),
                    'max_length': max(text_lengths),
                    'sample_size': len(text_lengths)
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Erro ao obter estatísticas do dataset: {e}")
            return None
    
    def delete_dataset(self, dataset_name: str) -> Tuple[bool, str]:
        """
        Remove dataset do disco e memória.
        
        Args:
            dataset_name: Nome do dataset
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            # Remove da memória
            if dataset_name in self._loaded_datasets:
                del self._loaded_datasets[dataset_name]
            
            # Remove do disco
            dataset_dir = self.datasets_dir / dataset_name
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
                self.logger.info(f"Dataset {dataset_name} removido do disco")
                return True, f"Dataset {dataset_name} removido com sucesso"
            else:
                return False, f"Dataset {dataset_name} não encontrado no disco"
                
        except Exception as e:
            self.logger.error(f"Erro ao remover dataset: {e}")
            return False, f"Erro ao remover dataset: {str(e)}"
    
    def get_loaded_datasets(self) -> List[str]:
        """
        Retorna lista de datasets carregados na memória.
        
        Returns:
            Lista com nomes dos datasets carregados
        """
        return list(self._loaded_datasets.keys())

