#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerenciador de modelos para o NeuralTrain Forge.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from huggingface_hub import hf_hub_download, list_repo_files
from ..utils.logging_utils import LoggerMixin
from ..utils.file_utils import FileUtils


class ModelManager(LoggerMixin):
    """Gerenciador de modelos de linguagem."""
    
    def __init__(self):
        self.file_utils = FileUtils()
        self.config = self._load_config()
        self.models_dir = Path(self.config.get('directories', {}).get('models', 'data/models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache de modelos carregados
        self._loaded_models = {}
        self._model_configs = {}
        
    def _load_config(self) -> dict:
        """Carrega configuração da aplicação."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Arquivo de configuração não encontrado: {config_path}")
            return {}
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Lista todos os modelos disponíveis no diretório de modelos.
        
        Returns:
            Lista de dicionários com informações dos modelos
        """
        models = []
        
        try:
            for model_dir in self.models_dir.iterdir():
                if model_dir.is_dir():
                    model_info = self._get_model_info(model_dir)
                    if model_info:
                        models.append(model_info)
        except Exception as e:
            self.logger.error(f"Erro ao listar modelos: {e}")
        
        return sorted(models, key=lambda x: x.get('name', ''))
    
    def _get_model_info(self, model_path: Path) -> Optional[Dict[str, Any]]:
        """
        Obtém informações de um modelo.
        
        Args:
            model_path: Caminho para o diretório do modelo
            
        Returns:
            Dicionário com informações do modelo ou None se inválido
        """
        try:
            info_file = model_path / "model_info.json"
            
            # Se existe arquivo de informações, carrega
            if info_file.exists():
                with open(info_file, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
            else:
                # Cria informações básicas
                model_info = {
                    'name': model_path.name,
                    'path': str(model_path),
                    'type': 'unknown',
                    'size': 0,
                    'created': datetime.now().isoformat()
                }
            
            # Atualiza informações básicas
            model_info['path'] = str(model_path)
            model_info['exists'] = model_path.exists()
            
            # Calcula tamanho total
            total_size = 0
            model_files = []
            
            for file_path in model_path.rglob('*'):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    
                    # Identifica arquivos importantes
                    if file_path.suffix.lower() in ['.bin', '.safetensors', '.gguf']:
                        model_files.append({
                            'name': file_path.name,
                            'size': file_size,
                            'type': file_path.suffix.lower()
                        })
            
            model_info['size'] = total_size
            model_info['size_mb'] = round(total_size / (1024 * 1024), 2)
            model_info['files'] = model_files
            
            # Tenta detectar tipo do modelo
            if 'type' not in model_info or model_info['type'] == 'unknown':
                model_info['type'] = self._detect_model_type(model_path)
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"Erro ao obter informações do modelo {model_path}: {e}")
            return None
    
    def _detect_model_type(self, model_path: Path) -> str:
        """
        Detecta o tipo do modelo baseado nos arquivos.
        
        Args:
            model_path: Caminho para o diretório do modelo
            
        Returns:
            Tipo do modelo detectado
        """
        try:
            # Verifica se existe config.json
            config_file = model_path / "config.json"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                model_type = config.get('model_type', '').lower()
                architectures = config.get('architectures', [])
                
                if model_type:
                    return model_type
                elif architectures:
                    arch = architectures[0].lower()
                    if 'llama' in arch:
                        return 'llama'
                    elif 'mistral' in arch:
                        return 'mistral'
                    elif 'gpt' in arch:
                        return 'gpt'
            
            # Verifica por nome do diretório
            name = model_path.name.lower()
            if 'llama' in name:
                return 'llama'
            elif 'mistral' in name:
                return 'mistral'
            elif 'gpt' in name:
                return 'gpt'
            
            # Verifica por extensão de arquivos
            for file_path in model_path.iterdir():
                if file_path.suffix.lower() == '.gguf':
                    return 'gguf'
                elif file_path.suffix.lower() in ['.bin', '.safetensors']:
                    return 'transformers'
            
        except Exception as e:
            self.logger.error(f"Erro ao detectar tipo do modelo {model_path}: {e}")
        
        return 'unknown'
    
    def upload_model(self, file_path: str, model_name: Optional[str] = None) -> Tuple[bool, str]:
        """
        Faz upload de um modelo local.
        
        Args:
            file_path: Caminho para o arquivo do modelo
            model_name: Nome para o modelo (opcional)
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            # Valida o arquivo
            is_valid, message = self.file_utils.validate_model_file(file_path)
            if not is_valid:
                return False, message
            
            # Define nome do modelo
            if not model_name:
                model_name = Path(file_path).stem
            
            model_name = self.file_utils.safe_filename(model_name)
            model_dir = self.models_dir / model_name
            
            # Cria diretório do modelo
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Copia arquivo
            dest_file = model_dir / Path(file_path).name
            shutil.copy2(file_path, dest_file)
            
            # Cria arquivo de informações
            model_info = {
                'name': model_name,
                'original_file': Path(file_path).name,
                'type': self._detect_model_type(model_dir),
                'uploaded': datetime.now().isoformat(),
                'source': 'local_upload'
            }
            
            info_file = model_dir / "model_info.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Modelo {model_name} carregado com sucesso")
            return True, f"Modelo {model_name} carregado com sucesso"
            
        except Exception as e:
            self.logger.error(f"Erro ao fazer upload do modelo: {e}")
            return False, f"Erro ao fazer upload: {str(e)}"
    
    def download_from_huggingface(self, model_id: str, model_name: Optional[str] = None) -> Tuple[bool, str]:
        """
        Baixa modelo do HuggingFace Hub.
        
        Args:
            model_id: ID do modelo no HuggingFace (ex: 'microsoft/DialoGPT-medium')
            model_name: Nome local para o modelo (opcional)
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            # Define nome do modelo
            if not model_name:
                model_name = model_id.replace('/', '_')
            
            model_name = self.file_utils.safe_filename(model_name)
            model_dir = self.models_dir / model_name
            
            # Cria diretório do modelo
            model_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Baixando modelo {model_id} do HuggingFace...")
            
            # Lista arquivos do repositório
            try:
                repo_files = list_repo_files(model_id)
            except Exception as e:
                return False, f"Erro ao acessar repositório {model_id}: {str(e)}"
            
            # Baixa arquivos principais
            essential_files = [
                'config.json', 'tokenizer.json', 'tokenizer_config.json',
                'vocab.json', 'merges.txt', 'special_tokens_map.json'
            ]
            
            # Baixa arquivos do modelo
            model_files = [f for f in repo_files if f.endswith(('.bin', '.safetensors', '.gguf'))]
            
            downloaded_files = []
            
            # Baixa arquivos essenciais
            for file_name in essential_files:
                if file_name in repo_files:
                    try:
                        file_path = hf_hub_download(
                            repo_id=model_id,
                            filename=file_name,
                            local_dir=model_dir,
                            local_dir_use_symlinks=False
                        )
                        downloaded_files.append(file_name)
                    except Exception as e:
                        self.logger.warning(f"Erro ao baixar {file_name}: {e}")
            
            # Baixa arquivos do modelo (limitado aos primeiros para evitar downloads muito grandes)
            for file_name in model_files[:5]:  # Limita a 5 arquivos
                try:
                    file_path = hf_hub_download(
                        repo_id=model_id,
                        filename=file_name,
                        local_dir=model_dir,
                        local_dir_use_symlinks=False
                    )
                    downloaded_files.append(file_name)
                except Exception as e:
                    self.logger.warning(f"Erro ao baixar {file_name}: {e}")
            
            if not downloaded_files:
                return False, "Nenhum arquivo foi baixado com sucesso"
            
            # Cria arquivo de informações
            model_info = {
                'name': model_name,
                'huggingface_id': model_id,
                'type': self._detect_model_type(model_dir),
                'downloaded': datetime.now().isoformat(),
                'source': 'huggingface',
                'downloaded_files': downloaded_files
            }
            
            info_file = model_dir / "model_info.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Modelo {model_id} baixado com sucesso como {model_name}")
            return True, f"Modelo {model_id} baixado com sucesso"
            
        except Exception as e:
            self.logger.error(f"Erro ao baixar modelo do HuggingFace: {e}")
            return False, f"Erro ao baixar modelo: {str(e)}"
    
    def load_model(self, model_name: str) -> Tuple[bool, str, Optional[Any]]:
        """
        Carrega um modelo na memória.
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            Tupla (sucesso, mensagem, modelo_carregado)
        """
        try:
            # Verifica se já está carregado
            if model_name in self._loaded_models:
                return True, "Modelo já carregado", self._loaded_models[model_name]
            
            model_dir = self.models_dir / model_name
            if not model_dir.exists():
                return False, f"Modelo {model_name} não encontrado", None
            
            # Carrega informações do modelo
            info_file = model_dir / "model_info.json"
            if info_file.exists():
                with open(info_file, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
            else:
                model_info = {'type': 'unknown'}
            
            model_type = model_info.get('type', 'unknown')
            
            # Carrega modelo baseado no tipo
            if model_type in ['transformers', 'llama', 'mistral', 'gpt']:
                return self._load_transformers_model(model_dir, model_name)
            elif model_type == 'gguf':
                return self._load_gguf_model(model_dir, model_name)
            else:
                return False, f"Tipo de modelo não suportado: {model_type}", None
                
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo {model_name}: {e}")
            return False, f"Erro ao carregar modelo: {str(e)}", None
    
    def _load_transformers_model(self, model_dir: Path, model_name: str) -> Tuple[bool, str, Optional[Any]]:
        """Carrega modelo do tipo Transformers."""
        try:
            # Verifica se tem GPU disponível
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Carrega configuração
            config = AutoConfig.from_pretrained(model_dir)
            
            # Carrega tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
            # Carrega modelo
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                config=config,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            # Armazena no cache
            self._loaded_models[model_name] = {
                'model': model,
                'tokenizer': tokenizer,
                'config': config,
                'device': device,
                'type': 'transformers'
            }
            
            self.logger.info(f"Modelo Transformers {model_name} carregado no {device}")
            return True, f"Modelo carregado no {device}", self._loaded_models[model_name]
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo Transformers: {e}")
            return False, f"Erro ao carregar modelo Transformers: {str(e)}", None
    
    def _load_gguf_model(self, model_dir: Path, model_name: str) -> Tuple[bool, str, Optional[Any]]:
        """Carrega modelo do tipo GGUF."""
        try:
            # Para modelos GGUF, precisaríamos de llama-cpp-python
            # Por enquanto, apenas registra as informações
            gguf_files = list(model_dir.glob("*.gguf"))
            
            if not gguf_files:
                return False, "Nenhum arquivo GGUF encontrado", None
            
            # Armazena informações básicas
            self._loaded_models[model_name] = {
                'model_file': str(gguf_files[0]),
                'type': 'gguf',
                'loaded': False  # Indica que precisa de implementação específica
            }
            
            self.logger.info(f"Modelo GGUF {model_name} registrado (implementação pendente)")
            return True, "Modelo GGUF registrado (suporte limitado)", self._loaded_models[model_name]
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo GGUF: {e}")
            return False, f"Erro ao carregar modelo GGUF: {str(e)}", None
    
    def unload_model(self, model_name: str) -> Tuple[bool, str]:
        """
        Remove modelo da memória.
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            if model_name in self._loaded_models:
                # Limpa referências
                del self._loaded_models[model_name]
                
                # Força garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.logger.info(f"Modelo {model_name} removido da memória")
                return True, f"Modelo {model_name} removido da memória"
            else:
                return False, f"Modelo {model_name} não estava carregado"
                
        except Exception as e:
            self.logger.error(f"Erro ao remover modelo da memória: {e}")
            return False, f"Erro ao remover modelo: {str(e)}"
    
    def delete_model(self, model_name: str) -> Tuple[bool, str]:
        """
        Remove modelo do disco.
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            # Remove da memória primeiro
            self.unload_model(model_name)
            
            # Remove do disco
            model_dir = self.models_dir / model_name
            if model_dir.exists():
                shutil.rmtree(model_dir)
                self.logger.info(f"Modelo {model_name} removido do disco")
                return True, f"Modelo {model_name} removido com sucesso"
            else:
                return False, f"Modelo {model_name} não encontrado no disco"
                
        except Exception as e:
            self.logger.error(f"Erro ao remover modelo do disco: {e}")
            return False, f"Erro ao remover modelo: {str(e)}"
    
    def get_loaded_models(self) -> List[str]:
        """
        Retorna lista de modelos carregados na memória.
        
        Returns:
            Lista com nomes dos modelos carregados
        """
        return list(self._loaded_models.keys())
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retorna configuração de um modelo carregado.
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            Configuração do modelo ou None
        """
        if model_name in self._loaded_models:
            return self._loaded_models[model_name].get('config')
        return None

