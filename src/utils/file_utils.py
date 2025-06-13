#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitários para manipulação de arquivos no NeuralTrain Forge.
"""

import os
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import yaml
import json
import pandas as pd
from .logging_utils import LoggerMixin


class FileUtils(LoggerMixin):
    """Utilitários para manipulação de arquivos."""
    
    def __init__(self):
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """Carrega configuração da aplicação."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Arquivo de configuração não encontrado: {config_path}")
            return {}
    
    def validate_file_size(self, file_path: str, max_size_mb: Optional[int] = None) -> bool:
        """
        Valida o tamanho do arquivo.
        
        Args:
            file_path: Caminho para o arquivo
            max_size_mb: Tamanho máximo em MB. Se None, usa configuração padrão
            
        Returns:
            True se o arquivo está dentro do limite
        """
        if max_size_mb is None:
            max_size_mb = self.config.get('upload', {}).get('max_file_size_mb', 1000)
        
        try:
            file_size = os.path.getsize(file_path)
            max_size_bytes = max_size_mb * 1024 * 1024
            
            if file_size > max_size_bytes:
                self.logger.warning(f"Arquivo muito grande: {file_size} bytes (máximo: {max_size_bytes})")
                return False
            
            return True
        except OSError as e:
            self.logger.error(f"Erro ao verificar tamanho do arquivo {file_path}: {e}")
            return False
    
    def validate_file_format(self, file_path: str, allowed_formats: List[str]) -> bool:
        """
        Valida o formato do arquivo.
        
        Args:
            file_path: Caminho para o arquivo
            allowed_formats: Lista de extensões permitidas (ex: ['.txt', '.json'])
            
        Returns:
            True se o formato é permitido
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in allowed_formats:
            self.logger.warning(f"Formato não permitido: {file_ext}. Permitidos: {allowed_formats}")
            return False
        
        return True
    
    def get_file_hash(self, file_path: str, algorithm: str = 'sha256') -> str:
        """
        Calcula hash do arquivo.
        
        Args:
            file_path: Caminho para o arquivo
            algorithm: Algoritmo de hash (md5, sha1, sha256)
            
        Returns:
            Hash do arquivo em hexadecimal
        """
        hash_obj = hashlib.new(algorithm)
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
        except OSError as e:
            self.logger.error(f"Erro ao calcular hash do arquivo {file_path}: {e}")
            return ""
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Obtém informações detalhadas do arquivo.
        
        Args:
            file_path: Caminho para o arquivo
            
        Returns:
            Dicionário com informações do arquivo
        """
        path = Path(file_path)
        
        try:
            stat = path.stat()
            mime_type, _ = mimetypes.guess_type(file_path)
            
            return {
                'name': path.name,
                'size': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'extension': path.suffix.lower(),
                'mime_type': mime_type,
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'hash': self.get_file_hash(file_path),
                'exists': path.exists(),
                'is_file': path.is_file()
            }
        except OSError as e:
            self.logger.error(f"Erro ao obter informações do arquivo {file_path}: {e}")
            return {}
    
    def safe_filename(self, filename: str) -> str:
        """
        Cria um nome de arquivo seguro removendo caracteres problemáticos.
        
        Args:
            filename: Nome original do arquivo
            
        Returns:
            Nome de arquivo seguro
        """
        # Remove caracteres problemáticos
        unsafe_chars = '<>:"/\\|?*'
        safe_name = filename
        
        for char in unsafe_chars:
            safe_name = safe_name.replace(char, '_')
        
        # Remove espaços extras e pontos no início/fim
        safe_name = safe_name.strip(' .')
        
        # Limita o tamanho
        if len(safe_name) > 255:
            name, ext = os.path.splitext(safe_name)
            safe_name = name[:255-len(ext)] + ext
        
        return safe_name
    
    def ensure_directory(self, directory: str) -> bool:
        """
        Garante que o diretório existe, criando se necessário.
        
        Args:
            directory: Caminho do diretório
            
        Returns:
            True se o diretório existe ou foi criado com sucesso
        """
        try:
            os.makedirs(directory, exist_ok=True)
            return True
        except OSError as e:
            self.logger.error(f"Erro ao criar diretório {directory}: {e}")
            return False
    
    def validate_model_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Valida arquivo de modelo.
        
        Args:
            file_path: Caminho para o arquivo do modelo
            
        Returns:
            Tupla (é_válido, mensagem)
        """
        allowed_formats = self.config.get('upload', {}).get('allowed_model_formats', ['.gguf', '.bin', '.safetensors'])
        
        # Verifica se o arquivo existe
        if not os.path.exists(file_path):
            return False, "Arquivo não encontrado"
        
        # Verifica formato
        if not self.validate_file_format(file_path, allowed_formats):
            return False, f"Formato não suportado. Permitidos: {', '.join(allowed_formats)}"
        
        # Verifica tamanho
        if not self.validate_file_size(file_path):
            max_size = self.config.get('upload', {}).get('max_file_size_mb', 1000)
            return False, f"Arquivo muito grande (máximo: {max_size}MB)"
        
        return True, "Arquivo válido"
    
    def validate_dataset_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Valida arquivo de dataset.
        
        Args:
            file_path: Caminho para o arquivo do dataset
            
        Returns:
            Tupla (é_válido, mensagem)
        """
        allowed_formats = self.config.get('upload', {}).get('allowed_dataset_formats', ['.txt', '.jsonl', '.csv', '.parquet'])
        
        # Verifica se o arquivo existe
        if not os.path.exists(file_path):
            return False, "Arquivo não encontrado"
        
        # Verifica formato
        if not self.validate_file_format(file_path, allowed_formats):
            return False, f"Formato não suportado. Permitidos: {', '.join(allowed_formats)}"
        
        # Verifica tamanho
        if not self.validate_file_size(file_path):
            max_size = self.config.get('upload', {}).get('max_file_size_mb', 1000)
            return False, f"Arquivo muito grande (máximo: {max_size}MB)"
        
        # Validação específica por formato
        try:
            ext = Path(file_path).suffix.lower()
            
            if ext == '.csv':
                # Tenta ler o CSV
                df = pd.read_csv(file_path, nrows=5)
                if df.empty:
                    return False, "Arquivo CSV vazio"
            
            elif ext == '.jsonl':
                # Tenta ler as primeiras linhas do JSONL
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 5:  # Verifica apenas as primeiras 5 linhas
                            break
                        try:
                            json.loads(line.strip())
                        except json.JSONDecodeError:
                            return False, f"Linha {i+1} não é um JSON válido"
            
            elif ext == '.parquet':
                # Tenta ler o Parquet
                df = pd.read_parquet(file_path)
                if df.empty:
                    return False, "Arquivo Parquet vazio"
            
        except Exception as e:
            return False, f"Erro ao validar conteúdo do arquivo: {str(e)}"
        
        return True, "Arquivo válido"
    
    def get_dataset_preview(self, file_path: str, max_rows: int = 10) -> Dict[str, Any]:
        """
        Obtém preview do dataset.
        
        Args:
            file_path: Caminho para o arquivo do dataset
            max_rows: Número máximo de linhas para preview
            
        Returns:
            Dicionário com informações do preview
        """
        try:
            ext = Path(file_path).suffix.lower()
            
            if ext == '.csv':
                df = pd.read_csv(file_path, nrows=max_rows)
                return {
                    'format': 'csv',
                    'rows': len(df),
                    'columns': list(df.columns),
                    'preview': df.to_dict('records'),
                    'total_rows': sum(1 for _ in open(file_path, 'r', encoding='utf-8')) - 1  # -1 para header
                }
            
            elif ext == '.jsonl':
                lines = []
                total_lines = 0
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        total_lines += 1
                        if i < max_rows:
                            try:
                                lines.append(json.loads(line.strip()))
                            except json.JSONDecodeError:
                                continue
                
                return {
                    'format': 'jsonl',
                    'rows': len(lines),
                    'preview': lines,
                    'total_rows': total_lines
                }
            
            elif ext == '.parquet':
                df = pd.read_parquet(file_path)
                preview_df = df.head(max_rows)
                
                return {
                    'format': 'parquet',
                    'rows': len(preview_df),
                    'columns': list(df.columns),
                    'preview': preview_df.to_dict('records'),
                    'total_rows': len(df)
                }
            
            elif ext == '.txt':
                lines = []
                total_lines = 0
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        total_lines += 1
                        if i < max_rows:
                            lines.append(line.strip())
                
                return {
                    'format': 'txt',
                    'rows': len(lines),
                    'preview': lines,
                    'total_rows': total_lines
                }
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar preview do dataset {file_path}: {e}")
            return {'error': str(e)}
        
        return {'error': 'Formato não suportado para preview'}

