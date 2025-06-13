#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitários de logging para o NeuralTrain Forge.
"""

import os
import logging
import logging.handlers
from typing import Optional
import yaml


def load_config() -> dict:
    """Carrega a configuração do arquivo YAML."""
    config_path = os.path.join(os.path.dirname(__file__), '../../configs/config.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Configuração padrão se o arquivo não existir
        return {
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/neuraltrain.log',
                'max_bytes': 10485760,
                'backup_count': 5
            }
        }


def setup_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Configura e retorna um logger para a aplicação.
    
    Args:
        name: Nome do logger. Se None, usa 'neuraltrain'
        
    Returns:
        Logger configurado
    """
    if name is None:
        name = 'neuraltrain'
    
    # Carrega configuração
    config = load_config()
    log_config = config.get('logging', {})
    
    # Cria logger
    logger = logging.getLogger(name)
    
    # Evita duplicação de handlers
    if logger.handlers:
        return logger
    
    # Configura nível
    level = getattr(logging, log_config.get('level', 'INFO').upper())
    logger.setLevel(level)
    
    # Formato das mensagens
    formatter = logging.Formatter(
        log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para arquivo (com rotação)
    log_file = log_config.get('file', 'logs/neuraltrain.log')
    log_dir = os.path.dirname(log_file)
    
    # Cria diretório de logs se não existir
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=log_config.get('max_bytes', 10485760),  # 10MB
        backupCount=log_config.get('backup_count', 5),
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Retorna um logger com o nome especificado.
    
    Args:
        name: Nome do logger
        
    Returns:
        Logger configurado
    """
    return setup_logger(name)


class LoggerMixin:
    """Mixin para adicionar logging a classes."""
    
    @property
    def logger(self) -> logging.Logger:
        """Retorna um logger para a classe."""
        return get_logger(self.__class__.__name__)


# Logger principal da aplicação
main_logger = setup_logger()

