#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuralTrain Forge - Chat Manager
Sistema de chat para teste e avaliação de modelos treinados

Funcionalidades:
- Chat conversacional com modelos
- Comparação lado a lado (duelo de modelos)
- Métricas de performance em tempo real
- Histórico de conversas
- Exportação de logs
- Avaliação automática de qualidade
"""

import os
import json
import time
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TextStreamer, BitsAndBytesConfig
)
from peft import PeftModel
import psutil
import threading
from queue import Queue

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Estrutura de uma mensagem de chat"""
    id: str
    timestamp: datetime
    role: str  # 'user', 'assistant', 'system'
    content: str
    model_name: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'role': self.role,
            'content': self.content,
            'model_name': self.model_name,
            'metadata': self.metadata or {}
        }

@dataclass
class ModelMetrics:
    """Métricas de performance do modelo"""
    model_name: str
    tokens_generated: int
    generation_time: float
    tokens_per_second: float
    memory_usage: float
    gpu_usage: float
    temperature: float
    max_tokens: int
    response_length: int
    
    def to_dict(self):
        return asdict(self)

@dataclass
class ChatSession:
    """Sessão de chat"""
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    messages: List[ChatMessage]
    models_used: List[str]
    total_tokens: int
    session_type: str  # 'single', 'comparison', 'evaluation'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'messages': [msg.to_dict() for msg in self.messages],
            'models_used': self.models_used,
            'total_tokens': self.total_tokens,
            'session_type': self.session_type
        }

class ModelLoader:
    """Carregador e gerenciador de modelos"""
    
    def __init__(self, models_dir: str = "data/models"):
        self.models_dir = Path(models_dir)
        self.loaded_models: Dict[str, Dict] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def list_available_models(self) -> List[Dict[str, str]]:
        """Lista modelos disponíveis"""
        models = []
        
        # Modelos locais
        if self.models_dir.exists():
            for model_path in self.models_dir.iterdir():
                if model_path.is_dir():
                    models.append({
                        'name': model_path.name,
                        'path': str(model_path),
                        'type': 'local',
                        'size': self._get_model_size(model_path)
                    })
        
        # Modelos carregados na memória
        for model_name in self.loaded_models:
            if not any(m['name'] == model_name for m in models):
                models.append({
                    'name': model_name,
                    'path': 'memory',
                    'type': 'loaded',
                    'size': 'N/A'
                })
        
        return models
    
    def _get_model_size(self, model_path: Path) -> str:
        """Calcula tamanho do modelo"""
        try:
            total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            return f"{total_size / (1024**3):.2f} GB"
        except:
            return "Unknown"
    
    def load_model(self, model_name: str, model_path: str = None, 
                   load_in_8bit: bool = False, load_in_4bit: bool = False) -> bool:
        """Carrega modelo na memória"""
        try:
            if model_name in self.loaded_models:
                logger.info(f"Modelo {model_name} já está carregado")
                return True
            
            logger.info(f"Carregando modelo {model_name}...")
            
            # Configurar quantização se necessário
            quantization_config = None
            if load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Carregar tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path or model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Carregar modelo
            model = AutoModelForCausalLM.from_pretrained(
                model_path or model_name,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            # Verificar se é modelo PEFT
            if model_path and (Path(model_path) / "adapter_config.json").exists():
                logger.info("Detectado modelo PEFT, carregando adaptadores...")
                model = PeftModel.from_pretrained(model, model_path)
            
            self.loaded_models[model_name] = {
                'model': model,
                'tokenizer': tokenizer,
                'path': model_path or model_name,
                'loaded_at': datetime.now(),
                'quantization': '4bit' if load_in_4bit else '8bit' if load_in_8bit else 'none'
            }
            
            logger.info(f"Modelo {model_name} carregado com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo {model_name}: {e}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """Descarrega modelo da memória"""
        try:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Modelo {model_name} descarregado")
                return True
            return False
        except Exception as e:
            logger.error(f"Erro ao descarregar modelo {model_name}: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Obtém informações do modelo"""
        if model_name in self.loaded_models:
            info = self.loaded_models[model_name].copy()
            info['loaded_at'] = info['loaded_at'].isoformat()
            del info['model']  # Não serializar o modelo
            del info['tokenizer']  # Não serializar o tokenizer
            return info
        return None

class ChatEngine:
    """Motor de chat para geração de respostas"""
    
    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
        self.generation_queue = Queue()
        self.metrics_history: List[ModelMetrics] = []
        
    def generate_response(self, model_name: str, messages: List[ChatMessage],
                         temperature: float = 0.7, max_tokens: int = 512,
                         top_p: float = 0.9, top_k: int = 50,
                         stream: bool = False) -> Tuple[str, ModelMetrics]:
        """Gera resposta do modelo"""
        
        if model_name not in self.model_loader.loaded_models:
            raise ValueError(f"Modelo {model_name} não está carregado")
        
        model_info = self.model_loader.loaded_models[model_name]
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        
        # Preparar prompt
        prompt = self._format_messages(messages, tokenizer)
        
        # Métricas iniciais
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        initial_gpu = self._get_gpu_usage()
        
        try:
            # Tokenizar entrada
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Configurar geração
            generation_config = {
                'max_new_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'do_sample': True,
                'pad_token_id': tokenizer.eos_token_id,
                'eos_token_id': tokenizer.eos_token_id,
                'repetition_penalty': 1.1
            }
            
            # Gerar resposta
            with torch.no_grad():
                if stream:
                    # Implementar streaming se necessário
                    outputs = model.generate(**inputs, **generation_config)
                else:
                    outputs = model.generate(**inputs, **generation_config)
            
            # Decodificar resposta
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Calcular métricas
            end_time = time.time()
            generation_time = end_time - start_time
            tokens_generated = len(outputs[0]) - inputs['input_ids'].shape[1]
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            metrics = ModelMetrics(
                model_name=model_name,
                tokens_generated=tokens_generated,
                generation_time=generation_time,
                tokens_per_second=tokens_per_second,
                memory_usage=self._get_memory_usage() - initial_memory,
                gpu_usage=self._get_gpu_usage(),
                temperature=temperature,
                max_tokens=max_tokens,
                response_length=len(response)
            )
            
            self.metrics_history.append(metrics)
            
            return response, metrics
            
        except Exception as e:
            logger.error(f"Erro na geração: {e}")
            raise
    
    def _format_messages(self, messages: List[ChatMessage], tokenizer) -> str:
        """Formata mensagens para o prompt"""
        # Formato básico - pode ser customizado por modelo
        formatted = ""
        for msg in messages:
            if msg.role == "user":
                formatted += f"Human: {msg.content}\n"
            elif msg.role == "assistant":
                formatted += f"Assistant: {msg.content}\n"
            elif msg.role == "system":
                formatted += f"System: {msg.content}\n"
        
        formatted += "Assistant: "
        return formatted
    
    def _get_memory_usage(self) -> float:
        """Obtém uso de memória RAM"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0
    
    def _get_gpu_usage(self) -> float:
        """Obtém uso de GPU"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024  # MB
            return 0.0
        except:
            return 0.0

class ChatManager:
    """Gerenciador principal do sistema de chat"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.sessions_dir = self.data_dir / "chat_sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_loader = ModelLoader(str(self.data_dir / "models"))
        self.chat_engine = ChatEngine(self.model_loader)
        
        self.active_sessions: Dict[str, ChatSession] = {}
        self.load_sessions()
    
    def create_session(self, name: str, session_type: str = "single") -> str:
        """Cria nova sessão de chat"""
        session_id = str(uuid.uuid4())
        session = ChatSession(
            id=session_id,
            name=name,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            messages=[],
            models_used=[],
            total_tokens=0,
            session_type=session_type
        )
        
        self.active_sessions[session_id] = session
        self.save_session(session_id)
        
        logger.info(f"Sessão criada: {name} ({session_id})")
        return session_id
    
    def send_message(self, session_id: str, content: str, model_name: str,
                    temperature: float = 0.7, max_tokens: int = 512) -> Dict[str, Any]:
        """Envia mensagem e obtém resposta"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Sessão {session_id} não encontrada")
        
        session = self.active_sessions[session_id]
        
        # Adicionar mensagem do usuário
        user_message = ChatMessage(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            role="user",
            content=content,
            model_name="user"
        )
        session.messages.append(user_message)
        
        try:
            # Gerar resposta
            response, metrics = self.chat_engine.generate_response(
                model_name=model_name,
                messages=session.messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Adicionar resposta do assistente
            assistant_message = ChatMessage(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                role="assistant",
                content=response,
                model_name=model_name,
                metadata=metrics.to_dict()
            )
            session.messages.append(assistant_message)
            
            # Atualizar sessão
            if model_name not in session.models_used:
                session.models_used.append(model_name)
            
            session.total_tokens += metrics.tokens_generated
            session.updated_at = datetime.now()
            
            # Salvar sessão
            self.save_session(session_id)
            
            return {
                'message': assistant_message.to_dict(),
                'metrics': metrics.to_dict(),
                'session_updated': session.updated_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro ao processar mensagem: {e}")
            raise
    
    def compare_models(self, session_id: str, content: str, 
                      model_names: List[str], **generation_params) -> Dict[str, Any]:
        """Compara respostas de múltiplos modelos"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Sessão {session_id} não encontrada")
        
        session = self.active_sessions[session_id]
        session.session_type = "comparison"
        
        # Adicionar mensagem do usuário
        user_message = ChatMessage(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            role="user",
            content=content,
            model_name="user"
        )
        session.messages.append(user_message)
        
        responses = {}
        
        for model_name in model_names:
            try:
                response, metrics = self.chat_engine.generate_response(
                    model_name=model_name,
                    messages=session.messages[:-1],  # Excluir mensagens de outros modelos
                    **generation_params
                )
                
                assistant_message = ChatMessage(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    role="assistant",
                    content=response,
                    model_name=model_name,
                    metadata=metrics.to_dict()
                )
                
                responses[model_name] = {
                    'message': assistant_message.to_dict(),
                    'metrics': metrics.to_dict()
                }
                
                session.messages.append(assistant_message)
                
                if model_name not in session.models_used:
                    session.models_used.append(model_name)
                
                session.total_tokens += metrics.tokens_generated
                
            except Exception as e:
                logger.error(f"Erro com modelo {model_name}: {e}")
                responses[model_name] = {
                    'error': str(e)
                }
        
        session.updated_at = datetime.now()
        self.save_session(session_id)
        
        return {
            'user_message': user_message.to_dict(),
            'responses': responses,
            'session_updated': session.updated_at.isoformat()
        }
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Obtém dados da sessão"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id].to_dict()
        return None
    
    def list_sessions(self) -> List[Dict]:
        """Lista todas as sessões"""
        sessions = []
        for session in self.active_sessions.values():
            sessions.append({
                'id': session.id,
                'name': session.name,
                'created_at': session.created_at.isoformat(),
                'updated_at': session.updated_at.isoformat(),
                'message_count': len(session.messages),
                'models_used': session.models_used,
                'session_type': session.session_type
            })
        return sorted(sessions, key=lambda x: x['updated_at'], reverse=True)
    
    def delete_session(self, session_id: str) -> bool:
        """Deleta sessão"""
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            session_file = self.sessions_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            
            logger.info(f"Sessão {session_id} deletada")
            return True
        except Exception as e:
            logger.error(f"Erro ao deletar sessão: {e}")
            return False
    
    def export_session(self, session_id: str, format: str = "json") -> Optional[str]:
        """Exporta sessão para arquivo"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            filename = f"chat_export_{session.name}_{timestamp}.json"
            filepath = self.data_dir / "exports" / filename
            filepath.parent.mkdir(exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
            
            return str(filepath)
        
        elif format == "jsonl":
            filename = f"chat_export_{session.name}_{timestamp}.jsonl"
            filepath = self.data_dir / "exports" / filename
            filepath.parent.mkdir(exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                for message in session.messages:
                    f.write(json.dumps(message.to_dict(), ensure_ascii=False) + '\n')
            
            return str(filepath)
        
        return None
    
    def save_session(self, session_id: str):
        """Salva sessão em arquivo"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session_file = self.sessions_dir / f"{session_id}.json"
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
    
    def load_sessions(self):
        """Carrega sessões salvas"""
        try:
            for session_file in self.sessions_dir.glob("*.json"):
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Reconstruir objetos
                messages = []
                for msg_data in data['messages']:
                    msg_data['timestamp'] = datetime.fromisoformat(msg_data['timestamp'])
                    messages.append(ChatMessage(**msg_data))
                
                session = ChatSession(
                    id=data['id'],
                    name=data['name'],
                    created_at=datetime.fromisoformat(data['created_at']),
                    updated_at=datetime.fromisoformat(data['updated_at']),
                    messages=messages,
                    models_used=data['models_used'],
                    total_tokens=data['total_tokens'],
                    session_type=data.get('session_type', 'single')
                )
                
                self.active_sessions[session.id] = session
                
        except Exception as e:
            logger.error(f"Erro ao carregar sessões: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Obtém resumo das métricas"""
        if not self.chat_engine.metrics_history:
            return {}
        
        metrics_by_model = {}
        for metric in self.chat_engine.metrics_history:
            if metric.model_name not in metrics_by_model:
                metrics_by_model[metric.model_name] = []
            metrics_by_model[metric.model_name].append(metric)
        
        summary = {}
        for model_name, metrics in metrics_by_model.items():
            summary[model_name] = {
                'total_generations': len(metrics),
                'avg_tokens_per_second': np.mean([m.tokens_per_second for m in metrics]),
                'avg_generation_time': np.mean([m.generation_time for m in metrics]),
                'total_tokens': sum([m.tokens_generated for m in metrics]),
                'avg_response_length': np.mean([m.response_length for m in metrics])
            }
        
        return summary

