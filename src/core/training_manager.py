#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerenciador de treinamento para o NeuralTrain Forge.
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import yaml
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
    Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
from ..utils.logging_utils import LoggerMixin
from .model_manager import ModelManager
from .dataset_manager import DatasetManager


class TrainingManager(LoggerMixin):
    """Gerenciador de treinamento de modelos."""
    
    def __init__(self):
        self.config = self._load_config()
        self.model_manager = ModelManager()
        self.dataset_manager = DatasetManager()
        
        self.outputs_dir = Path(self.config.get('directories', {}).get('outputs', 'data/outputs'))
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Estado do treinamento
        self._training_state = {
            'is_training': False,
            'current_job': None,
            'progress': 0,
            'logs': [],
            'metrics': {}
        }
        
        # Thread de treinamento
        self._training_thread = None
        self._stop_training = False
        
        # Callbacks para atualização de progresso
        self._progress_callbacks = []
        
    def _load_config(self) -> dict:
        """Carrega configuração da aplicação."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Arquivo de configuração não encontrado: {config_path}")
            return {}
    
    def _load_training_config(self) -> dict:
        """Carrega configuração específica de treinamento."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "training" / "default.yaml"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Arquivo de configuração de treinamento não encontrado: {config_path}")
            return {}
    
    def _load_model_config(self) -> dict:
        """Carrega configuração específica de modelos."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "models" / "default.yaml"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Arquivo de configuração de modelos não encontrado: {config_path}")
            return {}
    
    def create_training_job(self, job_config: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
        """
        Cria um novo job de treinamento.
        
        Args:
            job_config: Configuração do job de treinamento
            
        Returns:
            Tupla (sucesso, mensagem, job_id)
        """
        try:
            # Valida configuração
            required_fields = ['model_name', 'dataset_name', 'job_name']
            for field in required_fields:
                if field not in job_config:
                    return False, f"Campo obrigatório ausente: {field}", None
            
            # Gera ID único para o job
            job_id = f"{job_config['job_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            job_dir = self.outputs_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            
            # Configuração completa do job
            full_config = {
                'job_id': job_id,
                'job_name': job_config['job_name'],
                'model_name': job_config['model_name'],
                'dataset_name': job_config['dataset_name'],
                'created': datetime.now().isoformat(),
                'status': 'created',
                'output_dir': str(job_dir),
                
                # Configurações de treinamento
                'training': self._merge_training_config(job_config.get('training', {})),
                
                # Configurações de LoRA
                'lora': self._merge_lora_config(job_config.get('lora', {})),
                
                # Configurações do dataset
                'dataset': job_config.get('dataset', {}),
                
                # Configurações do modelo
                'model': job_config.get('model', {})
            }
            
            # Salva configuração do job
            config_file = job_dir / "job_config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(full_config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Job de treinamento {job_id} criado com sucesso")
            return True, f"Job {job_id} criado com sucesso", job_id
            
        except Exception as e:
            self.logger.error(f"Erro ao criar job de treinamento: {e}")
            return False, f"Erro ao criar job: {str(e)}", None
    
    def _merge_training_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Mescla configuração do usuário com padrões de treinamento."""
        default_config = self._load_training_config().get('training', {})
        
        # Mescla configurações
        merged = default_config.copy()
        merged.update(user_config)
        
        return merged
    
    def _merge_lora_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Mescla configuração do usuário com padrões de LoRA."""
        default_config = self._load_model_config().get('lora', {})
        
        # Mescla configurações
        merged = default_config.copy()
        merged.update(user_config)
        
        return merged
    
    def start_training(self, job_id: str) -> Tuple[bool, str]:
        """
        Inicia treinamento de um job.
        
        Args:
            job_id: ID do job de treinamento
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            # Verifica se já está treinando
            if self._training_state['is_training']:
                return False, "Já existe um treinamento em andamento"
            
            # Carrega configuração do job
            job_dir = self.outputs_dir / job_id
            config_file = job_dir / "job_config.json"
            
            if not config_file.exists():
                return False, f"Job {job_id} não encontrado"
            
            with open(config_file, 'r', encoding='utf-8') as f:
                job_config = json.load(f)
            
            # Atualiza estado
            self._training_state['is_training'] = True
            self._training_state['current_job'] = job_id
            self._training_state['progress'] = 0
            self._training_state['logs'] = []
            self._training_state['metrics'] = {}
            self._stop_training = False
            
            # Inicia thread de treinamento
            self._training_thread = threading.Thread(
                target=self._run_training,
                args=(job_config,),
                daemon=True
            )
            self._training_thread.start()
            
            self.logger.info(f"Treinamento do job {job_id} iniciado")
            return True, f"Treinamento do job {job_id} iniciado"
            
        except Exception as e:
            self._training_state['is_training'] = False
            self.logger.error(f"Erro ao iniciar treinamento: {e}")
            return False, f"Erro ao iniciar treinamento: {str(e)}"
    
    def _run_training(self, job_config: Dict[str, Any]):
        """
        Executa o treinamento em thread separada.
        
        Args:
            job_config: Configuração do job
        """
        job_id = job_config['job_id']
        
        try:
            self._log_training(f"Iniciando treinamento do job {job_id}")
            
            # 1. Carrega modelo
            self._log_training("Carregando modelo...")
            success, message, model_data = self._load_model_for_training(job_config)
            if not success:
                self._log_training(f"Erro ao carregar modelo: {message}")
                return
            
            model = model_data['model']
            tokenizer = model_data['tokenizer']
            
            # 2. Carrega e preprocessa dataset
            self._log_training("Carregando dataset...")
            success, message, dataset = self._load_dataset_for_training(job_config, tokenizer)
            if not success:
                self._log_training(f"Erro ao carregar dataset: {message}")
                return
            
            # 3. Configura LoRA
            self._log_training("Configurando LoRA...")
            success, message, peft_model = self._setup_lora(model, job_config)
            if not success:
                self._log_training(f"Erro ao configurar LoRA: {message}")
                return
            
            # 4. Configura treinamento
            self._log_training("Configurando treinamento...")
            training_args = self._create_training_arguments(job_config)
            trainer = self._create_trainer(peft_model, tokenizer, dataset, training_args)
            
            # 5. Executa treinamento
            self._log_training("Iniciando treinamento...")
            self._execute_training(trainer, job_config)
            
            # 6. Salva modelo final
            self._log_training("Salvando modelo...")
            self._save_final_model(trainer, job_config)
            
            self._log_training("Treinamento concluído com sucesso!")
            
        except Exception as e:
            self.logger.error(f"Erro durante treinamento: {e}")
            self._log_training(f"Erro durante treinamento: {str(e)}")
        finally:
            self._training_state['is_training'] = False
            self._training_state['current_job'] = None
    
    def _load_model_for_training(self, job_config: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Carrega modelo para treinamento."""
        try:
            model_name = job_config['model_name']
            
            # Carrega modelo usando o ModelManager
            success, message, model_data = self.model_manager.load_model(model_name)
            if not success:
                return False, message, None
            
            return True, "Modelo carregado com sucesso", model_data
            
        except Exception as e:
            return False, f"Erro ao carregar modelo: {str(e)}", None
    
    def _load_dataset_for_training(self, job_config: Dict[str, Any], tokenizer) -> Tuple[bool, str, Optional[Dataset]]:
        """Carrega e preprocessa dataset para treinamento."""
        try:
            dataset_name = job_config['dataset_name']
            dataset_config = job_config.get('dataset', {})
            
            # Carrega dataset
            success, message, dataset = self.dataset_manager.load_dataset(dataset_name)
            if not success:
                return False, message, None
            
            # Preprocessa dataset
            text_column = dataset_config.get('text_column', 'text')
            max_length = dataset_config.get('max_length', 512)
            
            success, message, processed_dataset = self.dataset_manager.preprocess_dataset(
                dataset_name, 
                tokenizer.name_or_path if hasattr(tokenizer, 'name_or_path') else 'unknown',
                text_column,
                max_length
            )
            
            if not success:
                return False, message, None
            
            # Divide dataset se necessário
            train_ratio = dataset_config.get('train_ratio', 0.9)
            if train_ratio < 1.0:
                success, message, splits = self.dataset_manager.split_dataset(
                    f"{dataset_name}_preprocessed",
                    train_ratio=train_ratio,
                    test_ratio=(1.0 - train_ratio) / 2,
                    val_ratio=(1.0 - train_ratio) / 2
                )
                
                if success:
                    return True, "Dataset preprocessado e dividido", splits['train']
            
            return True, "Dataset preprocessado", processed_dataset
            
        except Exception as e:
            return False, f"Erro ao processar dataset: {str(e)}", None
    
    def _setup_lora(self, model, job_config: Dict[str, Any]) -> Tuple[bool, str, Optional[Any]]:
        """Configura LoRA no modelo."""
        try:
            lora_config = job_config.get('lora', {})
            
            # Configuração LoRA
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_config.get('r', 8),
                lora_alpha=lora_config.get('lora_alpha', 32),
                lora_dropout=lora_config.get('lora_dropout', 0.1),
                target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
                bias=lora_config.get('bias', "none")
            )
            
            # Aplica LoRA ao modelo
            peft_model = get_peft_model(model, peft_config)
            
            # Log de parâmetros treináveis
            trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in peft_model.parameters())
            
            self._log_training(f"Parâmetros treináveis: {trainable_params:,}")
            self._log_training(f"Total de parâmetros: {total_params:,}")
            self._log_training(f"Percentual treinável: {100 * trainable_params / total_params:.2f}%")
            
            return True, "LoRA configurado com sucesso", peft_model
            
        except Exception as e:
            return False, f"Erro ao configurar LoRA: {str(e)}", None
    
    def _create_training_arguments(self, job_config: Dict[str, Any]) -> TrainingArguments:
        """Cria argumentos de treinamento."""
        training_config = job_config.get('training', {})
        output_dir = job_config['output_dir']
        
        return TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=training_config.get('overwrite_output_dir', True),
            
            # Configurações básicas
            num_train_epochs=training_config.get('num_train_epochs', 3),
            per_device_train_batch_size=training_config.get('per_device_train_batch_size', 4),
            per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 4),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
            
            # Configurações de otimização
            learning_rate=training_config.get('learning_rate', 2e-4),
            weight_decay=training_config.get('weight_decay', 0.01),
            adam_beta1=training_config.get('adam_beta1', 0.9),
            adam_beta2=training_config.get('adam_beta2', 0.999),
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            
            # Configurações de scheduler
            lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
            warmup_ratio=training_config.get('warmup_ratio', 0.03),
            
            # Configurações de logging e salvamento
            logging_dir=f"{output_dir}/logs",
            logging_steps=training_config.get('logging_steps', 10),
            save_strategy=training_config.get('save_strategy', 'steps'),
            save_steps=training_config.get('save_steps', 500),
            save_total_limit=training_config.get('save_total_limit', 3),
            
            # Configurações de avaliação
            evaluation_strategy=training_config.get('evaluation_strategy', 'steps'),
            eval_steps=training_config.get('eval_steps', 500),
            
            # Configurações de hardware
            fp16=training_config.get('fp16', True),
            gradient_checkpointing=training_config.get('gradient_checkpointing', True),
            dataloader_num_workers=training_config.get('dataloader_num_workers', 4),
            
            # Outras configurações
            remove_unused_columns=training_config.get('remove_unused_columns', False),
            load_best_model_at_end=training_config.get('load_best_model_at_end', True),
            metric_for_best_model=training_config.get('metric_for_best_model', 'eval_loss'),
            greater_is_better=training_config.get('greater_is_better', False),
            
            # Relatórios
            report_to=training_config.get('report_to', [])
        )
    
    def _create_trainer(self, model, tokenizer, dataset, training_args) -> Trainer:
        """Cria objeto Trainer."""
        # Data collator para modelos de linguagem
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Não é masked language modeling
        )
        
        # Callback personalizado para progresso
        class ProgressCallback:
            def __init__(self, training_manager):
                self.training_manager = training_manager
            
            def on_log(self, args, state, control, model=None, logs=None, **kwargs):
                if logs:
                    # Atualiza progresso
                    if state.max_steps > 0:
                        progress = (state.global_step / state.max_steps) * 100
                        self.training_manager._training_state['progress'] = progress
                    
                    # Adiciona logs
                    log_entry = {
                        'step': state.global_step,
                        'epoch': state.epoch,
                        'logs': logs,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.training_manager._training_state['logs'].append(log_entry)
                    self.training_manager._training_state['metrics'].update(logs)
                    
                    # Chama callbacks de progresso
                    for callback in self.training_manager._progress_callbacks:
                        callback(progress if state.max_steps > 0 else 0, logs)
        
        # Cria trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[ProgressCallback(self)]
        )
        
        return trainer
    
    def _execute_training(self, trainer, job_config: Dict[str, Any]):
        """Executa o treinamento."""
        try:
            # Inicia treinamento
            trainer.train()
            
            # Salva métricas finais
            metrics = trainer.state.log_history
            metrics_file = Path(job_config['output_dir']) / "training_metrics.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            self.logger.error(f"Erro durante execução do treinamento: {e}")
            raise
    
    def _save_final_model(self, trainer, job_config: Dict[str, Any]):
        """Salva modelo final."""
        try:
            output_dir = Path(job_config['output_dir'])
            
            # Salva modelo
            trainer.save_model(str(output_dir / "final_model"))
            
            # Salva tokenizer
            trainer.tokenizer.save_pretrained(str(output_dir / "final_model"))
            
            # Salva informações do job
            job_info = {
                'job_id': job_config['job_id'],
                'completed': datetime.now().isoformat(),
                'final_metrics': self._training_state['metrics'],
                'model_path': str(output_dir / "final_model")
            }
            
            info_file = output_dir / "job_info.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(job_info, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo final: {e}")
            raise
    
    def _log_training(self, message: str):
        """Adiciona log de treinamento."""
        log_entry = {
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self._training_state['logs'].append(log_entry)
        self.logger.info(f"[TRAINING] {message}")
    
    def stop_training(self) -> Tuple[bool, str]:
        """
        Para o treinamento atual.
        
        Returns:
            Tupla (sucesso, mensagem)
        """
        try:
            if not self._training_state['is_training']:
                return False, "Nenhum treinamento em andamento"
            
            self._stop_training = True
            self._log_training("Solicitação de parada recebida...")
            
            return True, "Solicitação de parada enviada"
            
        except Exception as e:
            self.logger.error(f"Erro ao parar treinamento: {e}")
            return False, f"Erro ao parar treinamento: {str(e)}"
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        Retorna status atual do treinamento.
        
        Returns:
            Dicionário com status do treinamento
        """
        return self._training_state.copy()
    
    def list_training_jobs(self) -> List[Dict[str, Any]]:
        """
        Lista todos os jobs de treinamento.
        
        Returns:
            Lista de jobs
        """
        jobs = []
        
        try:
            for job_dir in self.outputs_dir.iterdir():
                if job_dir.is_dir():
                    config_file = job_dir / "job_config.json"
                    if config_file.exists():
                        with open(config_file, 'r', encoding='utf-8') as f:
                            job_config = json.load(f)
                            
                        # Adiciona informações de status
                        info_file = job_dir / "job_info.json"
                        if info_file.exists():
                            with open(info_file, 'r', encoding='utf-8') as f:
                                job_info = json.load(f)
                                job_config.update(job_info)
                        
                        jobs.append(job_config)
        except Exception as e:
            self.logger.error(f"Erro ao listar jobs: {e}")
        
        return sorted(jobs, key=lambda x: x.get('created', ''), reverse=True)
    
    def add_progress_callback(self, callback: Callable[[float, Dict[str, Any]], None]):
        """
        Adiciona callback para atualizações de progresso.
        
        Args:
            callback: Função que recebe (progresso, métricas)
        """
        self._progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable[[float, Dict[str, Any]], None]):
        """
        Remove callback de progresso.
        
        Args:
            callback: Função a ser removida
        """
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)

