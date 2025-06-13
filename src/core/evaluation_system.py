#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuralTrain Forge - Metrics and Evaluation System
Sistema de métricas e avaliação para modelos de linguagem

Funcionalidades:
- Métricas automáticas: BLEU, ROUGE, similaridade semântica
- Avaliação de qualidade de texto
- Comparação entre modelos
- Relatórios detalhados
- Métricas personalizadas
- Integração com LLM-as-a-Judge
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import logging
import re
import math
from collections import Counter

# Bibliotecas para métricas
try:
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize, sent_tokenize
    import nltk
    # Download necessário para NLTK
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Resultado de uma avaliação"""
    id: str
    model_name: str
    prompt: str
    reference: str
    generated: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class ComparisonResult:
    """Resultado de comparação entre modelos"""
    model_a: str
    model_b: str
    prompt: str
    response_a: str
    response_b: str
    metrics_a: Dict[str, float]
    metrics_b: Dict[str, float]
    winner: Optional[str]
    confidence: float
    timestamp: datetime
    
    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class MetricsCalculator:
    """Calculadora de métricas para avaliação de texto"""
    
    def __init__(self):
        self.rouge_scorer = None
        self.sentence_model = None
        self.smoothing_function = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Inicializa componentes de métricas"""
        
        # Inicializar ROUGE
        if ROUGE_AVAILABLE:
            try:
                self.rouge_scorer = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'], 
                    use_stemmer=True
                )
                logger.info("ROUGE scorer inicializado")
            except Exception as e:
                logger.warning(f"Erro ao inicializar ROUGE: {e}")
        
        # Inicializar BLEU smoothing
        if NLTK_AVAILABLE:
            self.smoothing_function = SmoothingFunction().method1
            logger.info("BLEU smoothing inicializado")
        
        # Inicializar modelo de embeddings
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence Transformer inicializado")
            except Exception as e:
                logger.warning(f"Erro ao inicializar Sentence Transformer: {e}")
    
    def calculate_bleu(self, reference: str, generated: str) -> Dict[str, float]:
        """Calcula métricas BLEU"""
        if not NLTK_AVAILABLE:
            return {"bleu_error": -1.0}
        
        try:
            # Tokenizar textos
            ref_tokens = word_tokenize(reference.lower())
            gen_tokens = word_tokenize(generated.lower())
            
            # BLEU-1 a BLEU-4
            bleu_scores = {}
            for n in range(1, 5):
                weights = [1.0/n] * n + [0.0] * (4-n)
                score = sentence_bleu(
                    [ref_tokens], 
                    gen_tokens, 
                    weights=weights,
                    smoothing_function=self.smoothing_function
                )
                bleu_scores[f'bleu_{n}'] = score
            
            # BLEU médio
            bleu_scores['bleu_avg'] = np.mean(list(bleu_scores.values()))
            
            return bleu_scores
            
        except Exception as e:
            logger.error(f"Erro ao calcular BLEU: {e}")
            return {"bleu_error": -1.0}
    
    def calculate_rouge(self, reference: str, generated: str) -> Dict[str, float]:
        """Calcula métricas ROUGE"""
        if not self.rouge_scorer:
            return {"rouge_error": -1.0}
        
        try:
            scores = self.rouge_scorer.score(reference, generated)
            
            rouge_metrics = {}
            for metric, score in scores.items():
                rouge_metrics[f'{metric}_precision'] = score.precision
                rouge_metrics[f'{metric}_recall'] = score.recall
                rouge_metrics[f'{metric}_fmeasure'] = score.fmeasure
            
            return rouge_metrics
            
        except Exception as e:
            logger.error(f"Erro ao calcular ROUGE: {e}")
            return {"rouge_error": -1.0}
    
    def calculate_semantic_similarity(self, reference: str, generated: str) -> Dict[str, float]:
        """Calcula similaridade semântica usando embeddings"""
        if not self.sentence_model:
            return {"semantic_error": -1.0}
        
        try:
            # Gerar embeddings
            embeddings = self.sentence_model.encode([reference, generated])
            
            # Calcular similaridade coseno
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return {
                'semantic_similarity': float(similarity),
                'semantic_distance': float(1 - similarity)
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular similaridade semântica: {e}")
            return {"semantic_error": -1.0}
    
    def calculate_lexical_metrics(self, reference: str, generated: str) -> Dict[str, float]:
        """Calcula métricas lexicais básicas"""
        try:
            # Tokenizar
            ref_words = reference.lower().split()
            gen_words = generated.lower().split()
            
            # Métricas básicas
            metrics = {
                'length_ratio': len(gen_words) / max(len(ref_words), 1),
                'word_overlap': len(set(ref_words) & set(gen_words)) / max(len(set(ref_words)), 1),
                'unique_words_ratio': len(set(gen_words)) / max(len(gen_words), 1),
                'repetition_ratio': 1 - (len(set(gen_words)) / max(len(gen_words), 1))
            }
            
            # Jaccard similarity
            ref_set = set(ref_words)
            gen_set = set(gen_words)
            jaccard = len(ref_set & gen_set) / max(len(ref_set | gen_set), 1)
            metrics['jaccard_similarity'] = jaccard
            
            # Levenshtein distance normalizada
            lev_distance = self._levenshtein_distance(reference, generated)
            max_len = max(len(reference), len(generated))
            metrics['normalized_levenshtein'] = 1 - (lev_distance / max(max_len, 1))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erro ao calcular métricas lexicais: {e}")
            return {"lexical_error": -1.0}
    
    def calculate_fluency_metrics(self, text: str) -> Dict[str, float]:
        """Calcula métricas de fluência do texto"""
        try:
            # Métricas básicas
            words = text.split()
            sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split('.')
            
            metrics = {
                'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
                'avg_sentence_length': np.mean([len(sent.split()) for sent in sentences]) if sentences else 0,
                'sentence_count': len(sentences),
                'word_count': len(words),
                'char_count': len(text)
            }
            
            # Diversidade lexical (TTR - Type-Token Ratio)
            unique_words = set(word.lower() for word in words)
            metrics['lexical_diversity'] = len(unique_words) / max(len(words), 1)
            
            # Complexidade sintática (aproximação)
            punctuation_count = sum(1 for char in text if char in '.,;:!?')
            metrics['syntactic_complexity'] = punctuation_count / max(len(sentences), 1)
            
            # Repetitividade
            word_freq = Counter(word.lower() for word in words)
            most_common_freq = word_freq.most_common(1)[0][1] if word_freq else 0
            metrics['repetitiveness'] = most_common_freq / max(len(words), 1)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erro ao calcular métricas de fluência: {e}")
            return {"fluency_error": -1.0}
    
    def calculate_coherence_score(self, text: str) -> Dict[str, float]:
        """Calcula pontuação de coerência do texto"""
        try:
            sentences = sent_tokenize(text) if NLTK_AVAILABLE else text.split('.')
            
            if len(sentences) < 2:
                return {'coherence_score': 1.0, 'coherence_variance': 0.0}
            
            # Calcular similaridade entre sentenças consecutivas
            if self.sentence_model:
                embeddings = self.sentence_model.encode(sentences)
                similarities = []
                
                for i in range(len(embeddings) - 1):
                    sim = np.dot(embeddings[i], embeddings[i+1]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
                    )
                    similarities.append(sim)
                
                coherence_score = np.mean(similarities)
                coherence_variance = np.var(similarities)
            else:
                # Fallback: similaridade lexical simples
                similarities = []
                for i in range(len(sentences) - 1):
                    words1 = set(sentences[i].lower().split())
                    words2 = set(sentences[i+1].lower().split())
                    sim = len(words1 & words2) / max(len(words1 | words2), 1)
                    similarities.append(sim)
                
                coherence_score = np.mean(similarities)
                coherence_variance = np.var(similarities)
            
            return {
                'coherence_score': float(coherence_score),
                'coherence_variance': float(coherence_variance)
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular coerência: {e}")
            return {"coherence_error": -1.0}
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calcula distância de Levenshtein entre duas strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def calculate_all_metrics(self, reference: str, generated: str) -> Dict[str, float]:
        """Calcula todas as métricas disponíveis"""
        all_metrics = {}
        
        # BLEU
        bleu_metrics = self.calculate_bleu(reference, generated)
        all_metrics.update(bleu_metrics)
        
        # ROUGE
        rouge_metrics = self.calculate_rouge(reference, generated)
        all_metrics.update(rouge_metrics)
        
        # Similaridade semântica
        semantic_metrics = self.calculate_semantic_similarity(reference, generated)
        all_metrics.update(semantic_metrics)
        
        # Métricas lexicais
        lexical_metrics = self.calculate_lexical_metrics(reference, generated)
        all_metrics.update(lexical_metrics)
        
        # Métricas de fluência
        fluency_metrics = self.calculate_fluency_metrics(generated)
        all_metrics.update({f'gen_{k}': v for k, v in fluency_metrics.items()})
        
        # Métricas de coerência
        coherence_metrics = self.calculate_coherence_score(generated)
        all_metrics.update(coherence_metrics)
        
        return all_metrics

class ModelEvaluator:
    """Avaliador de modelos de linguagem"""
    
    def __init__(self, results_dir: str = "data/evaluation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_calculator = MetricsCalculator()
        self.evaluation_history: List[EvaluationResult] = []
        self.comparison_history: List[ComparisonResult] = []
        
        self.load_history()
    
    def evaluate_response(self, model_name: str, prompt: str, 
                         reference: str, generated: str,
                         metadata: Optional[Dict] = None) -> EvaluationResult:
        """Avalia uma resposta do modelo"""
        
        # Calcular métricas
        metrics = self.metrics_calculator.calculate_all_metrics(reference, generated)
        
        # Criar resultado
        result = EvaluationResult(
            id=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_name=model_name,
            prompt=prompt,
            reference=reference,
            generated=generated,
            metrics=metrics,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        
        # Adicionar ao histórico
        self.evaluation_history.append(result)
        self.save_history()
        
        return result
    
    def compare_models(self, model_a: str, model_b: str, prompt: str,
                      response_a: str, response_b: str,
                      reference: Optional[str] = None) -> ComparisonResult:
        """Compara respostas de dois modelos"""
        
        # Calcular métricas para ambos os modelos
        if reference:
            metrics_a = self.metrics_calculator.calculate_all_metrics(reference, response_a)
            metrics_b = self.metrics_calculator.calculate_all_metrics(reference, response_b)
        else:
            # Sem referência, usar apenas métricas intrínsecas
            metrics_a = self.metrics_calculator.calculate_fluency_metrics(response_a)
            metrics_a.update(self.metrics_calculator.calculate_coherence_score(response_a))
            
            metrics_b = self.metrics_calculator.calculate_fluency_metrics(response_b)
            metrics_b.update(self.metrics_calculator.calculate_coherence_score(response_b))
        
        # Determinar vencedor baseado em métricas agregadas
        winner, confidence = self._determine_winner(metrics_a, metrics_b)
        
        # Criar resultado de comparação
        result = ComparisonResult(
            model_a=model_a,
            model_b=model_b,
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            metrics_a=metrics_a,
            metrics_b=metrics_b,
            winner=winner,
            confidence=confidence,
            timestamp=datetime.now()
        )
        
        # Adicionar ao histórico
        self.comparison_history.append(result)
        self.save_history()
        
        return result
    
    def _determine_winner(self, metrics_a: Dict[str, float], 
                         metrics_b: Dict[str, float]) -> Tuple[Optional[str], float]:
        """Determina o vencedor baseado nas métricas"""
        
        # Pesos para diferentes métricas (podem ser ajustados)
        weights = {
            'bleu_avg': 0.2,
            'rouge1_fmeasure': 0.15,
            'rouge2_fmeasure': 0.15,
            'rougeL_fmeasure': 0.15,
            'semantic_similarity': 0.2,
            'coherence_score': 0.1,
            'lexical_diversity': 0.05
        }
        
        score_a = 0.0
        score_b = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics_a and metric in metrics_b:
                # Normalizar métricas para [0, 1] se necessário
                val_a = max(0, min(1, metrics_a[metric]))
                val_b = max(0, min(1, metrics_b[metric]))
                
                score_a += val_a * weight
                score_b += val_b * weight
                total_weight += weight
        
        if total_weight == 0:
            return None, 0.0
        
        # Normalizar pontuações
        score_a /= total_weight
        score_b /= total_weight
        
        # Determinar vencedor
        if abs(score_a - score_b) < 0.05:  # Empate
            return None, abs(score_a - score_b)
        elif score_a > score_b:
            return "model_a", score_a - score_b
        else:
            return "model_b", score_b - score_a
    
    def generate_evaluation_report(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Gera relatório de avaliação"""
        
        # Filtrar resultados por modelo se especificado
        if model_name:
            results = [r for r in self.evaluation_history if r.model_name == model_name]
        else:
            results = self.evaluation_history
        
        if not results:
            return {"error": "Nenhum resultado encontrado"}
        
        # Calcular estatísticas
        all_metrics = {}
        for result in results:
            for metric, value in result.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                if isinstance(value, (int, float)) and value >= 0:  # Ignorar erros
                    all_metrics[metric].append(value)
        
        # Estatísticas por métrica
        metric_stats = {}
        for metric, values in all_metrics.items():
            if values:
                metric_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        # Tendências temporais
        df = pd.DataFrame([r.to_dict() for r in results])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        report = {
            'model_name': model_name or "All Models",
            'total_evaluations': len(results),
            'date_range': {
                'start': df['timestamp'].min().isoformat() if not df.empty else None,
                'end': df['timestamp'].max().isoformat() if not df.empty else None
            },
            'metric_statistics': metric_stats,
            'recent_performance': self._get_recent_performance(results),
            'recommendations': self._generate_recommendations(metric_stats)
        }
        
        return report
    
    def _get_recent_performance(self, results: List[EvaluationResult], 
                               days: int = 7) -> Dict[str, Any]:
        """Obtém performance recente"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_results = [r for r in results if r.timestamp >= cutoff_date]
        
        if not recent_results:
            return {"message": f"Nenhuma avaliação nos últimos {days} dias"}
        
        # Calcular métricas médias recentes
        recent_metrics = {}
        for result in recent_results:
            for metric, value in result.metrics.items():
                if isinstance(value, (int, float)) and value >= 0:
                    if metric not in recent_metrics:
                        recent_metrics[metric] = []
                    recent_metrics[metric].append(value)
        
        avg_metrics = {
            metric: np.mean(values) 
            for metric, values in recent_metrics.items() 
            if values
        }
        
        return {
            'period_days': days,
            'evaluation_count': len(recent_results),
            'average_metrics': avg_metrics
        }
    
    def _generate_recommendations(self, metric_stats: Dict[str, Dict]) -> List[str]:
        """Gera recomendações baseadas nas métricas"""
        recommendations = []
        
        # Verificar BLEU scores
        if 'bleu_avg' in metric_stats:
            bleu_avg = metric_stats['bleu_avg']['mean']
            if bleu_avg < 0.3:
                recommendations.append("BLEU score baixo - considere mais dados de treinamento ou ajuste de hiperparâmetros")
            elif bleu_avg > 0.7:
                recommendations.append("Excelente BLEU score - modelo bem alinhado com referências")
        
        # Verificar coerência
        if 'coherence_score' in metric_stats:
            coherence = metric_stats['coherence_score']['mean']
            if coherence < 0.5:
                recommendations.append("Baixa coerência - considere treinar com textos mais longos e estruturados")
        
        # Verificar diversidade lexical
        if 'gen_lexical_diversity' in metric_stats:
            diversity = metric_stats['gen_lexical_diversity']['mean']
            if diversity < 0.3:
                recommendations.append("Baixa diversidade lexical - modelo pode estar repetitivo")
            elif diversity > 0.8:
                recommendations.append("Alta diversidade lexical - boa variedade de vocabulário")
        
        # Verificar similaridade semântica
        if 'semantic_similarity' in metric_stats:
            similarity = metric_stats['semantic_similarity']['mean']
            if similarity < 0.5:
                recommendations.append("Baixa similaridade semântica - verificar alinhamento com objetivos")
        
        if not recommendations:
            recommendations.append("Performance geral satisfatória - continue monitorando")
        
        return recommendations
    
    def export_results(self, format: str = "json", 
                      model_name: Optional[str] = None) -> str:
        """Exporta resultados em formato especificado"""
        
        # Filtrar por modelo se especificado
        if model_name:
            results = [r for r in self.evaluation_history if r.model_name == model_name]
            comparisons = [c for c in self.comparison_history 
                          if model_name in [c.model_a, c.model_b]]
        else:
            results = self.evaluation_history
            comparisons = self.comparison_history
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_suffix = f"_{model_name}" if model_name else ""
        
        if format.lower() == "json":
            filename = f"evaluation_results{model_suffix}_{timestamp}.json"
            filepath = self.results_dir / filename
            
            data = {
                'evaluations': [r.to_dict() for r in results],
                'comparisons': [c.to_dict() for c in comparisons],
                'export_timestamp': datetime.now().isoformat(),
                'model_filter': model_name
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == "csv":
            filename = f"evaluation_results{model_suffix}_{timestamp}.csv"
            filepath = self.results_dir / filename
            
            # Converter para DataFrame
            df_data = []
            for result in results:
                row = {
                    'id': result.id,
                    'model_name': result.model_name,
                    'timestamp': result.timestamp.isoformat(),
                    'prompt_length': len(result.prompt),
                    'reference_length': len(result.reference),
                    'generated_length': len(result.generated)
                }
                row.update(result.metrics)
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(filepath, index=False, encoding='utf-8')
        
        else:
            raise ValueError(f"Formato não suportado: {format}")
        
        return str(filepath)
    
    def save_history(self):
        """Salva histórico em arquivo"""
        try:
            history_file = self.results_dir / "evaluation_history.json"
            
            data = {
                'evaluations': [r.to_dict() for r in self.evaluation_history],
                'comparisons': [c.to_dict() for c in self.comparison_history],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Erro ao salvar histórico: {e}")
    
    def load_history(self):
        """Carrega histórico de arquivo"""
        try:
            history_file = self.results_dir / "evaluation_history.json"
            
            if not history_file.exists():
                return
            
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruir avaliações
            for eval_data in data.get('evaluations', []):
                eval_data['timestamp'] = datetime.fromisoformat(eval_data['timestamp'])
                self.evaluation_history.append(EvaluationResult(**eval_data))
            
            # Reconstruir comparações
            for comp_data in data.get('comparisons', []):
                comp_data['timestamp'] = datetime.fromisoformat(comp_data['timestamp'])
                self.comparison_history.append(ComparisonResult(**comp_data))
            
            logger.info(f"Histórico carregado: {len(self.evaluation_history)} avaliações, {len(self.comparison_history)} comparações")
            
        except Exception as e:
            logger.error(f"Erro ao carregar histórico: {e}")
    
    def get_model_rankings(self) -> Dict[str, Any]:
        """Obtém ranking de modelos baseado em performance"""
        
        if not self.evaluation_history:
            return {"message": "Nenhuma avaliação disponível"}
        
        # Agrupar por modelo
        model_metrics = {}
        for result in self.evaluation_history:
            model = result.model_name
            if model not in model_metrics:
                model_metrics[model] = []
            
            # Calcular score agregado
            metrics = result.metrics
            score = 0.0
            count = 0
            
            for metric in ['bleu_avg', 'semantic_similarity', 'coherence_score']:
                if metric in metrics and isinstance(metrics[metric], (int, float)) and metrics[metric] >= 0:
                    score += metrics[metric]
                    count += 1
            
            if count > 0:
                model_metrics[model].append(score / count)
        
        # Calcular médias e ranking
        model_scores = {}
        for model, scores in model_metrics.items():
            if scores:
                model_scores[model] = {
                    'average_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'evaluation_count': len(scores)
                }
        
        # Ordenar por score médio
        ranked_models = sorted(
            model_scores.items(), 
            key=lambda x: x[1]['average_score'], 
            reverse=True
        )
        
        return {
            'rankings': ranked_models,
            'total_models': len(ranked_models),
            'evaluation_period': {
                'start': min(r.timestamp for r in self.evaluation_history).isoformat(),
                'end': max(r.timestamp for r in self.evaluation_history).isoformat()
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas gerais do avaliador"""
        
        total_evaluations = len(self.evaluation_history)
        total_comparisons = len(self.comparison_history)
        
        # Modelos únicos
        unique_models = set(r.model_name for r in self.evaluation_history)
        
        # Métricas disponíveis
        available_metrics = set()
        for result in self.evaluation_history:
            available_metrics.update(result.metrics.keys())
        
        return {
            'total_evaluations': total_evaluations,
            'total_comparisons': total_comparisons,
            'unique_models': len(unique_models),
            'model_names': list(unique_models),
            'available_metrics': list(available_metrics),
            'components_status': {
                'nltk': NLTK_AVAILABLE,
                'rouge': ROUGE_AVAILABLE,
                'sentence_transformers': SENTENCE_TRANSFORMERS_AVAILABLE
            }
        }

