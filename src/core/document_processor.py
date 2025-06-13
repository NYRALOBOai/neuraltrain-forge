#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuralTrain Forge - Document Processor
Sistema de processamento de documentos para chat com modelos

Funcionalidades:
- Processamento de PDF, TXT, MD, JSON, CSV
- Sistema de chunks inteligente
- Navegação entre partes do documento
- Extração de metadados
- Cache de documentos processados
- Suporte a documentos grandes (até 20MB)
"""

import os
import json
import hashlib
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import logging

# Bibliotecas para processamento de documentos
import pandas as pd
import PyPDF2
import pdfplumber
from markdown import markdown
from bs4 import BeautifulSoup
import tiktoken
import re

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Representa um chunk de documento"""
    id: str
    document_id: str
    chunk_index: int
    content: str
    metadata: Dict[str, Any]
    token_count: int
    start_position: int
    end_position: int
    
    def to_dict(self):
        return asdict(self)

@dataclass
class DocumentInfo:
    """Informações sobre um documento processado"""
    id: str
    filename: str
    file_path: str
    file_type: str
    file_size: int
    processed_at: datetime
    total_chunks: int
    total_tokens: int
    metadata: Dict[str, Any]
    
    def to_dict(self):
        data = asdict(self)
        data['processed_at'] = self.processed_at.isoformat()
        return data

class DocumentProcessor:
    """Processador principal de documentos"""
    
    def __init__(self, cache_dir: str = "data/document_cache", 
                 max_chunk_size: int = 1000, overlap_size: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
        # Inicializar tokenizer para contagem de tokens
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        except Exception as e:
            logger.warning(f"Erro ao carregar tokenizer: {e}")
            self.tokenizer = None
        
        # Cache de documentos processados
        self.document_cache: Dict[str, DocumentInfo] = {}
        self.chunk_cache: Dict[str, List[DocumentChunk]] = {}
        
        self.load_cache()
    
    def get_supported_formats(self) -> List[str]:
        """Retorna formatos suportados"""
        return ['.txt', '.pdf', '.md', '.json', '.csv']
    
    def is_supported_file(self, file_path: Union[str, Path]) -> bool:
        """Verifica se o arquivo é suportado"""
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.get_supported_formats()
    
    def process_document(self, file_path: Union[str, Path], 
                        force_reprocess: bool = False) -> Optional[DocumentInfo]:
        """Processa um documento e retorna informações"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"Arquivo não encontrado: {file_path}")
            return None
        
        if not self.is_supported_file(file_path):
            logger.error(f"Formato não suportado: {file_path.suffix}")
            return None
        
        # Verificar tamanho do arquivo (máximo 20MB)
        file_size = file_path.stat().st_size
        if file_size > 20 * 1024 * 1024:  # 20MB
            logger.error(f"Arquivo muito grande: {file_size / 1024 / 1024:.1f}MB (máximo 20MB)")
            return None
        
        # Gerar ID único baseado no conteúdo
        document_id = self._generate_document_id(file_path)
        
        # Verificar cache
        if not force_reprocess and document_id in self.document_cache:
            logger.info(f"Documento já processado: {file_path.name}")
            return self.document_cache[document_id]
        
        try:
            # Extrair texto do documento
            text_content, metadata = self._extract_text(file_path)
            
            if not text_content:
                logger.error(f"Não foi possível extrair texto de: {file_path}")
                return None
            
            # Criar chunks
            chunks = self._create_chunks(text_content, document_id)
            
            # Calcular tokens totais
            total_tokens = sum(chunk.token_count for chunk in chunks)
            
            # Criar informações do documento
            doc_info = DocumentInfo(
                id=document_id,
                filename=file_path.name,
                file_path=str(file_path),
                file_type=file_path.suffix.lower(),
                file_size=file_size,
                processed_at=datetime.now(),
                total_chunks=len(chunks),
                total_tokens=total_tokens,
                metadata=metadata
            )
            
            # Salvar no cache
            self.document_cache[document_id] = doc_info
            self.chunk_cache[document_id] = chunks
            self.save_cache()
            
            logger.info(f"Documento processado: {file_path.name} ({len(chunks)} chunks, {total_tokens} tokens)")
            return doc_info
            
        except Exception as e:
            logger.error(f"Erro ao processar documento {file_path}: {e}")
            return None
    
    def _generate_document_id(self, file_path: Path) -> str:
        """Gera ID único para o documento baseado no conteúdo"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Hash do conteúdo + nome do arquivo + tamanho
            hash_input = content + str(file_path.name).encode() + str(file_path.stat().st_size).encode()
            return hashlib.md5(hash_input).hexdigest()
        except Exception as e:
            logger.error(f"Erro ao gerar ID do documento: {e}")
            return hashlib.md5(str(file_path).encode()).hexdigest()
    
    def _extract_text(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extrai texto do arquivo baseado no tipo"""
        file_type = file_path.suffix.lower()
        metadata = {
            'filename': file_path.name,
            'file_type': file_type,
            'file_size': file_path.stat().st_size,
            'extraction_method': None
        }
        
        try:
            if file_type == '.txt':
                return self._extract_from_txt(file_path, metadata)
            elif file_type == '.pdf':
                return self._extract_from_pdf(file_path, metadata)
            elif file_type == '.md':
                return self._extract_from_markdown(file_path, metadata)
            elif file_type == '.json':
                return self._extract_from_json(file_path, metadata)
            elif file_type == '.csv':
                return self._extract_from_csv(file_path, metadata)
            else:
                raise ValueError(f"Tipo de arquivo não suportado: {file_type}")
                
        except Exception as e:
            logger.error(f"Erro na extração de texto: {e}")
            return "", metadata
    
    def _extract_from_txt(self, file_path: Path, metadata: Dict) -> Tuple[str, Dict]:
        """Extrai texto de arquivo TXT"""
        metadata['extraction_method'] = 'plain_text'
        
        # Tentar diferentes encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                metadata['encoding'] = encoding
                metadata['line_count'] = content.count('\n') + 1
                return content, metadata
            except UnicodeDecodeError:
                continue
        
        raise ValueError("Não foi possível decodificar o arquivo TXT")
    
    def _extract_from_pdf(self, file_path: Path, metadata: Dict) -> Tuple[str, Dict]:
        """Extrai texto de arquivo PDF"""
        text_content = ""
        
        try:
            # Tentar com pdfplumber primeiro (melhor para tabelas)
            with pdfplumber.open(file_path) as pdf:
                metadata['extraction_method'] = 'pdfplumber'
                metadata['page_count'] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n--- Página {page_num + 1} ---\n"
                        text_content += page_text + "\n"
                
                if text_content.strip():
                    return text_content, metadata
        
        except Exception as e:
            logger.warning(f"Erro com pdfplumber: {e}")
        
        try:
            # Fallback para PyPDF2
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                metadata['extraction_method'] = 'pypdf2'
                metadata['page_count'] = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n--- Página {page_num + 1} ---\n"
                        text_content += page_text + "\n"
                
                return text_content, metadata
                
        except Exception as e:
            logger.error(f"Erro ao extrair PDF: {e}")
            raise
    
    def _extract_from_markdown(self, file_path: Path, metadata: Dict) -> Tuple[str, Dict]:
        """Extrai texto de arquivo Markdown"""
        metadata['extraction_method'] = 'markdown'
        
        with open(file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Converter Markdown para HTML e depois para texto
        html = markdown(md_content)
        soup = BeautifulSoup(html, 'html.parser')
        text_content = soup.get_text()
        
        # Adicionar metadados específicos do Markdown
        metadata['has_headers'] = bool(re.search(r'^#+\s', md_content, re.MULTILINE))
        metadata['has_links'] = bool(re.search(r'\[.*?\]\(.*?\)', md_content))
        metadata['has_code_blocks'] = bool(re.search(r'```', md_content))
        
        return text_content, metadata
    
    def _extract_from_json(self, file_path: Path, metadata: Dict) -> Tuple[str, Dict]:
        """Extrai texto de arquivo JSON"""
        metadata['extraction_method'] = 'json'
        
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Converter JSON para texto legível
        text_content = self._json_to_text(json_data)
        
        metadata['json_structure'] = self._analyze_json_structure(json_data)
        
        return text_content, metadata
    
    def _extract_from_csv(self, file_path: Path, metadata: Dict) -> Tuple[str, Dict]:
        """Extrai texto de arquivo CSV"""
        metadata['extraction_method'] = 'csv'
        
        # Tentar diferentes separadores
        separators = [',', ';', '\t', '|']
        
        for sep in separators:
            try:
                df = pd.read_csv(file_path, separator=sep, encoding='utf-8')
                if len(df.columns) > 1:  # Separador correto encontrado
                    break
            except:
                continue
        else:
            # Fallback para separador padrão
            df = pd.read_csv(file_path, encoding='utf-8')
        
        # Converter DataFrame para texto
        text_content = f"Arquivo CSV: {file_path.name}\n"
        text_content += f"Colunas: {', '.join(df.columns)}\n"
        text_content += f"Número de linhas: {len(df)}\n\n"
        
        # Adicionar dados (limitado para evitar chunks muito grandes)
        max_rows = min(1000, len(df))
        text_content += df.head(max_rows).to_string(index=False)
        
        metadata['columns'] = list(df.columns)
        metadata['row_count'] = len(df)
        metadata['column_count'] = len(df.columns)
        metadata['separator'] = sep
        
        return text_content, metadata
    
    def _json_to_text(self, data: Any, level: int = 0) -> str:
        """Converte estrutura JSON para texto legível"""
        indent = "  " * level
        text = ""
        
        if isinstance(data, dict):
            for key, value in data.items():
                text += f"{indent}{key}:\n"
                text += self._json_to_text(value, level + 1)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                text += f"{indent}Item {i + 1}:\n"
                text += self._json_to_text(item, level + 1)
        else:
            text += f"{indent}{str(data)}\n"
        
        return text
    
    def _analyze_json_structure(self, data: Any) -> Dict[str, Any]:
        """Analisa estrutura do JSON"""
        if isinstance(data, dict):
            return {
                'type': 'object',
                'keys': list(data.keys()),
                'key_count': len(data)
            }
        elif isinstance(data, list):
            return {
                'type': 'array',
                'length': len(data),
                'item_types': list(set(type(item).__name__ for item in data))
            }
        else:
            return {
                'type': type(data).__name__,
                'value': str(data)[:100]  # Primeiros 100 caracteres
            }
    
    def _create_chunks(self, text: str, document_id: str) -> List[DocumentChunk]:
        """Cria chunks do texto"""
        chunks = []
        
        # Limpar e normalizar texto
        text = self._clean_text(text)
        
        if not text:
            return chunks
        
        # Dividir em sentenças primeiro
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        start_position = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            # Se a sentença sozinha é maior que o chunk máximo, dividir por palavras
            if sentence_tokens > self.max_chunk_size:
                # Salvar chunk atual se não estiver vazio
                if current_chunk:
                    chunk = self._create_chunk(
                        document_id, chunk_index, current_chunk,
                        current_tokens, start_position, start_position + len(current_chunk)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    start_position += len(current_chunk)
                    current_chunk = ""
                    current_tokens = 0
                
                # Dividir sentença longa em sub-chunks
                word_chunks = self._split_long_sentence(sentence, document_id, chunk_index, start_position)
                chunks.extend(word_chunks)
                chunk_index += len(word_chunks)
                start_position += len(sentence)
                
            # Se adicionar a sentença exceder o limite, finalizar chunk atual
            elif current_tokens + sentence_tokens > self.max_chunk_size:
                if current_chunk:
                    chunk = self._create_chunk(
                        document_id, chunk_index, current_chunk,
                        current_tokens, start_position, start_position + len(current_chunk)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Aplicar overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    start_position += len(current_chunk) - len(overlap_text)
                    current_chunk = overlap_text + " " + sentence
                    current_tokens = self._count_tokens(current_chunk)
                else:
                    current_chunk = sentence
                    current_tokens = sentence_tokens
            
            else:
                # Adicionar sentença ao chunk atual
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens
        
        # Adicionar último chunk se não estiver vazio
        if current_chunk:
            chunk = self._create_chunk(
                document_id, chunk_index, current_chunk,
                current_tokens, start_position, start_position + len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, document_id: str, chunk_index: int, content: str,
                     token_count: int, start_pos: int, end_pos: int) -> DocumentChunk:
        """Cria um objeto DocumentChunk"""
        chunk_id = f"{document_id}_chunk_{chunk_index}"
        
        return DocumentChunk(
            id=chunk_id,
            document_id=document_id,
            chunk_index=chunk_index,
            content=content.strip(),
            metadata={
                'word_count': len(content.split()),
                'char_count': len(content),
                'has_code': bool(re.search(r'```|`.*`', content)),
                'has_urls': bool(re.search(r'https?://', content)),
                'language': 'pt'  # Detectar idioma se necessário
            },
            token_count=token_count,
            start_position=start_pos,
            end_position=end_pos
        )
    
    def _clean_text(self, text: str) -> str:
        """Limpa e normaliza o texto"""
        # Remover caracteres de controle
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
        # Normalizar espaços em branco
        text = re.sub(r'\s+', ' ', text)
        
        # Remover linhas vazias excessivas
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Divide texto em sentenças"""
        # Padrão simples para divisão de sentenças
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_long_sentence(self, sentence: str, document_id: str, 
                           start_chunk_index: int, start_position: int) -> List[DocumentChunk]:
        """Divide sentença longa em chunks menores"""
        words = sentence.split()
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = start_chunk_index
        position = start_position
        
        for word in words:
            word_tokens = self._count_tokens(word)
            
            if current_tokens + word_tokens > self.max_chunk_size and current_chunk:
                # Finalizar chunk atual
                chunk = self._create_chunk(
                    document_id, chunk_index, current_chunk,
                    current_tokens, position, position + len(current_chunk)
                )
                chunks.append(chunk)
                chunk_index += 1
                position += len(current_chunk)
                
                # Iniciar novo chunk com overlap
                overlap_words = current_chunk.split()[-self.overlap_size//10:]  # Aproximadamente
                current_chunk = " ".join(overlap_words) + " " + word
                current_tokens = self._count_tokens(current_chunk)
            else:
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word
                current_tokens += word_tokens
        
        # Adicionar último chunk
        if current_chunk:
            chunk = self._create_chunk(
                document_id, chunk_index, current_chunk,
                current_tokens, position, position + len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Obtém texto de overlap para o próximo chunk"""
        words = text.split()
        if len(words) <= self.overlap_size:
            return text
        
        overlap_words = words[-self.overlap_size:]
        return " ".join(overlap_words)
    
    def _count_tokens(self, text: str) -> int:
        """Conta tokens no texto"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except:
                pass
        
        # Fallback: estimativa baseada em palavras
        return len(text.split()) * 1.3  # Aproximação
    
    def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Obtém chunks de um documento"""
        return self.chunk_cache.get(document_id, [])
    
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Obtém um chunk específico"""
        for chunks in self.chunk_cache.values():
            for chunk in chunks:
                if chunk.id == chunk_id:
                    return chunk
        return None
    
    def search_in_document(self, document_id: str, query: str, 
                          max_results: int = 5) -> List[DocumentChunk]:
        """Busca texto em um documento"""
        chunks = self.get_document_chunks(document_id)
        if not chunks:
            return []
        
        # Busca simples por palavras-chave
        query_words = query.lower().split()
        scored_chunks = []
        
        for chunk in chunks:
            content_lower = chunk.content.lower()
            score = 0
            
            for word in query_words:
                score += content_lower.count(word)
            
            if score > 0:
                scored_chunks.append((chunk, score))
        
        # Ordenar por relevância
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return [chunk for chunk, score in scored_chunks[:max_results]]
    
    def get_document_summary(self, document_id: str) -> Dict[str, Any]:
        """Obtém resumo de um documento"""
        if document_id not in self.document_cache:
            return {}
        
        doc_info = self.document_cache[document_id]
        chunks = self.get_document_chunks(document_id)
        
        return {
            'document_info': doc_info.to_dict(),
            'chunk_count': len(chunks),
            'total_tokens': sum(chunk.token_count for chunk in chunks),
            'avg_chunk_size': sum(chunk.token_count for chunk in chunks) / len(chunks) if chunks else 0,
            'content_preview': chunks[0].content[:200] + "..." if chunks else ""
        }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """Lista todos os documentos processados"""
        return [doc.to_dict() for doc in self.document_cache.values()]
    
    def delete_document(self, document_id: str) -> bool:
        """Remove documento do cache"""
        try:
            if document_id in self.document_cache:
                del self.document_cache[document_id]
            
            if document_id in self.chunk_cache:
                del self.chunk_cache[document_id]
            
            self.save_cache()
            return True
        except Exception as e:
            logger.error(f"Erro ao deletar documento: {e}")
            return False
    
    def save_cache(self):
        """Salva cache em arquivo"""
        try:
            cache_file = self.cache_dir / "document_cache.json"
            
            cache_data = {
                'documents': {doc_id: doc.to_dict() for doc_id, doc in self.document_cache.items()},
                'chunks': {doc_id: [chunk.to_dict() for chunk in chunks] 
                          for doc_id, chunks in self.chunk_cache.items()}
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Erro ao salvar cache: {e}")
    
    def load_cache(self):
        """Carrega cache de arquivo"""
        try:
            cache_file = self.cache_dir / "document_cache.json"
            
            if not cache_file.exists():
                return
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Reconstruir documentos
            for doc_id, doc_data in cache_data.get('documents', {}).items():
                doc_data['processed_at'] = datetime.fromisoformat(doc_data['processed_at'])
                self.document_cache[doc_id] = DocumentInfo(**doc_data)
            
            # Reconstruir chunks
            for doc_id, chunks_data in cache_data.get('chunks', {}).items():
                chunks = []
                for chunk_data in chunks_data:
                    chunks.append(DocumentChunk(**chunk_data))
                self.chunk_cache[doc_id] = chunks
                
            logger.info(f"Cache carregado: {len(self.document_cache)} documentos")
            
        except Exception as e:
            logger.error(f"Erro ao carregar cache: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas do processador"""
        total_documents = len(self.document_cache)
        total_chunks = sum(len(chunks) for chunks in self.chunk_cache.values())
        total_tokens = sum(doc.total_tokens for doc in self.document_cache.values())
        
        file_types = {}
        for doc in self.document_cache.values():
            file_type = doc.file_type
            if file_type not in file_types:
                file_types[file_type] = 0
            file_types[file_type] += 1
        
        return {
            'total_documents': total_documents,
            'total_chunks': total_chunks,
            'total_tokens': total_tokens,
            'file_types': file_types,
            'avg_chunks_per_document': total_chunks / total_documents if total_documents > 0 else 0,
            'avg_tokens_per_document': total_tokens / total_documents if total_documents > 0 else 0,
            'cache_size_mb': self._get_cache_size()
        }
    
    def _get_cache_size(self) -> float:
        """Calcula tamanho do cache em MB"""
        try:
            cache_file = self.cache_dir / "document_cache.json"
            if cache_file.exists():
                return cache_file.stat().st_size / 1024 / 1024
            return 0.0
        except:
            return 0.0

